"""Sign an arbitrary video: embed the VEED watermark on every frame while preserving
the original audio, frame rate, orientation, sample aspect ratio and (best-effort) duration.

The pipeline-side embed (``embed_watermark``) only transforms an in-memory frame array.
Signing a real, arbitrary-length video additionally has to decode the *whole* video
(``video_io.read_video_frames`` deliberately caps at a handful of frames for detection),
re-encode it, and carry the original audio + container metadata into the output. We stream
through ffmpeg pipes in fixed-size chunks so a long or high-resolution clip never has to fit
in memory (host or GPU) at once. The fixed marker is embedded identically on every chunk, so
chunking does not change what the verifier recovers.
"""

import json
import os
import subprocess
import sys

import numpy as np

from ._heartbeat import Heartbeat
from .constants import METADATA_TAGS

# NOTE: `embed_watermark` (and thus torch) is imported lazily inside sign_video so that the
# pure ffmpeg helpers here — _probe_video, _probe_rotation, _build_encode_cmd — and their
# unit tests don't pull in torch just to inspect a command list.

# Frames per GPU embed call. Bounds host/GPU memory on long or high-res inputs; because the
# embedded message is the same fixed marker on every frame, the chunk size is purely a
# memory/throughput knob and has no effect on detection.
DEFAULT_CHUNK_FRAMES = 16


def _valid_rate(rate: str | None) -> str | None:
    """Return ``rate`` (an ffprobe ``num/den`` string) verbatim if it denotes a positive
    frame rate, else None. Kept as the exact rational so ffmpeg gets e.g. ``30000/1001``
    rather than a lossy float, avoiding cumulative A/V drift on NTSC-style rates."""
    if not rate or "/" not in rate:
        return None
    num, den = rate.split("/", 1)
    try:
        if float(num) <= 0 or float(den) == 0:
            return None
    except ValueError:
        return None
    return rate


def _valid_sar(sar: str | None) -> str | None:
    """Normalize an ffprobe ``sample_aspect_ratio`` (a ``num:den`` string, e.g. ``"40:33"``) to the
    ``num/den`` form the ffmpeg ``setsar`` filter wants, or None when the SAR is undefined or square.

    ffprobe reports ``"1:1"`` for square pixels, ``"0:1"`` / ``"N/A"`` / nothing when unknown; all of
    those (and any ``num==den``) return None so the common square-pixel case adds no filter. Parsed
    defensively (like _valid_rate): a non-numeric field or zero denominator is treated as unknown."""
    if not sar or ":" not in sar:
        return None
    num, den = sar.split(":", 1)
    try:
        num_f, den_f = float(num), float(den)
    except ValueError:
        return None
    if num_f <= 0 or den_f <= 0 or num_f == den_f:
        return None
    return f"{num}/{den}"


def _probe_video(path: str) -> tuple[int, int, str]:
    """Return (width, height, frame_rate_string) of the first video stream via ffprobe.

    Prefers ``avg_frame_rate`` (true average over the file) and falls back to
    ``r_frame_rate`` (the base/most-common rate), then to a literal ``"25"``. Raises
    ValueError if the input has no video stream (or a stream missing its dimensions)
    rather than crashing with an IndexError/KeyError.
    """
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", path]
    ).decode()
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise ValueError(f"no video stream found in {path}")
    stream = streams[0]
    if stream.get("width") is None or stream.get("height") is None:
        raise ValueError(f"video stream in {path} has no width/height")
    width, height = int(stream["width"]), int(stream["height"])
    rate = _valid_rate(stream.get("avg_frame_rate")) or _valid_rate(stream.get("r_frame_rate")) or "25"
    return width, height, rate


def _probe_rotation(path: str) -> int:
    """Return the first video stream's display rotation in the ``rotate``-tag convention
    (a portrait phone clip stored landscape reads as a non-zero multiple of 90), normalized
    to [0, 360). 0 when there is none.

    Prefers the display-matrix side data (modern ffmpeg) over the legacy ``rotate`` tag.
    ffprobe reports the side-data angle with the opposite sign to the ``rotate`` tag, so it
    is negated here; that keeps read↔write idempotent: writing ``-metadata rotate=R`` stores
    side data ``-R``, which this reads back as ``R``.
    """
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream_side_data=rotation:stream_tags=rotate", "-of", "json", path]
    ).decode()
    streams = json.loads(out).get("streams", [])
    if not streams:
        return 0
    stream = streams[0]
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return int(-float(side_data["rotation"])) % 360
    tag = stream.get("tags", {}).get("rotate")
    if tag is not None:
        return int(float(tag)) % 360
    return 0


def _probe_sar(path: str) -> str | None:
    """Return the first video stream's sample aspect ratio as a ``num/den`` string for ffmpeg's
    ``setsar`` filter, or None for square/undefined SAR. Decoding to rawvideo drops the SAR (like
    rotation), so it must be re-stamped on the output or anamorphic sources play squished."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=sample_aspect_ratio", "-of", "json", path]
    ).decode()
    streams = json.loads(out).get("streams", [])
    if not streams:
        return None
    return _valid_sar(streams[0].get("sample_aspect_ratio"))


# Audio codecs that stream-copy cleanly into an mp4 container, so we can carry the source audio
# across untouched. Anything else (opus, vorbis, pcm_*, flac, ...) is re-encoded to AAC because the
# mp4 muxer rejects or mishandles a raw copy of it.
_MP4_COPYABLE_AUDIO = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})


def _audio_codecs(path: str) -> list[str]:
    """Return every audio stream's ``codec_name`` (e.g. ``["aac"]``), or ``[]`` when the source has
    no audio. Used to decide whether the audio can be stream-copied into the mp4 or must be
    re-encoded to AAC. One line per audio stream, so a multi-track source is fully described; a
    present-but-unnamed stream yields an empty-string entry (not copyable → re-encoded, not dropped)."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", path]
    ).decode()
    return out.splitlines()  # [] when no audio; one (possibly empty) codec_name per audio stream


def _build_encode_cmd(width: int, height: int, rate: str, audio_codecs: list[str], in_path: str,
                      out_path: str, sar: str | None = None, faststart: bool = True) -> list[str]:
    """Build the ffmpeg encoder command for the signed output.

    Input 0 is our rawvideo pipe; when the source has audio (``audio_codecs`` non-empty), input 1 is
    the original file so its audio and container metadata can be carried across. Kept a pure
    function of its args (no I/O) so the mapping/SAR/audio logic is unit-testable without a real
    encode.

    Audio is carried through untouched (``-c:a copy``) when EVERY audio stream's codec fits mp4, else
    all streams are re-encoded to AAC — the copy/re-encode choice covers all mapped tracks, so a
    mixed-codec source is never copied into a container that can't hold one of its tracks. We
    deliberately do NOT pad or clamp it to the video length: the source audio is preserved as-is, so
    a signed clip keeps exactly the audio it started with.

    ``faststart`` moves the moov atom to the front for progressive playback; the caller disables it
    for a temp encode that a later ``-c copy`` remux (which re-faststarts) will consume.

    Display rotation is NOT applied here: -metadata rotate= is a silent no-op on ffmpeg >=7, so
    rotation is stamped by a separate -c copy remux pass (see _build_rotate_remux_cmd).
    """
    has_audio = bool(audio_codecs)
    copy_audio = has_audio and all(codec in _MP4_COPYABLE_AUDIO for codec in audio_codecs)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", rate, "-i", "-"]
    if has_audio:
        cmd += ["-i", in_path]
    # Build the -vf chain (ffmpeg allows only one). yuv420p needs even dimensions, so crop a stray
    # odd row/column (only when actually odd, so the common even case incurs no resampling). setsar
    # re-stamps the source sample aspect ratio: the rawvideo pipe carries none, so without it an
    # anamorphic (non-square-pixel) source would be re-encoded as 1:1 and play squished/stretched.
    # max=1000000 overrides setsar's default max=100, which would otherwise re-rationalize SARs with
    # a component >100 (e.g. the standard H.264 160:99, 159:100) into a *different*, distorted ratio.
    filters = []
    even_w, even_h = width - (width % 2), height - (height % 2)
    if (even_w, even_h) != (width, height):
        filters.append(f"crop={even_w}:{even_h}")
    if sar:
        filters.append(f"setsar={sar}:max=1000000")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a?", "-map_metadata", "1"]  # all audio streams + source tags
    cmd += ["-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    if has_audio:
        # Preserve the source audio: stream-copy when every track's codec fits mp4, else re-encode
        # all to AAC (-c:a copy into mp4 fails for opus/vorbis/pcm from webm/ogg/wav/mov). No
        # apad/-shortest: we don't pad or clamp to the video length, so audio is carried as-is.
        if copy_audio:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
    # Stamp the marker tags last so they override any same-named source tag from -map_metadata.
    for key, value in METADATA_TAGS.items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd.append(out_path)
    return cmd


def _build_rotate_remux_cmd(in_path: str, rotation: int, out_path: str) -> list[str]:
    """Build the ffmpeg command that stamps the display rotation onto an already-encoded file via a
    stream-copy remux (no re-encode). Pure function of its args so it is unit-testable.

    ``-metadata:s:v:0 rotate=`` no longer writes a display matrix on ffmpeg >=7 (tested against
    8.0.1), so rotation must be applied through the ``-display_rotation`` *input*
    option instead — which only sticks with ``-c copy`` (a re-encode drops the matrix). The angle is
    negated because ``-display_rotation`` is counter-clockwise while the ``rotate`` tag / _probe_rotation
    convention is clockwise; ``-display_rotation -R`` therefore reads back as ``rotation == R``,
    keeping the round-trip idempotent."""
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-display_rotation", str(-rotation), "-i", in_path,
            "-map", "0", "-c", "copy", "-movflags", "+faststart", out_path]


def _remove_quietly(path: str) -> None:
    """Remove a file if present, ignoring a missing file (so failure cleanup never masks the real
    error)."""
    try:
        os.remove(path)
    except OSError:
        pass


def _close_quietly(stream) -> None:
    """Close a pipe end, swallowing the BrokenPipeError/ValueError that arises when the peer
    process has already died (so teardown never masks the real error or skips reaping)."""
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def sign_video(in_path: str, out_path: str, device: str = "auto", ckpt_dir=None,
               chunk_frames: int = DEFAULT_CHUNK_FRAMES) -> dict:
    """Watermark every frame of ``in_path`` and write the signed video to ``out_path``.

    Preserves frame rate, display rotation, sample aspect ratio, and carries the source audio
    across untouched (stream-copied when its codec fits mp4, else re-encoded to AAC; never padded
    or clamped to the video length), carries source container metadata, and stamps the marker
    ``METADATA_TAGS``. Odd dimensions are
    cropped to even (yuv420p requires even width/height). Returns
    ``{"out": str, "frames": int, "gpu_compute_ms": float|None}`` (GPU time is the summed
    per-chunk embed time, or None when not running on CUDA).

    Note: variable-frame-rate sources are re-timed to a constant ``avg_frame_rate`` (rawvideo
    has no per-frame timestamps); duration is preserved to within the average-rate rounding.
    Subtitle/data streams are not carried (mp4 codec constraints); video + all audio are.
    """
    width, height, rate = _probe_video(in_path)
    audio_codecs = _audio_codecs(in_path)
    rotation = _probe_rotation(in_path)
    sar = _probe_sar(in_path)

    # Decoder: full video (no frame cap), same options as read_video_frames so rotated inputs
    # aren't transposed and a stray cover-art stream isn't picked. Spawned before the encoder;
    # if the encoder fails to spawn we must still reap this one (handled below).
    decode = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-noautorotate", "-i", in_path,
         "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )

    # When the source is rotated we encode to a temp file and stamp the display matrix in a second
    # -c copy pass (rotation can't be written during the libx264 encode on ffmpeg >=7). Otherwise
    # we encode straight to out_path. The final out_path therefore only ever appears complete.
    encode_target = f"{out_path}.prerotate.mp4" if rotation else out_path
    # Skip faststart on the temp encode: the rotation remux (-c copy) re-faststarts, so doing it
    # here too would be a wasted moov-atom relocation over the whole encoded file.
    encode_cmd = _build_encode_cmd(width, height, rate, audio_codecs, in_path, encode_target,
                                   sar=sar, faststart=not rotation)
    try:
        encode = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)
    except BaseException:
        decode.kill()
        decode.wait()
        raise

    # Deferred (function-scope) import of the torch-backed embedder: keeps this module and its
    # pure ffmpeg helpers (_probe_video, _build_encode_cmd, ...) importable without torch, while
    # not re-importing per chunk.
    from .embed import embed_watermark

    frame_size = width * height * 3
    read_size = frame_size * chunk_frames
    total_frames = 0
    gpu_ms_total = 0.0
    gpu_measured = False
    leftover = b""
    # Heartbeat before the first (model-loading) chunk: a caller may kill a child that emits no
    # output for too long, and model load + first inference are otherwise silent. Progress goes
    # to stderr, never stdout (which is reserved for the single final JSON object).
    print("sign: starting", file=sys.stderr, flush=True)
    signed_ok = False
    try:
        try:
            with Heartbeat("sign: loading model / signing"):
                while True:
                    data = decode.stdout.read(read_size)
                    if not data:
                        break
                    leftover += data
                    whole = len(leftover) // frame_size
                    if whole == 0:
                        continue  # partial frame straddling a read boundary; wait for the rest
                    usable, leftover = leftover[: whole * frame_size], leftover[whole * frame_size:]
                    frames = np.frombuffer(usable, dtype=np.uint8).reshape(whole, height, width, 3).copy()
                    result = embed_watermark(frames, device=device, ckpt_dir=ckpt_dir)
                    try:
                        encode.stdin.write(result.frames.tobytes())
                    except BrokenPipeError:
                        break  # encoder died; the returncode check below reports it cleanly
                    total_frames += whole
                    if result.gpu_compute_ms is not None:
                        gpu_ms_total += result.gpu_compute_ms
                        gpu_measured = True
                    print(f"sign: signed {total_frames} frames", file=sys.stderr, flush=True)
        finally:
            # Close pipe ends quietly (a dead peer makes close() raise) and always reap both
            # children so a mid-stream failure never leaks ffmpeg processes.
            _close_quietly(decode.stdout)
            _close_quietly(encode.stdin)
            decode.wait()
            encode.wait()

        if decode.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg decode of {in_path} failed (exit {decode.returncode})")
        if encode.returncode != 0:
            raise RuntimeError(f"ffmpeg encode to {out_path} failed (exit {encode.returncode})")
        if total_frames == 0:
            raise ValueError(f"decoded no video frames from {in_path}")

        if rotation:
            # Second pass: stamp the display matrix that the libx264 encode can't write on ffmpeg >=7.
            remux = subprocess.run(_build_rotate_remux_cmd(encode_target, rotation, out_path))
            if remux.returncode != 0:
                raise RuntimeError(f"ffmpeg rotation remux to {out_path} failed (exit {remux.returncode})")
            _remove_quietly(encode_target)  # temp consumed by the remux
        signed_ok = True
    finally:
        # On any failure (including an exception raised mid-stream, e.g. from embed_watermark)
        # remove every artifact so a failed sign never leaves a temp or a stale/partial out_path.
        if not signed_ok:
            _remove_quietly(encode_target)
            if encode_target != out_path:
                _remove_quietly(out_path)

    return {"out": out_path, "frames": total_frames,
            "gpu_compute_ms": gpu_ms_total if gpu_measured else None}
