"""End-to-end embed -> detect round-trip. Needs the VideoSeal model + weights, so it is
marked `gpu` (deselected in CI) but runs on CPU/MPS locally. Uses real natural images:
the ConvNeXt extractor is trained on natural content and scores ~chance on synthetic noise.
"""

import json
import os
import subprocess

import numpy as np
import pytest

pytest.importorskip("videoseal")
pytest.importorskip("skimage")

from veed_videoseal.cli import main as cli_main  # noqa: E402
from veed_videoseal.constants import SCALING_W  # noqa: E402
from veed_videoseal.detect import detect_watermark  # noqa: E402
from veed_videoseal.embed import embed_watermark, embed_watermark_tensor  # noqa: E402
from veed_videoseal.model import load_model  # noqa: E402
from veed_videoseal.sign import (  # noqa: E402
    _build_rotate_remux_cmd, _probe_rotation, _probe_sar, sign_video)
from veed_videoseal.video_io import read_video_frames  # noqa: E402

# These are deselected in CI: GitHub-hosted runners have no accelerator and no model weights, and
# this repo does not stand up GPU CI. Run them locally on CPU/MPS/CUDA before changing embed,
# detect, or the sign pipeline (rotation, SAR and audio preservation included): `pytest -m gpu`.
pytestmark = pytest.mark.gpu


def _encode_h264(frames: np.ndarray, path: str, crf: int) -> None:
    """Encode (T,H,W,3) uint8 frames to H.264 mp4 via ffmpeg (self-contained)."""
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0


def _real_frames() -> np.ndarray:
    from skimage import data
    from skimage.transform import resize

    imgs = [data.astronaut(), data.chelsea(), data.coffee(), data.rocket()]
    frames = [(resize(im, (256, 256), anti_aliasing=True) * 255).astype(np.uint8) for im in imgs]
    return np.stack(frames * 2)  # (8, 256, 256, 3) uint8


def test_embed_then_detect_recovers_marker():
    frames = _real_frames()
    wm = embed_watermark(frames, device="cpu").frames
    assert wm.shape == frames.shape
    assert wm.dtype == np.uint8

    res = detect_watermark(wm, device="cpu")
    assert res["detected"] is True
    assert res["bit_accuracy"] > 0.9


def test_clean_video_is_not_detected():
    res = detect_watermark(_real_frames(), device="cpu")
    assert res["detected"] is False
    assert res["bit_accuracy"] < 0.75


def test_chunked_detect_matches_extract_message():
    """The equivalence the stubbed unit tests can't show: on the real model, detect_watermark's
    chunked model.detect accumulation recovers the same message — same bit_accuracy for any
    chunk_frames — as videoseal's single-batch extract_message('avg') (the code this replaced).

    Uses a watermarked clip so every bit carries strong signal (logits far from 0), the regime
    where floating-point summation order can't flip a bit, making the match exact rather than
    approximate.
    """
    import torch

    from veed_videoseal._frames import frames_to_chw01
    from veed_videoseal.constants import WATERMARK_TAG
    from veed_videoseal.message import bit_accuracy, tag_to_bits
    from veed_videoseal.model import model_nbits

    wm = embed_watermark(_real_frames(), device="cpu").frames

    # Reference: the exact single-batch call detect_watermark used to make.
    model = load_model("cpu")
    nbits = model_nbits(model)
    with torch.no_grad():
        reference = model.extract_message(frames_to_chw01(wm, "cpu")).squeeze()[:nbits]
    reference_acc = bit_accuracy(reference.int().cpu().numpy(), tag_to_bits(WATERMARK_TAG, nbits))
    assert reference_acc > 0.9  # a real watermark: strong signal, no bits near the 0 boundary

    for chunk in (1, 3, 8, 100):
        res = detect_watermark(wm, device="cpu", chunk_frames=chunk)
        assert res["bit_accuracy"] == pytest.approx(reference_acc), f"chunk_frames={chunk} diverged"


def test_loader_applies_scaling_w():
    model = load_model("cpu")
    assert model.blender.scaling_w == SCALING_W


def test_embed_tensor_matches_numpy_embed():
    """On the real model, a SINGLE-SHOT numpy embed produces the same watermarked pixels as the
    on-GPU tensor path (it's the same core), and the result is detectable — proving embed_watermark
    is a faithful wrapper and the tensor primitive is safe for the pipeline to adopt.

    Bit-equality only holds within one chunk: pass chunk_frames=len(frames) so embed_watermark does
    a single-shot embed like embed_watermark_tensor. Multi-chunk embed re-embeds per chunk and
    diverges at boundaries — that path is covered by test_chunked_embed_stays_detectable."""
    from veed_videoseal._frames import chw01_to_frames, frames_to_chw01

    frames = _real_frames()
    wm_tensor = embed_watermark_tensor(frames_to_chw01(frames, "cpu"))
    tensor_frames = chw01_to_frames(wm_tensor).numpy()

    numpy_frames = embed_watermark(frames, device="cpu", chunk_frames=len(frames)).frames  # single shot
    np.testing.assert_array_equal(tensor_frames, numpy_frames)  # bit-identical within one chunk

    assert detect_watermark(tensor_frames, device="cpu")["detected"] is True


def test_chunked_embed_stays_detectable():
    """Multi-chunk embed genuinely re-embeds each chunk (loop runs 3x for 8 frames @ chunk_frames=3),
    so boundary pixels differ from a single-shot embed — hence NO bit-equality assert. The invariant
    that matters, and that sign_video relies on, is that the fixed marker is still recoverable from
    the stitched output. Also guards the chunk loop's stitching (wrong order/gaps -> not detected)."""
    frames = _real_frames()  # 8 frames

    single = embed_watermark(frames, device="cpu", chunk_frames=len(frames)).frames
    multi = embed_watermark(frames, device="cpu", chunk_frames=3).frames  # 3 + 3 + 2

    assert multi.shape == frames.shape
    assert not np.array_equal(multi, single)  # chunk boundaries at 3,6 (unaligned to step_size 4) differ
    assert detect_watermark(multi, device="cpu")["detected"] is True


def _real_frames_480p(n: int = 8) -> np.ndarray:
    from skimage import data
    from skimage.transform import resize

    imgs = [data.astronaut(), data.chelsea(), data.coffee(), data.rocket()]
    base = [(resize(im, (480, 854), anti_aliasing=True) * 255).astype(np.uint8) for im in imgs]
    return np.stack([base[i % 4] for i in range(n)])


def test_survives_realistic_reencode(tmp_path):
    """The headline guarantee: at 480p the watermark survives H.264 crf 28 re-encode."""
    wm = embed_watermark(_real_frames_480p(), device="cpu").frames
    out = str(tmp_path / "wm_crf28.mp4")
    _encode_h264(wm, out, crf=28)
    res = detect_watermark(read_video_frames(out), device="cpu")
    assert res["detected"] is True
    assert res["bit_accuracy"] > 0.85


def _encode_with_audio(frames: np.ndarray, path: str, fps: int = 25) -> None:
    """Encode (T,H,W,3) uint8 frames to mp4 with a generated AAC audio track.

    Gives `sign` a real, arbitrary input video (clean, with audio) to round-trip.
    """
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0


def _stream_count(path: str, kind: str) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=index", "-of", "csv=p=0", path]
    ).decode().strip()
    return len([line for line in out.splitlines() if line])


def test_sign_video_preserves_audio_and_detects(tmp_path):
    """sign_video on an arbitrary clip with audio: output keeps audio/dims and verifies."""
    frames = _real_frames_480p(n=16)
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_with_audio(frames, src, fps=25)

    result = sign_video(src, out, device="cpu", chunk_frames=4)
    assert result["out"] == out
    assert result["frames"] > 0
    assert set(result) == {"out", "frames", "gpu_compute_ms"}

    # audio survived, video dimensions preserved
    assert _stream_count(out, "a") == 1
    signed = read_video_frames(out)
    assert signed.shape[1:3] == frames.shape[1:3]

    res = detect_watermark(signed, device="cpu")
    assert res["detected"] is True
    assert res["bit_accuracy"] > 0.85


def test_sign_then_verify_cli_json(tmp_path, capsys):
    """The CLI contract: `sign --json` then `verify --json` on arbitrary video."""
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_with_audio(_real_frames_480p(n=12), src, fps=25)

    assert cli_main(["sign", "--video", src, "--out", out, "--device", "cpu", "--json"]) == 0
    sign_out = json.loads(capsys.readouterr().out.strip())
    assert sign_out["out"] == out and sign_out["frames"] > 0

    assert cli_main(["verify", "--video", out, "--device", "cpu", "--json"]) == 0
    verify_out = json.loads(capsys.readouterr().out.strip())
    assert verify_out["detected"] is True
    assert verify_out["metadata_present"] is True
    assert verify_out["nbits"] > 0


def _encode_short_audio(frames: np.ndarray, path: str, fps: int = 25, audio_seconds: float = 0.1) -> None:
    """Encode frames with an audio track deliberately SHORTER than the video (no -shortest),
    reproducing the common 'audio ends before video' case that used to break the sign pipe."""
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={audio_seconds}",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0


def _encode_yuv444(frames: np.ndarray, path: str, fps: int = 25) -> None:
    """Encode frames at their (possibly odd) dimensions using yuv444p, which — unlike yuv420p —
    permits odd width/height, so we can build an odd-dimensioned source to sign."""
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv444p", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0


def _real_frames_odd(n: int = 12) -> np.ndarray:
    from skimage import data
    from skimage.transform import resize

    imgs = [data.astronaut(), data.chelsea(), data.coffee(), data.rocket()]
    base = [(resize(im, (271, 481), anti_aliasing=True) * 255).astype(np.uint8) for im in imgs]
    return np.stack([base[i % 4] for i in range(n)])


def test_sign_video_no_audio(tmp_path):
    """A video with no audio stream signs cleanly (the audio-mux branch is skipped)."""
    frames = _real_frames_480p(n=12)
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_h264(frames, src, crf=18)  # video only, no audio

    result = sign_video(src, out, device="cpu", chunk_frames=4)
    assert result["frames"] == 12
    assert _stream_count(out, "a") == 0
    assert detect_watermark(read_video_frames(out), device="cpu")["detected"] is True


def _audio_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
         "-of", "csv=p=0", path]
    ).decode().strip()
    return float(out)


def test_sign_video_audio_shorter_than_video(tmp_path):
    """Audio shorter than video must NOT truncate the video, and (no apad) must be preserved short
    rather than padded with silence to the video length."""
    frames = _real_frames_480p(n=24)  # ~1s of video at 25fps
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_short_audio(frames, src, fps=25, audio_seconds=0.1)  # 0.1s audio vs ~1s video

    result = sign_video(src, out, device="cpu", chunk_frames=4)
    assert result["frames"] == 24  # every video frame signed despite the short audio
    assert _stream_count(out, "a") == 1
    assert _audio_duration(out) < 0.5  # preserved short, NOT padded up to the ~1s video length
    assert detect_watermark(read_video_frames(out), device="cpu")["detected"] is True


def test_sign_video_odd_dimensions(tmp_path):
    """Regression: odd-dimensioned input is cropped to even and encodes (yuv420p needs even)."""
    frames = _real_frames_odd(n=12)  # 271x481 — both odd
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_yuv444(frames, src, fps=25)

    result = sign_video(src, out, device="cpu", chunk_frames=4)
    assert result["frames"] == 12
    signed = read_video_frames(out)
    assert signed.shape[1] % 2 == 0 and signed.shape[2] % 2 == 0  # cropped to even


def _encode_rotated(frames: np.ndarray, path: str, rotate: int, fps: int = 25) -> None:
    """Encode frames with a real display-matrix rotation, so the source plays rotated (as a
    phone portrait clip stored landscape does). A plain encode can't carry the matrix (-metadata
    rotate= is a no-op on ffmpeg >=7), so we stamp it in a -c copy pass — the same mechanism the
    signer uses — which makes _probe_rotation(path) == rotate."""
    _, h, w, _ = frames.shape
    pre = path + ".pre.mp4"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", pre,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0
    r = subprocess.run(_build_rotate_remux_cmd(pre, rotate, path))
    os.remove(pre)
    assert r.returncode == 0


def _encode_long_audio(frames: np.ndarray, path: str, fps: int = 25, audio_seconds: float = 5.0) -> None:
    """Encode frames with an audio track deliberately LONGER than the video (no -shortest),
    reproducing the 'audio outlasts video' case that used to leave a trailing audio tail."""
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={audio_seconds}",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0


def test_sign_preserves_rotation(tmp_path):
    """A rotated (phone-portrait-style) source keeps its display rotation through signing, so
    the signed output isn't played sideways."""
    frames = _real_frames_480p(n=12)
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_rotated(frames, src, rotate=90, fps=25)

    r_src = _probe_rotation(src)
    assert r_src != 0  # sanity: the source really carries a rotation

    sign_video(src, out, device="cpu", chunk_frames=4)
    assert _probe_rotation(out) == r_src  # rotation carried across (was dropped before the fix)


def _encode_anamorphic(frames: np.ndarray, path: str, sar: str = "40/33", fps: int = 25) -> None:
    """Encode frames with a non-square sample aspect ratio (setsar), so the source stores square
    dimensions but displays anamorphically (as DVD/HDV content does). max=1000000 overrides setsar's
    default max=100 so a SAR with a component >100 (e.g. 159:100) is stored exactly, not clamped."""
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-vf", f"setsar={sar}:max=1000000",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    assert p.returncode == 0


def test_sign_preserves_sar(tmp_path):
    """An anamorphic (non-square-pixel) source keeps its sample aspect ratio through signing, so
    the signed output isn't played squished/stretched (SAR was dropped by the rawvideo pipe).

    Uses 159:100 (a component >100) on purpose: it exercises the setsar max=100 default that would
    otherwise re-rationalize the SAR into a different, distorted ratio."""
    frames = _real_frames_480p(n=12)
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_anamorphic(frames, src, sar="159/100", fps=25)

    sar_src = _probe_sar(src)
    assert sar_src == "159/100"  # sanity: the source really carries a non-square SAR

    sign_video(src, out, device="cpu", chunk_frames=4)
    assert _probe_sar(out) == sar_src  # SAR carried across exactly (no max=100 clamp distortion)


def test_sign_video_audio_longer_than_video(tmp_path):
    """Audio longer than the video is PRESERVED, not clamped: the signer carries the source audio
    across as-is (no -shortest), so a music-over-stills clip keeps its full soundtrack."""
    frames = _real_frames_480p(n=12)  # ~0.48s of video at 25fps
    src = str(tmp_path / "src.mp4")
    out = str(tmp_path / "signed.mp4")
    _encode_long_audio(frames, src, fps=25, audio_seconds=5.0)  # 5s audio vs ~0.48s video

    sign_video(src, out, device="cpu", chunk_frames=4)
    assert _stream_count(out, "a") == 1
    assert _audio_duration(out) > 4.0  # full ~5s soundtrack kept, not clamped to the ~0.48s video
    assert detect_watermark(read_video_frames(out), device="cpu")["detected"] is True
