"""Self-contained ffmpeg/ffprobe video I/O for the verifier (no dependency on any calling pipeline)."""

import json
import logging
import subprocess

import numpy as np

log = logging.getLogger(__name__)

# Cap frames decoded for verification. Detection averages over frames and saturates well
# under this (a dozen frames already clear the threshold), so this only bounds memory on
# long inputs without affecting the verdict. Kept modest because the frames are buffered
# whole in host RAM and then uploaded to the device: at 4K, rgb24 is ~24 MB/frame, so 64
# frames (~1.5 GB) leaves ample detection margin while avoiding OOM on high-res inputs.
DEFAULT_MAX_FRAMES = 64


def _probe_dimensions(path: str) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path]
    ).decode().strip()
    w, h = map(int, out.split("x"))
    return w, h


def read_video_frames(path: str, max_frames: int = DEFAULT_MAX_FRAMES) -> np.ndarray:
    """Decode up to ``max_frames`` of a video to an (T, H, W, 3) uint8 array via ffmpeg.

    ``-noautorotate`` keeps ffmpeg's output dimensions equal to ffprobe's coded width/height;
    without it, rotated/portrait inputs would be emitted with swapped dimensions and the
    reshape below would silently transpose the pixels.
    """
    width, height = _probe_dimensions(path)
    # -map 0:v:0 decodes the same stream _probe_dimensions measured (else ffmpeg may pick a
    # different "best" stream, e.g. cover art, and the reshape below would be wrong).
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-noautorotate", "-i", path,
         "-map", "0:v:0", "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    frame_size = width * height * 3
    n = len(raw) // frame_size
    if n == 0:
        raise ValueError(f"decoded no video frames from {path}")
    # .copy() -> writable array (frombuffer is read-only, which torch.from_numpy warns on)
    return np.frombuffer(raw[: n * frame_size], dtype=np.uint8).reshape(n, height, width, 3).copy()


def read_metadata_tags(path: str) -> dict:
    """Return the container-level format tags as a dict (empty if the file genuinely has none).

    Probe failures (ffprobe missing, unreadable file, malformed output) are logged and
    treated as "no tags" rather than crashing the verifier, but are not silently identical
    to a clean read — the warning distinguishes an environment bug from an unmarked file.
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
            stderr=subprocess.PIPE,
        ).decode()
        return json.loads(out).get("format", {}).get("tags", {})
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("could not read metadata tags from %s: %s", path, e)
        return {}
