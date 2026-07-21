"""Embed the fixed VEED AI-generated-content marker into video frames (in-pixel watermark)."""

from dataclasses import dataclass

import numpy as np
import torch

from ._frames import chw01_to_frames, frames_to_chw01, unit_range_violation
from ._perf import cuda_timer
from .constants import WATERMARK_TAG
from .message import tag_to_bits
from .model import load_model, model_nbits, resolve_device

# Frames uploaded + embedded per batch in embed_watermark, bounding peak device memory on long or
# high-resolution clips (the full clip never sits on the device at once — only one chunk plus its
# downloaded host result). Mirrors detect.DEFAULT_DETECT_CHUNK_FRAMES / sign.DEFAULT_CHUNK_FRAMES.
# Only the host numpy path chunks: embed_watermark_tensor gets an already-resident device tensor,
# so wrapper chunking cannot lower its peak (see its docstring).
DEFAULT_EMBED_CHUNK_FRAMES = 16


@dataclass
class WatermarkResult:
    """Watermarked frames plus the pure-GPU embed time.

    gpu_compute_ms is the CUDA-event-measured time of the on-GPU watermark work only
    (host<->device transfers excluded); None when not running on CUDA.
    """

    frames: np.ndarray  # (T,H,W,3) uint8
    gpu_compute_ms: float | None


def _watermark_msg(model, device) -> torch.Tensor:
    """Build the fixed WATERMARK_TAG message tensor (1, nbits) on ``device``.

    Kept out of the timed embed region: ``model_nbits`` is a GPU op and the ``torch.tensor(...,
    device=...)`` is a host->device upload, neither of which is the embed compute we measure.
    """
    nbits = model_nbits(model)
    bits = tag_to_bits(WATERMARK_TAG, nbits)
    return torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)  # (1, nbits)


def _embed_core(model, x: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    """Core embed op: (T,3,H,W) float[0,1] -> watermarked (T,3,H,W) float[0,1] on the same device.

    model.embed chunks the embedder forward internally (self.chunk_size), so this does not run the
    network on all frames at once. Shared by the numpy and tensor public entrypoints so the embed
    compute lives in one place. Output is clamped to [0,1]: real embedders can push pixels slightly
    out of range, and the numpy path's uint8 quantization would otherwise wrap (e.g. 1.01 -> 257 &
    0xff -> 1) instead of saturating.
    """
    with torch.no_grad():
        return model.embed(x, msgs=msg, is_video=True)["imgs_w"].clamp(0, 1)


def _validate_chw01(x: torch.Tensor) -> None:
    """Enforce the (T,3,H,W) float32-in-[0,1] tensor contract shared with ``frames_to_chw01``."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"embed_watermark_tensor expects a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected (T,3,H,W) frames, got shape {tuple(x.shape)}")
    # VideoSeal weights are float32; a half/bfloat/double tensor would crash inside model.embed
    # ("Input type ... and weight type ... should be the same"), so reject it with a clear message
    # rather than an autocast that silently doubles the input's VRAM.
    if x.dtype != torch.float32:
        raise ValueError(
            f"expected a float32 tensor in [0,1], got {x.dtype} "
            "(cast fp16/bf16 generation output with .float(); pass uint8 numpy to embed_watermark)"
        )
    violation = unit_range_violation(x)
    if violation is not None:
        lo, hi = violation
        raise ValueError(
            f"expected values in [0,1], got range [{lo:.4g}, {hi:.4g}] — convert generation "
            "output (often [-1,1]) to [0,1] before embedding"
        )


def embed_watermark_tensor(x: torch.Tensor, ckpt_dir=None) -> torch.Tensor:
    """Watermark an on-device tensor without any host<->device transfer, numpy, or ffmpeg.

    ``x`` is ``(T,3,H,W)`` float32 in [0,1] already on the target device; returns the watermarked
    frames in the same shape/dtype on the same device. The fixed WATERMARK_TAG is embedded.

    The model is loaded (and cached) on ``x``'s device, so warm ``load_model`` on that same device
    to reuse the cached copy instead of loading a second one. Everything — model, input, and the
    message — stays on ``x.device``; nothing is relocated, so there is no cross-device mismatch.

    This is the primitive behind ``embed_watermark`` (the numpy wrapper) — use it to avoid the
    round-trip when the frames are already a GPU tensor (e.g. straight off a generation model).
    """
    _validate_chw01(x)
    model = load_model(str(x.device), ckpt_dir)
    msg = _watermark_msg(model, x.device)
    return _embed_core(model, x, msg)


def embed_watermark(
    frames: np.ndarray,
    device: str = "auto",
    ckpt_dir=None,
    chunk_frames: int = DEFAULT_EMBED_CHUNK_FRAMES,
) -> WatermarkResult:
    """Return a watermarked copy of ``frames`` ((T,H,W,3) uint8) and its GPU embed time.

    The embedded message is the fixed marker derived from WATERMARK_TAG, so every video
    carries the same signature. Runs on the given device and fails if it can't — there is
    deliberately no CPU fallback: silently degrading a GPU embed to CPU would be a
    catastrophic, hidden slowdown. A GPU OOM propagates and (fail-closed) aborts the run.

    Frames are uploaded and embedded in batches of ``chunk_frames`` so a long or high-resolution
    clip never has to sit on the device at once (only one chunk plus its downloaded host result).
    The fixed marker is embedded identically on every chunk, so chunking does not change what the
    verifier recovers — but the per-key-frame propagation is recomputed per chunk, so pixels near a
    chunk boundary differ slightly from a single-shot embed. ``gpu_compute_ms`` is the sum of the
    per-chunk GPU times (None off CUDA).

    This is the numpy convenience wrapper for the shared embed core (see ``embed_watermark_tensor``
    for the on-device primitive): it uploads the frames, runs that core, and downloads uint8 frames.
    Callers that already hold a GPU tensor should use ``embed_watermark_tensor`` to skip the
    host<->device round-trip.
    """
    if len(frames) == 0:
        raise ValueError("no video frames to watermark")
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")
    dev = resolve_device(device)

    # Write each downloaded chunk straight into the preallocated result — never hold the whole clip
    # on the device, nor a second host copy (which np.concatenate of per-chunk arrays would need).
    frames_out = np.empty(np.asarray(frames).shape, dtype=np.uint8)  # (T,H,W,3)
    total_ms = None  # running sum of per-chunk GPU compute (stays None off CUDA)
    model = None  # loaded once on the first chunk's concrete device, then cache-reused
    msg = None
    for start in range(0, len(frames), chunk_frames):
        x = frames_to_chw01(frames[start:start + chunk_frames], dev)  # host->device upload (not timed)
        if model is None:
            # Key the model on the tensor's concrete device (e.g. "cuda:0"), the same string
            # embed_watermark_tensor uses, so both entrypoints share one cached model copy. The
            # fixed message is (1, nbits) — independent of the chunk's frame count — so it is
            # built once and reused across chunks.
            model = load_model(str(x.device), ckpt_dir)
            msg = _watermark_msg(model, x.device)  # model_nbits GPU op + upload — not embed compute

        # Time only the on-GPU watermark work: embed + uint8 quantization, between the upload
        # above and the device->host download below.
        with torch.no_grad(), cuda_timer(dev) as t:
            wm = chw01_to_frames(_embed_core(model, x, msg))  # (T,H,W,3) uint8, still on device

        frames_out[start:start + wm.shape[0]] = wm.cpu().numpy()  # download; frees chunk's device tensors
        if t[0] is not None:
            total_ms = (total_ms or 0.0) + t[0]

    return WatermarkResult(frames=frames_out, gpu_compute_ms=total_ms)
