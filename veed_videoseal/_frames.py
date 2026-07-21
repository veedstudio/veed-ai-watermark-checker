"""Shared frame<->tensor conversion used by both embed and detect."""

import numpy as np
import torch

# Tolerance for the float [0,1] contract, absorbing rounding on both bounds. Single source of
# truth for the numpy path (frames_to_chw01) and the tensor path (embed._validate_chw01).
UNIT_RANGE_EPS = 1e-3


def unit_range_violation(x: torch.Tensor):
    """Return (lo, hi) if float tensor ``x`` has values outside [0,1] (± UNIT_RANGE_EPS), else None.

    Empty tensors return None (nothing to check). Callers format their own error so each can add
    context (uint8 hint for numpy, [-1,1] hint for generation output)."""
    if not x.numel():
        return None
    lo, hi = float(x.min()), float(x.max())
    if lo < -UNIT_RANGE_EPS or hi > 1.0 + UNIT_RANGE_EPS:
        return lo, hi
    return None


def frames_to_chw01(frames: np.ndarray, device: str) -> torch.Tensor:
    """(T,H,W,3) uint8 or float[0,1] -> (T,3,H,W) float[0,1] on ``device``.

    uint8 is scaled by 1/255; float must already be in [0,1]. A float input outside [0,1]
    (e.g. [0,255]) is rejected rather than silently embedded/detected into garbage. The float
    range contract is enforced on both bounds (a tiny epsilon absorbs rounding).
    """
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected frames of shape (T,H,W,3), got {arr.shape}")
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    if arr.dtype == np.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        violation = unit_range_violation(t)
        if violation is not None:
            lo, hi = violation
            raise ValueError(
                f"float frames must be in [0,1]; got range [{lo:.4g}, {hi:.4g}] "
                "(pass uint8 for [0,255])"
            )
    return t.permute(0, 3, 1, 2).contiguous()


def chw01_to_frames(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``frames_to_chw01``: (T,3,H,W) float[0,1] -> (T,H,W,3) uint8 on the same device.

    Quantizes with round-then-clamp so values are mapped to the nearest byte and out-of-range
    inputs saturate instead of wrapping. Kept device-resident (no ``.cpu()``) so callers control
    when the download happens and can time the GPU work separately.
    """
    return (x * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
