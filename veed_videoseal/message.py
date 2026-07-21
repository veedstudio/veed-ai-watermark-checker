"""Pure message helpers: derive a fixed bit-message from a tag, compare bits, classify.

No torch/videoseal/GPU dependency — this is the CI-testable core shared by embed and
verify, so the canonical marker and the match decision live in exactly one place.
"""

import hashlib

import numpy as np


def tag_to_bits(tag: str, nbits: int) -> np.ndarray:
    """Deterministically derive a fixed length-``nbits`` bit vector (0/1) from a tag.

    Uses SHAKE-256 so any ``nbits`` is supported. We only ever compare the recovered
    bits to this constant (a yes/no match), so the tag text need not be recoverable.
    """
    if nbits <= 0:
        raise ValueError("nbits must be positive")
    digest = hashlib.shake_256(tag.encode("utf-8")).digest((nbits + 7) // 8)
    bits = [(digest[i // 8] >> (i % 8)) & 1 for i in range(nbits)]
    return np.array(bits, dtype=np.int64)


def bit_accuracy(recovered, expected) -> float:
    """Fraction of bits that match. Raises ValueError on length mismatch."""
    recovered = np.asarray(recovered)
    expected = np.asarray(expected)
    if recovered.shape != expected.shape:
        raise ValueError(f"length mismatch: {recovered.shape} vs {expected.shape}")
    return float((recovered == expected).mean())


def classify_watermark(acc: float, bit_thresh: float) -> bool:
    """The watermark is present when bit-accuracy ``acc`` clears the threshold.

    VideoSeal's detection bit is uninformative for our checkpoint (≈0.5 even when
    watermarked), so presence is decided by bit-accuracy alone. With a 256-bit message,
    random content scores ≈0.5 ± 0.03, so a threshold around 0.75 is ~8σ away
    (false-positive probability ≈ 1e-15).
    """
    return bool(acc >= bit_thresh)
