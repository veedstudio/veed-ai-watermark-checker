"""detect_watermark guards. Needs torch (import chain) but no model/GPU."""

import numpy as np
import pytest

pytest.importorskip("torch")

from veed_videoseal.detect import detect_watermark  # noqa: E402


def test_detect_rejects_empty_frames():
    # Must raise a clear error before touching the model, not feed an empty tensor in.
    with pytest.raises(ValueError):
        detect_watermark(np.empty((0, 4, 4, 3), dtype=np.uint8), device="cpu")
