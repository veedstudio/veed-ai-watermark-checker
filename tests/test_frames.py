"""Frame<->tensor conversion. Needs torch but no model/GPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from veed_videoseal._frames import frames_to_chw01  # noqa: E402


def test_uint8_is_scaled_to_unit_range_and_chw():
    frames = np.full((2, 4, 5, 3), 255, dtype=np.uint8)
    t = frames_to_chw01(frames, "cpu")
    assert t.shape == (2, 3, 4, 5)
    assert torch.allclose(t, torch.ones_like(t))


def test_float_unit_range_passes_through():
    frames = np.zeros((1, 4, 4, 3), dtype=np.float32)
    t = frames_to_chw01(frames, "cpu")
    assert float(t.max()) == 0.0


def test_float_in_0_255_is_rejected():
    frames = (np.random.rand(1, 4, 4, 3) * 255).astype(np.float32)
    with pytest.raises(ValueError):
        frames_to_chw01(frames, "cpu")


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        frames_to_chw01(np.zeros((4, 4, 3), dtype=np.uint8), "cpu")
