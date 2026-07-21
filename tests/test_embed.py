"""embed_watermark_tensor / embed_watermark: tensor-in/out contract, the fixed tag, and that the
numpy wrapper is exactly the tensor core + uint8 convert.

Needs torch (import chain + the real frames_to_chw01) but no model/GPU: model.embed is stubbed with
a deterministic affine so both public entrypoints are exercised without weights.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from veed_videoseal import embed as embed_mod  # noqa: E402
from veed_videoseal._frames import chw01_to_frames, frames_to_chw01  # noqa: E402
from veed_videoseal.constants import WATERMARK_TAG  # noqa: E402
from veed_videoseal.message import tag_to_bits  # noqa: E402

NBITS = 16


class FakeModel:
    """model.embed stub: applies a fixed affine so output is deterministic, records the msgs it was
    handed (to assert the embedded tag) and the exact imgs object (to assert no relocation)."""

    def __init__(self, scale=0.5, bias=0.25):
        self.scale = scale
        self.bias = bias
        self.msgs = None
        self.received = None
        self.embed_calls = 0

    def embed(self, imgs, msgs=None, is_video=True, **kwargs):
        assert is_video is True
        self.msgs = msgs
        self.received = imgs
        self.embed_calls += 1
        return {"imgs_w": imgs * self.scale + self.bias}


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(embed_mod, "load_model", lambda dev, ckpt=None: model)
    monkeypatch.setattr(embed_mod, "model_nbits", lambda m: NBITS)


def _frames(t=6, h=4, w=4):
    return (np.arange(t * h * w * 3) % 256).astype(np.uint8).reshape(t, h, w, 3)


def test_tensor_in_tensor_out_same_device_and_shape(monkeypatch):
    _patch_model(monkeypatch, FakeModel())
    x = torch.rand(5, 3, 8, 8)  # (T,3,H,W) float[0,1]
    out = embed_mod.embed_watermark_tensor(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out.device == x.device
    torch.testing.assert_close(out, x * 0.5 + 0.25)  # no numpy, no host round-trip


def test_input_not_copied_and_output_on_input_device(monkeypatch):
    """embed_watermark_tensor hands x to model.embed as the SAME object and returns on x's device.

    The identity assert catches any edit that materializes a new input tensor — .clone(),
    .contiguous(), or .to(other_device) — which is how an accidental host round-trip shows up. It
    does NOT catch a same-device .cpu() no-op (returns self on CPU): a genuine device round-trip is
    only observable on CUDA, which CI lacks, so this is the strongest CPU-checkable form."""
    model = FakeModel()
    _patch_model(monkeypatch, model)
    x = torch.rand(4, 3, 8, 8)
    out = embed_mod.embed_watermark_tensor(x)
    assert model.received is x  # input reached model.embed without being copied/relocated
    assert out.device == x.device  # result returned on the input's device


def test_output_clamped_to_unit_range(monkeypatch):
    """model.embed can push pixels slightly out of [0,1]; the core must clamp so downstream uint8
    quantization saturates instead of wrapping."""
    model = FakeModel(scale=2.0, bias=-0.5)  # maps [0,1] -> [-0.5, 1.5], i.e. out of range
    _patch_model(monkeypatch, model)
    out = embed_mod.embed_watermark_tensor(torch.rand(3, 3, 4, 4))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_embeds_fixed_watermark_tag(monkeypatch):
    model = FakeModel()
    _patch_model(monkeypatch, model)
    embed_mod.embed_watermark_tensor(torch.rand(3, 3, 4, 4))
    expected = torch.tensor(tag_to_bits(WATERMARK_TAG, NBITS), dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(model.msgs.cpu(), expected)


def test_tensor_contract_rejects_bad_input(monkeypatch):
    _patch_model(monkeypatch, FakeModel())
    with pytest.raises(TypeError):
        embed_mod.embed_watermark_tensor(np.zeros((3, 3, 4, 4), dtype=np.float32))  # not a tensor
    with pytest.raises(ValueError):
        embed_mod.embed_watermark_tensor(torch.rand(3, 4, 4))  # not 4-D
    with pytest.raises(ValueError):
        embed_mod.embed_watermark_tensor(torch.rand(3, 4, 4, 4))  # channel dim != 3
    with pytest.raises(ValueError):
        embed_mod.embed_watermark_tensor(torch.zeros(3, 3, 4, 4, dtype=torch.uint8))  # non-float
    with pytest.raises(ValueError):
        embed_mod.embed_watermark_tensor(torch.rand(3, 3, 4, 4, dtype=torch.float16))  # not fp32
    with pytest.raises(ValueError):
        embed_mod.embed_watermark_tensor(torch.rand(3, 3, 4, 4) * 2 - 1)  # [-1,1], out of range


def test_chunking_stitches_and_matches_single_shot(monkeypatch):
    """embed_watermark uploads/embeds in chunk_frames batches and concatenates the host results.
    With a frame-independent stub the stitched output is bit-identical to a single-shot embed, so
    this pins the loop's batching + concatenation (real models differ only at chunk boundaries)."""
    frames = _frames(t=20, h=4, w=4)

    m_single = FakeModel()
    _patch_model(monkeypatch, m_single)
    single = embed_mod.embed_watermark(frames, device="cpu", chunk_frames=1000)  # one batch
    assert m_single.embed_calls == 1

    m_chunk = FakeModel()
    _patch_model(monkeypatch, m_chunk)
    chunked = embed_mod.embed_watermark(frames, device="cpu", chunk_frames=8)  # 8 + 8 + 4
    assert m_chunk.embed_calls == 3

    assert chunked.frames.shape == frames.shape
    np.testing.assert_array_equal(chunked.frames, single.frames)


def test_embed_rejects_empty_and_bad_chunk(monkeypatch):
    _patch_model(monkeypatch, FakeModel())
    with pytest.raises(ValueError):
        embed_mod.embed_watermark(np.zeros((0, 4, 4, 3), dtype=np.uint8), device="cpu")  # no frames
    with pytest.raises(ValueError):
        embed_mod.embed_watermark(_frames(), device="cpu", chunk_frames=0)  # chunk_frames < 1


def test_numpy_wrapper_equals_tensor_core_plus_convert(monkeypatch):
    _patch_model(monkeypatch, FakeModel())
    frames = _frames()

    result = embed_mod.embed_watermark(frames, device="cpu")
    assert result.frames.dtype == np.uint8
    assert result.frames.shape == frames.shape

    # The numpy wrapper must be exactly: the tensor core on the uploaded frames, then uint8 NHWC.
    x = frames_to_chw01(frames, "cpu")
    wm = embed_mod.embed_watermark_tensor(x)
    expected = chw01_to_frames(wm).numpy()
    np.testing.assert_array_equal(result.frames, expected)
