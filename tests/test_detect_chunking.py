"""Chunked detection is chunk-invariant (same verdict for any chunk_frames) and never exceeds
chunk_frames on the device.

Needs torch (import chain + the real frames_to_chw01) but no model/GPU: model.detect is stubbed so
the per-frame bit logits are a deterministic function of each frame, letting us assert the chunk
boundaries and that the accumulated verdict is independent of chunk size. Equivalence to videoseal's
own extract_message('avg') can't be shown with a stub — that is checked against the real model in
tests/test_roundtrip.py::test_chunked_detect_matches_extract_message.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from veed_videoseal import detect as detect_mod  # noqa: E402
from veed_videoseal.constants import WATERMARK_TAG  # noqa: E402
from veed_videoseal.message import tag_to_bits  # noqa: E402

NBITS = 16


class FakeModel:
    """Returns per-frame bit logits from a caller-supplied function of each frame's mean pixel
    value, and records the batch size of every detect() call so chunk bounds can be asserted."""

    def __init__(self, logits_for):
        self._logits_for = logits_for  # (mean: float) -> np.ndarray (NBITS,) of logits
        self.batch_sizes = []
        self.interps = []  # interpolation dict passed to each detect() call

    def detect(self, imgs, is_video=True, interpolation=None):
        assert is_video is True
        self.batch_sizes.append(imgs.shape[0])
        self.interps.append(interpolation)
        means = imgs.reshape(imgs.shape[0], -1).mean(dim=1)  # (f,)
        rows = np.stack([self._logits_for(float(m)) for m in means])  # (f, NBITS)
        bit_logits = torch.tensor(rows, dtype=torch.float32)
        det_col = torch.zeros((imgs.shape[0], 1))  # detection bit, unused by detect_watermark
        return {"preds": torch.cat([det_col, bit_logits], dim=1)}  # (f, 1 + NBITS)


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(detect_mod, "load_model", lambda dev, ckpt: model)
    monkeypatch.setattr(detect_mod, "model_nbits", lambda m: NBITS)


def _frames(values):
    """One (H,W,3) uint8 frame per value, each filled with that constant."""
    return np.stack([np.full((4, 4, 3), v, dtype=np.uint8) for v in values])


def test_chunking_is_bit_exact_and_bounds_batches(monkeypatch):
    # Logits vary per frame (all bits share the frame's mean-0.5), so an unweighted chunk-mean bug
    # would diverge on uneven chunks — the single-batch vs chunked comparison catches it.
    model = FakeModel(lambda mean: np.full(NBITS, mean - 0.5, dtype=np.float64))
    _patch_model(monkeypatch, model)
    frames = _frames([10, 40, 80, 120, 160, 200, 240, 30, 90, 150])  # 10 distinct frames

    model.batch_sizes.clear()
    single = detect_mod.detect_watermark(frames, device="cpu", chunk_frames=1000)
    assert model.batch_sizes == [10]  # one batch when the chunk exceeds the frame count

    model.batch_sizes.clear()
    chunked = detect_mod.detect_watermark(frames, device="cpu", chunk_frames=3)
    assert model.batch_sizes == [3, 3, 3, 1]  # every frame seen once, never more than chunk_frames

    assert chunked == single  # identical verdict regardless of chunk size


def test_marked_video_detected(monkeypatch):
    # End-to-end positive: every frame's logits carry the tag's bit signs, so the recovered message
    # equals the tag -> detected. NOTE: because all frames agree, the verdict is aggregation-
    # invariant by construction; chunk-boundary/weighting correctness is covered by
    # test_chunking_is_bit_exact_and_bounds_batches (mixed-sign logits over uneven chunks), not here.
    tag_sign = np.where(tag_to_bits(WATERMARK_TAG, NBITS) > 0, 1.0, -1.0)
    model = FakeModel(lambda mean: tag_sign * (0.5 + mean))
    _patch_model(monkeypatch, model)

    res = detect_mod.detect_watermark(_frames([10, 40, 80, 120, 160, 200]), device="cpu", chunk_frames=4)
    assert res["detected"] is True
    assert res["bit_accuracy"] == pytest.approx(1.0)


def test_clean_video_not_detected(monkeypatch):
    # End-to-end negative control: inverted tag signs -> ~0 accuracy -> absent. Like the positive
    # test every frame agrees, so this exercises the verdict/threshold path, not chunk boundaries.
    inverted_sign = np.where(tag_to_bits(WATERMARK_TAG, NBITS) > 0, -1.0, 1.0)
    model = FakeModel(lambda mean: inverted_sign * (0.5 + mean))
    _patch_model(monkeypatch, model)

    res = detect_mod.detect_watermark(_frames([10, 40, 80, 120, 160, 200]), device="cpu", chunk_frames=4)
    assert res["detected"] is False


def test_forwards_detect_interp_to_model(monkeypatch):
    # detect_watermark must pass _DETECT_INTERP to model.detect on every chunk. The real model's own
    # default is antialias=True, so silently dropping the kwarg would resize frames differently than
    # extract_message and drift the verdict — the drift-guard test only checks the constant's value,
    # not that it actually reaches model.detect, so this closes that gap.
    model = FakeModel(lambda mean: np.zeros(NBITS))
    _patch_model(monkeypatch, model)

    detect_mod.detect_watermark(_frames([10, 40, 80, 120, 160]), device="cpu", chunk_frames=2)
    assert model.interps  # detect() was actually called
    assert all(interp == detect_mod._DETECT_INTERP for interp in model.interps)


def test_detect_interp_matches_videoseal_extract_message():
    # _DETECT_INTERP is a hand-copied pin of extract_message's interpolation default (we call the
    # lower-level model.detect, whose own default differs). If a videoseal upgrade changes it,
    # chunked detection would resize frames differently than the library and drift silently — fail
    # loudly here instead. Skips when videoseal isn't installed (same as the model-backed tests).
    import inspect

    videoseal_models = pytest.importorskip("videoseal.models.videoseal")
    default = inspect.signature(
        videoseal_models.Videoseal.extract_message
    ).parameters["interpolation"].default
    assert detect_mod._DETECT_INTERP == default


def test_rejects_bad_chunk_size(monkeypatch):
    _patch_model(monkeypatch, FakeModel(lambda mean: np.zeros(NBITS)))
    with pytest.raises(ValueError):
        detect_mod.detect_watermark(_frames([1, 2, 3]), device="cpu", chunk_frames=0)


def test_detect_keys_model_on_uploaded_tensor_device(monkeypatch):
    """Regression guard for the embed/detect cache-key split: detect_watermark must load the model
    on the *uploaded tensor's* concrete device (str(x.device)) — the string embed_watermark uses —
    not on resolve_device(device). On GPU those differ ('cuda:0' vs bare 'cuda'), and keying on the
    latter caches a second 218 MB model copy away from embed's.

    Simulated on CPU: stub resolve_device to a sentinel that differs from the tensor's real device,
    and frames_to_chw01 to a plain CPU tensor, so the OLD code (load_model(resolve_device(...))) is
    distinguishable from the FIXED code (load_model(str(x.device)))."""
    recorded = []
    model = FakeModel(lambda mean: np.zeros(NBITS))
    monkeypatch.setattr(detect_mod, "load_model", lambda dev, ckpt: recorded.append(dev) or model)
    monkeypatch.setattr(detect_mod, "model_nbits", lambda m: NBITS)
    monkeypatch.setattr(detect_mod, "resolve_device", lambda d: "cuda:7")  # != the tensor's device
    monkeypatch.setattr(detect_mod, "frames_to_chw01", lambda f, dev: torch.zeros(len(f), 3, 4, 4))  # cpu

    detect_mod.detect_watermark(_frames([10, 40, 80]), device="auto")

    assert recorded == ["cpu"]  # keyed on str(x.device)=='cpu', NOT resolve_device's 'cuda:7'
