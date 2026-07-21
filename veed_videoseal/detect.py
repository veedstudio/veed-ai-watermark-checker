"""Detect the fixed VEED AI-generated-content marker in video frames (blind verification)."""

import numpy as np
import torch

from ._frames import frames_to_chw01
from .constants import BIT_THRESH, WATERMARK_TAG
from .message import bit_accuracy, classify_watermark, tag_to_bits
from .model import load_model, model_nbits, resolve_device

# Frames per detect call on the device. Bounds device memory on long or high-resolution inputs, the
# same way sign chunks embedding (an independent knob — embed and detect have different per-frame
# footprints — but kept equal to sign.DEFAULT_CHUNK_FRAMES for consistency). Detection averages the
# per-frame bit logits, so accumulating the logit sum across chunks and thresholding once reproduces
# a single-batch run (see detect_watermark); the chunk size is otherwise a memory/throughput knob.
DEFAULT_DETECT_CHUNK_FRAMES = 16

# Interpolation videoseal's own extract_message("avg") passes to model.detect. Pinned here because
# we call model.detect directly (to accumulate raw logits across chunks instead of letting
# extract_message threshold per batch), and must resize frames the way extract_message would.
# NOTE: model.detect's own default is antialias=True; extract_message overrides it to False, which
# is what we replicate. test_detect_chunking asserts this stays in sync with the installed videoseal.
_DETECT_INTERP = {"mode": "bilinear", "align_corners": False, "antialias": False}


def detect_watermark(
    frames: np.ndarray,
    device: str = "auto",
    ckpt_dir=None,
    chunk_frames: int = DEFAULT_DETECT_CHUNK_FRAMES,
) -> dict:
    """Blindly verify the VEED watermark in ``frames`` ((T,H,W,3) uint8 or float[0,1]).

    Returns ``{detected: bool, bit_accuracy: float, nbits: int}``. Needs no original
    video, seed, or sidecar — only the model checkpoint and the canonical tag.

    Frames are processed in batches of ``chunk_frames`` so a long or high-resolution clip never
    has to sit on the device at once. VideoSeal's ``extract_message('avg')`` averages the per-frame
    bit logits and thresholds at 0; accumulating the logit sum across chunks and thresholding once
    at the end is the same computation, up to floating-point summation order. The two can only
    disagree on a bit whose averaged logit falls within float rounding of 0 — a bit carrying no
    watermark signal — so ``chunk_frames`` does not change the verdict in practice.
    """
    if len(frames) == 0:
        raise ValueError("no video frames to verify")
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")
    dev = resolve_device(device)

    model = None  # loaded on the first chunk's concrete device (str(x.device)), matching
    nbits = None  # embed_watermark so embed and detect share one cached model copy (not two)
    logit_sum = None  # running sum over frames of the per-frame bit logits, shape (K,)
    frame_count = 0
    with torch.no_grad():
        for start in range(0, len(frames), chunk_frames):
            x = frames_to_chw01(frames[start:start + chunk_frames], dev)
            if model is None:
                # Key on the tensor's concrete device (e.g. "cuda:0"), the string embed_watermark
                # uses — frames_to_chw01 uploads to `dev`, but a bare "cuda" materializes as
                # "cuda:0", so keying on `dev` would cache a second copy away from embed's.
                model = load_model(str(x.device), ckpt_dir)
                nbits = model_nbits(model)
            # model.detect is what extract_message calls; preds[:, 1:] are the raw per-frame bit
            # logits (<0 -> bit 0, >0 -> bit 1) before videoseal's threshold/aggregation.
            preds = model.detect(x, is_video=True, interpolation=_DETECT_INTERP)["preds"]
            chunk_sum = preds[:, 1:].sum(dim=0)  # (K,) sum of this chunk's per-frame bit logits
            logit_sum = chunk_sum if logit_sum is None else logit_sum + chunk_sum
            frame_count += preds.shape[0]

    # Mean per-frame logit thresholded at 0: extract_message('avg') over every frame at once, up to
    # float summation order (see docstring).
    recovered = (logit_sum / frame_count > 0).int().cpu().numpy()[:nbits]
    expected = tag_to_bits(WATERMARK_TAG, nbits)
    acc = bit_accuracy(recovered, expected)
    return {"detected": classify_watermark(acc, BIT_THRESH), "bit_accuracy": acc, "nbits": nbits}
