"""Canonical, single-source-of-truth constants shared by embed and verify.

Both the signing side (embed) and the queue-side verifier (detect) import
these, so the marker and the match decision can never drift between the two sides.
"""

# The fixed, human-readable marker. Goes verbatim into MP4 metadata, and is hashed to the
# VideoSeal bit-message embedded in the pixels (see message.tag_to_bits). Deliberately
# model-agnostic: VEED runs multiple generation models, so the marker identifies "VEED
# AI-generated content", not which model produced it — a verifier never needs updating when a
# new model ships. Changing this string invalidates detection of already-signed video, so treat
# it as frozen once real content has been signed with it.
WATERMARK_TAG = "VEED:AI-generated"

# VideoSeal model card (256-bit). Resolved from the installed videoseal package.
MODEL_CARD = "videoseal_1.0"

# Watermark embedding strength (model.blender.scaling_w) — the VideoSeal card default.
# We deliberately target re-encode robustness ONLY, not crop/rescale robustness: the
# generation pipeline delivers a single H.264 encode (libx264 crf 17), so that — plus a
# margin for a later platform re-upload — is all the mark must survive. 0.2 clears it with
# room to spare: signed frames re-encoded from crf 17 through crf 28 still detect at ~1.0
# bit-accuracy, while distorting the image ~2 dB less than the old 0.4 (46.9 vs 44.7 dB PSNR
# on a 720p clip). We previously bumped this to 0.4 to also survive crops/rescaling; that
# requirement is dropped, so we revert to 0.2 for the lower quality impact. SCALING_W only
# affects embedding (detection is unchanged).
SCALING_W = 0.2

# Bit-accuracy threshold for declaring the watermark present. The VideoSeal detection
# bit is uninformative for this checkpoint, so presence is decided by bit-accuracy alone.
# With 256 bits, random content scores ~0.5 ± 0.03, so 0.75 is ~8σ (false positive ~1e-15).
BIT_THRESH = 0.75

# MP4 container metadata tags (the convenient, strippable carrier). Only standard mov/mp4
# tags survive the muxer (custom keys are silently dropped), so the marker lives in `comment`.
METADATA_TAGS = {
    "comment": WATERMARK_TAG,
    "title": "AI-generated video",
}
