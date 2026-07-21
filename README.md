# ai-content-watermarking

Shared Python package (`veed-videoseal`) for embedding and verifying an invisible
**AI-generated-content watermark** in videos, built on Meta
[VideoSeal](https://github.com/facebookresearch/videoseal). One source of truth for the VEED
marker: a durable, invisible label that marks video as AI-generated and survives ordinary
re-encoding. Integrate it as a CLI subprocess (`--json`) or as a Python library.

- **In-pixel watermark** (robust to re-encoding): a fixed 256-bit message derived from
  `WATERMARK_TAG`, embedded in the frames and recoverable after H.264 re-encoding. Tuned for
  low quality impact rather than crop/rescale survival (see `SCALING_W` in `constants.py`).
- **Blind verification**: detection needs only the model + the canonical tag — no source
  video, seed, or sidecar.
- **Arbitrary videos**: `sign` streams the whole video through ffmpeg in chunks, preserving
  resolution, frame rate, duration and the original audio track.
- **Configurable device**: `auto` (CUDA→CPU), `cpu`, `cuda`, or `mps` (opt-in).

## Installation

**Requires Python >= 3.12** — the `videoseal` extra pins `scipy==1.18.0` and `numpy`, which
themselves require 3.12+, so the install cannot resolve on 3.10/3.11. Note this may be newer
than your system default `python3`; use an explicit `python3.12` (or later) interpreter.

`ffmpeg`/`ffprobe` must be on `PATH` for the video I/O (`brew install ffmpeg` on macOS,
`apt install ffmpeg` on Debian/Ubuntu).

VideoSeal declares `decord`, which has **no macOS arm64 wheel** and is **not needed** for
embed/detect (it is only VideoSeal's video loader; we do I/O via ffmpeg). So VideoSeal's real
runtime deps live in this package's `videoseal` extra, and VideoSeal *itself* is installed
`--no-deps` by `scripts/install_videoseal.sh`:

```bash
pip install ".[videoseal]"          # this package + VideoSeal's runtime deps (and the CLI)
bash scripts/install_videoseal.sh   # VideoSeal itself, installed --no-deps
```



### Notes on VideoSeal packaging quirks (handled by this package)

- VideoSeal's loader uses **cwd-relative** paths for its model cards and configs. We resolve
  the card by absolute path from the installed package instead.
- The card references `configs/attenuation.yaml`, which is **not shipped in the wheel**. We
  vendor it at `veed_videoseal/assets/attenuation.yaml`.
- Weights download from `dl.fbaipublicfiles.com` on first use. For offline/airgapped hosts,
  pre-place `y_256b_img.pth` in a directory and point `VEED_VIDEOSEAL_CKPT_DIR` (or
  `--ckpt-dir`) at it.

## CLI

```bash
# Sign an arbitrary video (audio, fps and duration preserved):
veed-videoseal sign --video in.mp4 --out signed.mp4 [--json] [--device auto|cpu|cuda|mps] [--ckpt-dir DIR]

# Verify:
veed-videoseal verify --video signed.mp4 [--json] [--device ...] [--ckpt-dir DIR]
# exit 0 if the pixel watermark is present, 1 otherwise
```

With `--json` each subcommand prints exactly one JSON object to stdout (and nothing else):

```jsonc
// sign
{"out": "signed.mp4", "frames": 1800, "gpu_compute_ms": 1234.5}
// verify
{"detected": true, "bit_accuracy": 0.99, "nbits": 256, "metadata_present": true}
```

### Build & run from a local venv (no PATH changes)

You don't have to put `veed-videoseal` on your `PATH`. Build a self-contained virtualenv from
the checkout, then invoke the CLI as a module — the package ships `veed_videoseal/__main__.py`,
so `python -m veed_videoseal` is equivalent to the console script. Nothing is installed globally
and your shell `PATH` is never touched.

```bash
# 1. Build: create a local venv and install the package + its deps into it
python3.12 -m venv .venv                      # 3.12+ required; see Installation
./.venv/bin/pip install ".[videoseal]"        # this package + VideoSeal's runtime deps
PIP=./.venv/bin/pip bash scripts/install_videoseal.sh   # VideoSeal itself, --no-deps

# 2. Run: call the venv's python directly (no `source .venv/bin/activate` needed)
./.venv/bin/python -m veed_videoseal verify --video signed.mp4 --json
./.venv/bin/python -m veed_videoseal sign --video in.mp4 --out signed.mp4 --json
```

`ffmpeg`/`ffprobe` must still be on `PATH` for video I/O, and **signing requires a GPU** (see
[GPU usage](#gpu-usage)). Once the venv exists, only step 2 is needed on subsequent runs.

### GPU usage

`--device auto` selects CUDA when a GPU is visible to PyTorch, else CPU (`cpu`/`cuda`/`mps`
force a device). **Signing requires a GPU** — embedding has no CPU fallback by design (a silent
CPU embed would be catastrophically slow), so on a CPU-only host `sign` raises rather than
crawling. **Verification runs on CPU or GPU.** A GPU machine is "ready" once: VideoSeal + its
deps are installed (`scripts/install_videoseal.sh`), this package is installed (`pip install .`),
`ffmpeg`/`ffprobe` are on `PATH`, and the weights are available (auto-downloaded on first run,
or pre-placed via `VEED_VIDEOSEAL_CKPT_DIR`). On Linux the default PyPI `torch` wheel is
CUDA-enabled, so `--device auto` then runs on the GPU with no extra steps.

## Library

```python
from veed_videoseal.sign import sign_video
from veed_videoseal.detect import detect_watermark
from veed_videoseal.video_io import read_video_frames

sign_video("in.mp4", "signed.mp4", device="auto")
result = detect_watermark(read_video_frames("signed.mp4"), device="auto")
# {"detected": True, "bit_accuracy": 0.99, "nbits": 256}
```

## Tests

```bash
pytest -m "not gpu"   # pure-logic, no model/GPU — runs in CI
pytest -m gpu         # embed→detect + sign→verify round-trips; needs weights, runs on CPU/MPS/CUDA
```

The round-trip tests use **real natural images**: VideoSeal's ConvNeXt extractor scores at
chance on synthetic noise, so watermark tests must use natural content.
