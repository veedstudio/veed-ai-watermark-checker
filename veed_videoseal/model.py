"""Load the VideoSeal model in a way that works from an installed wheel.

VideoSeal's own loader assumes the current working directory is its source checkout
(``cards_dir = Path("videoseal/cards")`` and ``configs/attenuation.yaml`` are both
cwd-relative, and the latter isn't even shipped in the wheel). We sidestep all of that:
resolve the card by absolute path from the installed package, point the attenuation
config at our vendored copy, and download/cache the checkpoint to a configurable dir.
No chdir, no global state.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path
from threading import Lock

import torch

from .constants import MODEL_CARD, SCALING_W

_ASSETS = Path(__file__).parent / "assets"
_cache = {}
_lock = Lock()


def resolve_device(device: str = "auto") -> str:
    """'auto' prefers CUDA, then CPU. MPS is opt-in (pass device='mps') as op coverage
    varies; set PYTORCH_ENABLE_MPS_FALLBACK=1 when using it."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _videoseal_pkg_dir() -> Path:
    spec = importlib.util.find_spec("videoseal")
    if spec is None or spec.origin is None:
        raise ImportError("videoseal is not installed (install with --no-deps; see README)")
    return Path(spec.origin).parent


def _ensure_checkpoint(url: str, ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / os.path.basename(url)
    if not dest.exists():
        # Download to a unique temp file then atomically rename, so concurrent warm-ups
        # (e.g. one per rank on a multi-GPU node with a cold cache) can't observe or leave
        # a half-written checkpoint.
        tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
        torch.hub.download_url_to_file(url, str(tmp))
        os.replace(tmp, dest)
    return dest


def load_model(device: str = "auto", ckpt_dir=None):
    """Load (and cache) the VideoSeal model on the given device.

    Args:
        device: "auto" (cuda→cpu), or an explicit torch device ("cuda", "cpu", "mps").
        ckpt_dir: where to cache weights. Defaults to ~/.cache/veed_videoseal.

    The cache is keyed on the raw device string, which is NOT normalized: "cuda" and "cuda:0"
    refer to the same GPU but are distinct keys, so mixing them loads the model twice (a second
    218 MB copy in VRAM). Caller contract: pass one explicit, consistent device string (e.g.
    "cuda:0") to every load_model / embed_watermark / detect_watermark call in a process rather
    than relying on "auto" (which resolves to bare "cuda") — that guarantees a single cached copy.

    In particular, PRELOAD/warm the model with the indexed device, not "auto": call
    load_model("cuda:0") at startup, because embed_watermark keys the model on the frames'
    concrete device (str(x.device) == "cuda:0"). Warming with "auto"/"cuda" would cache under a
    different key and the first embed would reload the model instead of reusing the warmed copy.
    """
    # decord (a videoseal dep) has no macOS arm64 wheel and is unused for embed/detect.
    # Stub a real (empty) module rather than None so `import decord` succeeds (the import
    # videoseal performs) and only fails if something actually uses a decord attribute.
    # The `not in sys.modules` short-circuit is required: once stubbed, the module's
    # __spec__ is None, and calling find_spec on it again would raise ValueError.
    if "decord" not in sys.modules and importlib.util.find_spec("decord") is None:
        sys.modules["decord"] = types.ModuleType("decord")
    import omegaconf
    from videoseal.utils.cfg import setup_model

    device = resolve_device(device)
    cache_dir = Path(
        ckpt_dir
        or os.environ.get("VEED_VIDEOSEAL_CKPT_DIR")
        or Path.home() / ".cache" / "veed_videoseal"
    )
    key = (device, str(cache_dir))
    with _lock:
        if key in _cache:
            return _cache[key]

        card_path = _videoseal_pkg_dir() / "cards" / f"{MODEL_CARD}.yaml"
        card = omegaconf.OmegaConf.load(card_path)
        # Point at our vendored attenuation config (absolute) instead of the cwd-relative,
        # wheel-missing path the card ships with.
        card.args.attenuation_config = str(_ASSETS / "attenuation.yaml")

        ckpt = _ensure_checkpoint(str(card.checkpoint_path), cache_dir)

        model = setup_model(card, str(ckpt)).eval().requires_grad_(False)
        # Strength lives on the blender. We set it from SCALING_W (currently the card default;
        # see constants.py for the re-encode-vs-crop tradeoff) rather than relying on the card,
        # so the strength is pinned in one place. SCALING_W only affects embedding (detection is
        # unchanged).
        model.blender.scaling_w = SCALING_W
        model = model.to(device)
        _cache[key] = model
        return model


def model_nbits(model) -> int:
    return int(model.get_random_msg(bsz=1).shape[-1])
