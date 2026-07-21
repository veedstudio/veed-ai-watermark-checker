"""Fast CLI tests for ``verify --out``: the verdict JSON is written to the caller-named file,
without needing torch, the model, or a real video. cli defers the torch-heavy imports
(detect/sign/video_io) into the subcommands, so stubbing them in sys.modules before the deferred
``from .detect import ...`` runs makes the subcommand resolve the stub at call time — mirroring the
module-stubbing pattern in test_sign_pipeline.py."""

import json
import sys
import types

import pytest


def _load_cli(monkeypatch, detect_result):
    """Import veed_videoseal.cli against stubbed heavy modules, with detect returning
    detect_result. Returns the freshly imported cli module."""
    detect = types.ModuleType("veed_videoseal.detect")
    detect.detect_watermark = lambda frames, device, ckpt_dir: detect_result

    sign = types.ModuleType("veed_videoseal.sign")
    sign.sign_video = lambda *a, **k: {}

    video_io = types.ModuleType("veed_videoseal.video_io")
    video_io.read_video_frames = lambda path: object()
    video_io.read_metadata_tags = lambda path: {}

    for name, mod in {
        "veed_videoseal.detect": detect,
        "veed_videoseal.sign": sign,
        "veed_videoseal.video_io": video_io,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Force a fresh import so cli binds the stubbed detect_watermark, not a real one.
    monkeypatch.delitem(sys.modules, "veed_videoseal.cli", raising=False)
    import veed_videoseal.cli as cli

    return cli


def test_verify_out_writes_verdict_file(monkeypatch, tmp_path):
    cli = _load_cli(monkeypatch, {"detected": True, "bit_accuracy": 0.98, "nbits": 256})
    out = tmp_path / "verdict.json"

    rc = cli.main(["verify", "--video", "in.mp4", "--out", str(out)])

    assert rc == 0  # watermark present -> exit 0
    verdict = json.loads(out.read_text())
    assert verdict["detected"] is True
    assert verdict["bit_accuracy"] == pytest.approx(0.98)
    assert verdict["nbits"] == 256
    assert "metadata_present" in verdict


def test_verify_exits_zero_when_absent(monkeypatch, tmp_path):
    # A clean video: detected False, but the check itself succeeded, so exit is 0 — "not detected"
    # is a valid verdict, carried in the file, not a failure. (A caller that treats non-zero as
    # failure must not choke on a legitimately-unwatermarked video.)
    cli = _load_cli(monkeypatch, {"detected": False, "bit_accuracy": 0.5, "nbits": 256})
    out = tmp_path / "verdict.json"

    rc = cli.main(["verify", "--video", "in.mp4", "--out", str(out)])

    assert rc == 0  # successful check, verdict absent
    verdict = json.loads(out.read_text())
    assert verdict["detected"] is False


def test_verify_errors_propagate_nonzero(monkeypatch, tmp_path):
    # A genuine failure (e.g. unreadable video) must NOT be swallowed as exit 0: detect raising
    # propagates out of main so the process exits non-zero (via the traceback), distinct from the
    # exit-0 "absent" verdict above.
    cli = _load_cli(monkeypatch, {"detected": False, "bit_accuracy": 0.5, "nbits": 256})

    def _boom(*a, **k):
        raise RuntimeError("decode failed")
    sys.modules["veed_videoseal.detect"].detect_watermark = _boom

    with pytest.raises(RuntimeError, match="decode failed"):
        cli.main(["verify", "--video", "in.mp4", "--out", str(tmp_path / "v.json")])


def test_main_emits_startup_liveness_line(monkeypatch, tmp_path, capsys):
    # main() must print a liveness line to stderr BEFORE dispatching, so a caller's no-progress
    # timer is reset before the (deferred, silent) heavy import runs. It must be stderr, not stdout.
    cli = _load_cli(monkeypatch, {"detected": True, "bit_accuracy": 0.9, "nbits": 256})

    cli.main(["verify", "--video", "in.mp4", "--out", str(tmp_path / "v.json")])

    captured = capsys.readouterr()
    assert "starting" in captured.err  # liveness on stderr
    assert "starting" not in captured.out  # never pollutes the stdout JSON channel


def test_verify_out_leaves_stdout_clean(monkeypatch, tmp_path, capsys):
    # Without --json, --out must not print the verdict (or human text) to stdout — stdout stays
    # empty so a consumer that did read stdout wouldn't get half a verdict.
    cli = _load_cli(monkeypatch, {"detected": True, "bit_accuracy": 0.9, "nbits": 256})
    out = tmp_path / "verdict.json"

    cli.main(["verify", "--video", "in.mp4", "--out", str(out)])

    captured = capsys.readouterr()
    assert captured.out == ""  # verdict went to the file, not stdout
    assert out.exists()
