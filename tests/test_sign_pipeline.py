"""Unit tests for sign_video's control flow that don't need the model, GPU or a real ffmpeg:
the probe error path, the zero-frame cleanup and the failure cleanup of temp/stale outputs.
Subprocess is monkeypatched."""

import io
import subprocess
import sys
import types

import pytest

from veed_videoseal import sign


def test_probe_missing_dimensions_raises_valueerror(monkeypatch):
    # A stream with no width/height must raise the same ValueError as the no-stream case,
    # not a raw KeyError.
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *a, **k: b'{"streams":[{"height":100,"avg_frame_rate":"30/1"}]}',
    )
    with pytest.raises(ValueError):
        sign._probe_video("x.mp4")


class _FakeProc:
    """Minimal Popen stand-in: decode yields no bytes, encode swallows writes."""

    def __init__(self):
        self.returncode = 0
        self.stdout = io.BytesIO(b"")   # decode: EOF immediately -> zero frames
        self.stdin = io.BytesIO()       # encode: accepts writes

    def wait(self):
        return 0

    def kill(self):
        pass


def test_zero_frames_removes_stray_output(tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    out.write_bytes(b"partial-encoder-output")  # simulate the encoder having created the file

    monkeypatch.setattr(sign, "_probe_video", lambda p: (64, 64, "25"))
    monkeypatch.setattr(sign, "_audio_codecs", lambda p: [])
    monkeypatch.setattr(sign, "_probe_rotation", lambda p: 0)
    monkeypatch.setattr(sign, "_probe_sar", lambda p: None)
    monkeypatch.setattr(sign.subprocess, "Popen", lambda *a, **k: _FakeProc())
    # sign_video defers `from .embed import embed_watermark` to function scope; stub it so this
    # test stays torch-free, and assert it's never actually called on a zero-frame input.
    stub = types.ModuleType("veed_videoseal.embed")
    stub.embed_watermark = lambda *a, **k: pytest.fail("embed_watermark must not run with no frames")
    monkeypatch.setitem(sys.modules, "veed_videoseal.embed", stub)

    with pytest.raises(ValueError):
        sign.sign_video("in.mp4", str(out))

    assert not out.exists()  # the bogus zero-frame file was cleaned up


class _OneFrameProc:
    """Popen stand-in that decodes exactly one 2x2 RGB frame, so the embed loop runs once."""

    def __init__(self):
        self.returncode = 0
        self.stdout = io.BytesIO(b"\x00" * (2 * 2 * 3))
        self.stdin = io.BytesIO()

    def wait(self):
        return 0

    def kill(self):
        pass


def test_rotation_failure_cleans_temp_and_stale_output(tmp_path, monkeypatch):
    # A rotated source whose embed raises mid-stream must leave neither the .prerotate temp nor a
    # stale prior out_path behind (so 'output exists' stays a reliable success signal).
    out = tmp_path / "out.mp4"
    out.write_bytes(b"stale-previous-signed-output")
    temp = tmp_path / "out.mp4.prerotate.mp4"
    temp.write_bytes(b"partial-temp")

    monkeypatch.setattr(sign, "_probe_video", lambda p: (2, 2, "25"))
    monkeypatch.setattr(sign, "_audio_codecs", lambda p: [])
    monkeypatch.setattr(sign, "_probe_rotation", lambda p: 90)  # rotation path -> temp encode target
    monkeypatch.setattr(sign, "_probe_sar", lambda p: None)
    monkeypatch.setattr(sign.subprocess, "Popen", lambda *a, **k: _OneFrameProc())

    def _boom(*a, **k):
        raise RuntimeError("embed boom")

    stub = types.ModuleType("veed_videoseal.embed")
    stub.embed_watermark = _boom
    monkeypatch.setitem(sys.modules, "veed_videoseal.embed", stub)

    with pytest.raises(RuntimeError, match="embed boom"):
        sign.sign_video("in.mp4", str(out))

    assert not temp.exists()  # temp cleaned even though the exception skipped the returncode checks
    assert not out.exists()   # stale prior output removed, not left to masquerade as success
