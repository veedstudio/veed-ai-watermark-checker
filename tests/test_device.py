"""Device-resolution tests. This machine has no CUDA, so we mock torch.cuda.is_available
to exercise the production `auto`→`cuda` branch without real hardware. Needs torch but no
GPU; not `gpu`-marked, so it runs anywhere torch is installed.
"""

import pytest

torch = pytest.importorskip("torch")

from veed_videoseal.model import resolve_device  # noqa: E402


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_explicit_device_is_passed_through(monkeypatch):
    # An explicit device is honored regardless of CUDA availability.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"
    assert resolve_device("cpu") == "cpu"
