"""cuda_timer helper. Needs torch but no GPU."""

import pytest

torch = pytest.importorskip("torch")

from veed_videoseal._perf import cuda_timer  # noqa: E402


def test_cuda_timer_reports_none_off_cuda():
    with cuda_timer("cpu") as t:
        _ = torch.ones(8) * 2
    assert t[0] is None


def test_cuda_timer_yields_single_element_list():
    with cuda_timer("cpu") as t:
        assert isinstance(t, list) and len(t) == 1
