"""Lightweight GPU timing via CUDA events.

CUDA kernels are async, so a wall clock around a GPU op measures launch time, not
compute. CUDA events are enqueued on the stream (no host sync, no pipeline serialization);
we read the elapsed time once at the end. Off CUDA (cpu/mps) this is a no-op and reports None.
"""

from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer(device):
    """Bracket pure-GPU work. Yields a 1-element list filled with elapsed GPU milliseconds
    (or None when ``device`` is not CUDA).

        with cuda_timer(dev) as t:
            ... gpu work ...
        gpu_ms = t[0]
    """
    out = [None]
    if torch.device(device).type != "cuda":
        yield out
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield out
    finally:
        end.record()
        end.synchronize()  # wait only for the bracketed work (already forced by a later .cpu())
        out[0] = start.elapsed_time(end)
