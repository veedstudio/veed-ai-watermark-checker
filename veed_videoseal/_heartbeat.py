"""A tiny liveness heartbeat for long, output-silent phases.

A caller that runs this CLI as a subprocess may kill a child that emits no output for a while
(a no-progress timeout).
Model loading + the first inference are otherwise completely silent, so wrap them in a
``Heartbeat`` to emit a periodic line to stderr (never stdout, which is reserved for the
single final JSON object) until the block finishes.
"""

import sys
import threading


class Heartbeat:
    """Context manager that prints ``label`` to ``out`` every ``interval`` seconds until the
    wrapped block exits. The worker is a daemon thread joined on exit, so it never outlives
    the block or leaks."""

    def __init__(self, label: str, interval: float = 5.0, out=None):
        self._label = label
        self._interval = interval
        self._out = out if out is not None else sys.stderr
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        # Event.wait returns True only when stop is set; on the interval timeout it returns
        # False, so we tick then loop. This means no tick fires before the first interval.
        while not self._stop.wait(self._interval):
            print(self._label, file=self._out, flush=True)

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
