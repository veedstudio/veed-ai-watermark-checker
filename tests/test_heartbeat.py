"""Heartbeat context manager: emits liveness lines while a slow block runs, and stops
cleanly on exit. Pure threading logic — no model, GPU or ffmpeg."""

import io
import time

from veed_videoseal._heartbeat import Heartbeat


def test_emits_while_block_runs():
    buf = io.StringIO()
    with Heartbeat("hb", interval=0.01, out=buf):
        # Bounded wait for the first tick to land (not a fixed sleep), so the test is fast
        # and not flaky.
        deadline = time.monotonic() + 2.0
        while "hb" not in buf.getvalue() and time.monotonic() < deadline:
            time.sleep(0.005)
    assert buf.getvalue().count("hb") >= 1


def test_stops_after_exit():
    buf = io.StringIO()
    with Heartbeat("hb", interval=0.01, out=buf):
        time.sleep(0.05)
    ticks_at_exit = buf.getvalue().count("hb")
    time.sleep(0.05)
    # No further ticks after the context manager joined its worker thread.
    assert buf.getvalue().count("hb") == ticks_at_exit


def test_no_tick_before_first_interval():
    buf = io.StringIO()
    with Heartbeat("hb", interval=10.0, out=buf):
        pass  # exits well before the interval elapses
    assert buf.getvalue() == ""
