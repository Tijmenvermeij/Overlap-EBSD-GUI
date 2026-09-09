"""Safe progress-boundary redraws and time-targeted numerical batches."""
from threading import Event
from time import monotonic

LIVE_UPDATE_SECONDS = 5.0


class LiveUpdateGate:
    """Pause the coordinator while the GUI reads a completed batch's state.

    Progress callbacks run between session updates. Pool workers may still
    compute isolated results, but only the coordinator commits them. Rendering
    stays on Tk's thread; the coordinator cannot write the next batch until
    that render finishes, including when drawing fails.
    """

    def __init__(self, interval=LIVE_UPDATE_SECONDS, *, clock=monotonic):
        self.interval = interval
        self.clock = clock
        self.last_update = clock()

    def at_boundary(self, enqueue, render):
        if self.clock() - self.last_update < self.interval:
            return
        finished = Event()

        def update():
            try:
                render()
            finally:
                finished.set()

        enqueue(update)
        finished.wait()
        self.last_update = self.clock()


def progress_batches(indices, maximum, progress_callback=None, *, clock=monotonic):
    """Yield disjoint batches, targeting five seconds when progress is requested.

    Existing memory limits remain hard caps. Start small, then use the measured
    batch duration to adjust subsequent batches. A single slow pattern cannot
    be interrupted midway. Without a callback, retain the existing batch size.
    """
    maximum = max(1, int(maximum))
    size = min(maximum, 16) if progress_callback is not None else maximum
    start = 0
    while start < len(indices):
        batch = indices[start:start + size]
        before = clock()
        yield start, batch
        elapsed = max(clock() - before, 0.001)
        start += len(batch)
        if progress_callback is not None:
            estimate = max(1, int(len(batch) * LIVE_UPDATE_SECONDS / elapsed))
            size = min(maximum, size * 4, estimate)
