"""Memory-bounded numerical batches and inexpensive progress-boundary redraws."""
from threading import Event
from time import monotonic

LIVE_UPDATE_SECONDS = 5.0
LIVE_UPDATE_COMPUTE_RATIO = 99.0


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
        self.next_interval = interval

    def at_boundary(self, enqueue, render):
        if self.clock() - self.last_update < self.next_interval:
            return
        finished = Event()
        before = self.clock()

        def update():
            try:
                render()
            finally:
                finished.set()

        enqueue(update)
        finished.wait()
        self.last_update = self.clock()
        # Include queueing as well as drawing. Expensive maps must not keep
        # pausing the coordinator: aim for at most 1% preview overhead.
        self.next_interval = max(self.interval,
                                 (self.last_update - before) * LIVE_UPDATE_COMPUTE_RATIO)


def progress_batches(indices, maximum):
    """Yield full memory-bounded batches independently of GUI refresh timing."""
    maximum = max(1, int(maximum))
    for start in range(0, len(indices), maximum):
        yield start, indices[start:start + maximum]
