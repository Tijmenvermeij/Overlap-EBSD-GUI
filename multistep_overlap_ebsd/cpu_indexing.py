"""Bounded CPU matching and scheduling helpers for Kikuchipy workflows."""
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from inspect import signature
import os

import numpy as np


def cpu_worker_count(requested=0):
    requested = int(requested)
    if requested < 0:
        raise ValueError("CPU cores must be 0 (all) or a positive integer.")
    available = max(1, os.cpu_count() or 1)
    return min(requested, available) if requested else available


def cpu_job(function):
    """Scope Dask's worker limit to one GUI job, restoring it on cancellation."""
    parameters = signature(function)
    @wraps(function)
    def run(*args, **kwargs):
        import dask
        requested = parameters.bind(*args, **kwargs).arguments.get("parallel_cores", 0)
        with dask.config.set(scheduler="threads", num_workers=cpu_worker_count(requested)):
            return function(*args, **kwargs)
    return run


def refinement_chunks(count, workers):
    # Enough tasks for load balancing, without many tiny SciPy/Dask tasks on
    # large maps. Count candidate fits, not map points or detector pixels.
    if count >= 64 * workers:
        return 64
    return max(1, min(64, (int(count) + 2 * int(workers) - 1) // (2 * int(workers))))


def unique_fit_rows(indices, eulers):
    """Keep first occurrences of identical seeds at the same map point."""
    seen, rows, inverse = {}, [], []
    for row, (index, euler) in enumerate(zip(indices, eulers)):
        key = (int(index), *map(float, euler))
        if key not in seen:
            seen[key] = len(rows)
            rows.append(row)
        inverse.append(seen[key])
    return np.asarray(rows, dtype=np.int64), np.asarray(inverse, dtype=np.int64)


def _top_matches(scores, keep_n, offset=0):
    """One partial selection; break exact ties by ascending dictionary index."""
    k = min(int(keep_n), scores.shape[1])
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    values = np.take_along_axis(scores, indices, axis=1)
    threshold = values.min(axis=1)
    # argpartition alone arbitrarily drops equal scores at the boundary.
    ties = np.count_nonzero(scores == threshold[:, None], axis=1)
    retained = np.count_nonzero(values == threshold[:, None], axis=1)
    for row in np.flatnonzero(ties > retained):
        greater = np.flatnonzero(scores[row] > threshold[row])
        equal = np.flatnonzero(scores[row] == threshold[row])[:k-len(greater)]
        indices[row] = np.concatenate((greater, equal))
    values = np.take_along_axis(scores, indices, axis=1)
    order = np.lexsort((indices, -values), axis=1)
    return (np.take_along_axis(indices, order, axis=1) + offset,
            np.take_along_axis(values, order, axis=1))


class PreparedDictionary:
    """Reuse exact float32 NCC means/norms, not a multi-GB pattern cache.

    Owned by one session and replaced when the dictionary or mask changes.
    Stores eight bytes per dictionary pattern, plus a validity bitmap.
    """
    def __init__(self, data, mask):
        self.data = data
        self.mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
        self.means = np.empty(len(data), dtype=np.float32)
        self.norms = np.empty(len(data), dtype=np.float32)
        self.ready = np.zeros(len(data), dtype=bool)

    def matches(self, data, mask):
        return self.data is data and ((mask is None and self.mask is None) or
            (mask is not None and self.mask is not None and np.array_equal(mask, self.mask)))

    def prepare(self, start, stop, workers):
        block = self.data[start:stop]
        if hasattr(block, "compute"):
            block = block.compute(scheduler="threads", num_workers=workers)
        block = np.asarray(block).reshape(stop-start, -1)
        if self.mask is not None:
            block = block[:, ~self.mask.ravel()]
        block = block.astype(np.float32)
        if not self.ready[start:stop].all():
            mean = np.mean(block, axis=1, keepdims=True)
            block -= mean
            norm = np.sqrt(np.sum(np.square(block), axis=1, keepdims=True))
            self.means[start:stop] = mean[:, 0]
            self.norms[start:stop] = norm[:, 0]
            self.ready[start:stop] = True
        else:
            block -= self.means[start:stop, None]
            norm = self.norms[start:stop, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            block /= norm
        return block

    def index(self, patterns, keep_n, block_size, workers, progress=None):
        from kikuchipy.indexing import NormalizedCrossCorrelationMetric

        count = int(np.prod(patterns.shape[:-2]))
        if not count or not len(self.data):
            raise ValueError("Indexing requires experimental and dictionary patterns.")
        if self.mask is not None and self.mask.all():
            raise ValueError("Pattern mask excludes all dictionary pixels.")
        metric = NormalizedCrossCorrelationMetric(n_experimental_patterns=count,
            n_dictionary_patterns=len(self.data), signal_mask=self.mask, dtype=np.float32)
        experimental = metric.prepare_experimental(patterns).compute(
            scheduler="threads", num_workers=workers)
        valid_experimental = np.isfinite(experimental).all(axis=1)
        experimental = np.nan_to_num(experimental, copy=False)
        k = min(max(1, int(keep_n)), len(self.data))
        best_indices = np.full((count, k), len(self.data), dtype=np.int64)
        best_scores = np.full((count, k), -np.inf, dtype=np.float32)
        # Matching tasks share a read-only dictionary block. Their combined
        # score storage stays within the caller's experimental batch budget.
        tile = max(128, (count + workers - 1) // workers)
        slices = [slice(i, min(i+tile, count)) for i in range(0, count, tile)]

        def match(part, block, offset):
            scores = np.einsum("ik,mk->im", experimental[part], block,
                               optimize=True, dtype=np.float32)
            np.nan_to_num(scores, copy=False, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            return _top_matches(scores, k, offset)

        with ThreadPoolExecutor(max_workers=min(workers, len(slices))) as pool:
            for offset in range(0, len(self.data), block_size):
                if progress is not None:
                    progress(offset / len(self.data))
                stop = min(offset+block_size, len(self.data))
                block = self.prepare(offset, stop, workers)
                results = pool.map(lambda part: match(part, block, offset), slices)
                for part, (indices, scores) in zip(slices, results):
                    merged_indices = np.concatenate((best_indices[part], indices), axis=1)
                    merged_scores = np.concatenate((best_scores[part], scores), axis=1)
                    order = np.lexsort((merged_indices, -merged_scores), axis=1)[:, :k]
                    best_indices[part] = np.take_along_axis(merged_indices, order, axis=1)
                    best_scores[part] = np.take_along_axis(merged_scores, order, axis=1)
        # Degenerate inputs have no defined NCC. Keep deterministic candidates
        # but never report an invalid correlation as a successful score.
        best_scores[~valid_experimental] = np.nan
        best_scores[~np.isfinite(best_scores)] = np.nan
        return best_indices, best_scores
