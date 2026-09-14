"""Read-only indexing benchmark and bounded-memory NCC prototype.

Source workflow, pattern data and dictionary are never modified. Production
code is unchanged. Timings exclude workflow restore and input selection.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

import dask
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd.core import WorkflowSession
from kikuchipy.indexing import NormalizedCrossCorrelationMetric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--counts', type=int, nargs='+', default=[16, 128, 512])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 10])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prototype', action='store_true')
    parser.add_argument('--gui-count', type=int, default=0,
                        help='Also compare the application path with and without progress callbacks.')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = WorkflowSession()
    print(session.restore_workflow_state(args.workflow), flush=True)
    cache = session.dictionary_cache
    indices = np.sort(np.random.default_rng(20260909).choice(
        np.flatnonzero(session.current_phases == cache.phase_id), max(args.counts), replace=False))
    signal = session._materialize_signal_batch(session._signal_from_indices(
        indices, software_binning=cache.software_binning, crop_extent=cache.crop_extent))
    patterns = np.asarray(signal.data)
    mask = session._signal_mask_for_dictionary_cache(cache)
    iteration = session._dictionary_n_per_iteration(cache, mask)
    report = dict(dictionary_count=cache.rotation_count, pattern_shape=cache.pattern_shape,
                  dictionary_iteration=iteration, keep_n=5, versions={}, records=[],
                  blas='Apple Accelerate', veclib_maximum_threads=os.getenv('VECLIB_MAXIMUM_THREADS'),
                  timing_scope='Workflow restore and experimental input selection excluded; dictionary I/O included.')
    import kikuchipy, scipy
    report['versions'] = dict(numpy=np.__version__, scipy=scipy.__version__,
                              kikuchipy=kikuchipy.__version__, dask=dask.__version__)
    reference = {}

    def save(record):
        report['records'].append(record)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(record), flush=True)

    for count in args.counts:
        sig = session._signal_from_pattern_data(patterns[:count].copy())
        for workers in args.threads:
            capture = io.StringIO()
            with dask.config.set(scheduler='threads', num_workers=workers), redirect_stdout(capture), redirect_stderr(capture):
                start, cpu_start = perf_counter(), process_time()
                result = sig.dictionary_indexing(
                    dictionary=cache.signal, metric='ncc', keep_n=5,
                    n_per_iteration=iteration, signal_mask=mask, rechunk=False)
                cpu_seconds, seconds = process_time()-cpu_start, perf_counter()-start
            scores = np.asarray(result.prop['scores'])
            matches = np.asarray(result.prop['simulation_indices'])
            if count not in reference:
                reference[count] = (scores.copy(), matches.copy())
            save(dict(method='current_kikuchipy', count=count, dask_workers=workers,
                      seconds=seconds, cpu_seconds=cpu_seconds, average_cpu_cores=cpu_seconds/seconds,
                      patterns_per_second=count/seconds, rss_bytes=psutil.Process().memory_info().rss,
                      candidate_indices_equal=bool(np.array_equal(matches, reference[count][1])),
                      maximum_score_difference=float(np.max(np.abs(scores-reference[count][0])))))

    if args.prototype:
        # Process every selected experimental pattern while each dictionary
        # block is resident. Normalize exactly as Kikuchipy; select top-k once.
        metric = NormalizedCrossCorrelationMetric(n_experimental_patterns=len(indices),
                    n_dictionary_patterns=cache.rotation_count, signal_mask=mask, dtype=np.float32)
        experimental = metric.prepare_experimental(patterns).compute()
        for count in args.counts:
            exp = experimental[:count]
            best_scores = np.full((count, 5), -np.inf, dtype=np.float32)
            best_indices = np.zeros((count, 5), dtype=np.int64)
            totals = dict(read_seconds=0., normalize_seconds=0., match_seconds=0., topk_seconds=0.)
            start, cpu_start = perf_counter(), process_time()
            # Align reads to source Dask chunk boundaries to avoid rereading
            # partially overlapping lazy chunks from the compressed dictionary.
            for offset in range(0, cache.rotation_count, 8192):
                tick = perf_counter()
                block = cache.signal.data[offset:offset+8192].compute(scheduler='threads', num_workers=1)
                totals['read_seconds'] += perf_counter()-tick
                tick = perf_counter()
                block = metric.prepare_dictionary(block.reshape(len(block), -1))
                totals['normalize_seconds'] += perf_counter()-tick
                tick = perf_counter()
                scores = np.einsum('ik,mk->im', exp, block, optimize=True, dtype=np.float32)
                totals['match_seconds'] += perf_counter()-tick
                tick = perf_counter()
                local = np.argpartition(scores, -5, axis=1)[:, -5:]
                values = np.take_along_axis(scores, local, axis=1)
                candidates = np.concatenate((best_indices, local + offset), axis=1)
                merged = np.concatenate((best_scores, values), axis=1)
                order = np.argsort(-merged, axis=1)[:, :5]
                best_scores = np.take_along_axis(merged, order, axis=1)
                best_indices = np.take_along_axis(candidates, order, axis=1)
                totals['topk_seconds'] += perf_counter()-tick
            cpu_seconds, seconds = process_time()-cpu_start, perf_counter()-start
            save(dict(method='streamed_ncc_prototype', count=count, seconds=seconds,
                      cpu_seconds=cpu_seconds, average_cpu_cores=cpu_seconds/seconds,
                      patterns_per_second=count/seconds, rss_bytes=psutil.Process().memory_info().rss,
                      candidate_indices_equal=bool(np.array_equal(best_indices, reference[count][1])),
                      maximum_score_difference=float(np.max(np.abs(best_scores-reference[count][0]))), **totals))
    if args.gui_count:
        gui_indices = indices[:args.gui_count]
        for with_callback in (False, True):
            batches = []
            def progress(percent, message):
                if message.startswith('Indexing batch'):
                    batches.append(message)
            with dask.config.set(scheduler='threads', num_workers=10), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                start, cpu_start = perf_counter(), process_time()
                session.dictionary_index_indices(gui_indices, phase_id=cache.phase_id,
                    keep_n=5, progress_callback=progress if with_callback else None)
                cpu_seconds, seconds = process_time()-cpu_start, perf_counter()-start
            save(dict(method='application_with_progress' if with_callback else 'application_without_progress',
                      count=len(gui_indices), dask_workers=10, seconds=seconds,
                      cpu_seconds=cpu_seconds, patterns_per_second=len(gui_indices)/seconds,
                      batches=batches))
    session.close()


if __name__ == '__main__':
    main()
