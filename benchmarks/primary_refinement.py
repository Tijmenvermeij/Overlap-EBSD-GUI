"""Paired primary refinement benchmark, keeping all candidate seeds/settings.

Use --reference-source with core.py saved before the primary refinement change.
The reference methods use their original adaptive batching. Input files are read
only; fits modify the benchmark session, and are reset before every measurement.
"""
import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from time import perf_counter, process_time
from types import MethodType
from unittest.mock import patch

import dask
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core


def adaptive_batches(indices, maximum, progress_callback=None, *, initial_size=16):
    size = min(maximum, initial_size) if progress_callback else maximum
    start = 0
    while start < len(indices):
        batch = indices[start:start+size]
        before = perf_counter()
        yield start, batch
        start += len(batch)
        if progress_callback:
            size = min(maximum, size*4, max(1, int(len(batch)*5/max(perf_counter()-before, .001))))


def initial_batch(candidate_count):
    return max(16, min(128, (64*dask.config.get('num_workers')+candidate_count-1)//candidate_count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=640)
    parser.add_argument('--cores', type=int, default=6)
    parser.add_argument('--binned', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--residual', action='store_true')
    args = parser.parse_args()
    refine_name = 'refine_overlap_residual_indices' if args.residual else 'refine_orientations_indices'
    namespace = dict(vars(core), progress_batches=adaptive_batches, refinement_initial_batch=initial_batch)
    source = ast.parse(args.reference_source.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'WorkflowSession')
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_run_kikuchipy_refinement', refine_name):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference>', 'exec'), namespace)
    session = core.WorkflowSession()
    records = [dict(count=args.count, cores=args.cores, keep_n=5, maxfev=25,
                    full_resolution=not args.binned, residual=args.residual)]
    try:
        print(session.restore_workflow_state(args.workflow), flush=True)
        phase = session.dictionary_cache.phase_id
        if args.residual:
            selected = np.array(sorted(session.residual_point_results))[3000:3000+args.count]
            session.index_overlap_residual_indices(selected, keep_n=5, parallel_cores=args.cores)
            candidate_store, euler_store = session.residual_candidate_eulers_rad, session.residual_eulers_rad
            metadata = {int(i): session._strip_residual_point_result(session.residual_point_results[int(i)]) for i in selected}
        else:
            selected = np.arange(3000, 3000+args.count)
            session.dictionary_index_indices(selected, phase_id=phase, keep_n=5, parallel_cores=args.cores)
            candidate_store, euler_store = session.indexed_candidate_eulers_rad, session.current_eulers_rad
        seeds, eulers = candidate_store[selected].copy(), euler_store[selected].copy()
        settings = dict(maxfev=25, trust_euler_deg=1.4, parallel_cores=args.cores,
                        use_full_resolution=not args.binned, progress_callback=lambda value, message: None)
        if not args.residual:
            settings['phase_id'] = phase
        reference_backend = MethodType(namespace['_run_kikuchipy_refinement'], session)
        reference_refine = MethodType(namespace[refine_name], session)
        implemented_refine = getattr(session, refine_name)
        implemented_backend = session._run_kikuchipy_refinement
        # Record first-use setup/compilation separately from steady throughput.
        with redirect_stdout(io.StringIO()):
            start = perf_counter()
            implemented_refine(selected[:2], **settings)
            records.append(dict(label='compiled_first_use_two_points', seconds=perf_counter()-start))
        expected = None
        for repeat in range(args.repeat):
            for label in ('reference', 'shared_pc_maximum_batch', 'implemented'):
                euler_store[selected], candidate_store[selected] = eulers, seeds
                if args.residual:
                    session.residual_point_results.update({i: replace(r) for i, r in metadata.items()})
                    session._residual_inspection_indices.clear()
                batches = []
                fit_seconds = [0.]
                def backend(signal, operation, **kwargs):
                    if label == 'shared_pc_maximum_batch':
                        pcs = np.asarray(kwargs['detector'].pc).reshape(-1, 3)
                        if np.all(pcs == pcs[:1]):
                            detector = kwargs['detector'].deepcopy()
                            detector.pc = pcs[:1]
                            kwargs['detector'] = detector
                    batches.append(int(signal.axes_manager.navigation_size))
                    run = implemented_backend if label == 'implemented' else reference_backend
                    start = perf_counter()
                    result = run(signal, operation, **kwargs)
                    fit_seconds[0] += perf_counter()-start
                    return result
                with patch.object(session, '_run_kikuchipy_refinement', side_effect=backend), redirect_stdout(io.StringIO()):
                    run = reference_refine if label == 'reference' else implemented_refine
                    start, cpu_start = perf_counter(), process_time()
                    run(selected, **settings)
                    seconds, cpu_seconds = perf_counter()-start, process_time()-cpu_start
                score_map = session.last_residual_scores_map if args.residual else session.last_scores_map
                actual = (euler_store[selected].copy(), score_map.reshape(-1)[selected].copy())
                if expected is None:
                    expected = actual
                differences = [float(np.max(np.abs(a-b))) for a, b in zip(actual, expected)]
                record = dict(label=label, repeat=repeat, seconds=seconds, fitting_seconds=fit_seconds[0],
                    process_cpu_seconds=cpu_seconds, average_cpu_cores=cpu_seconds/seconds,
                    candidate_fit_batches=batches, max_euler_score_differences=differences)
                records.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(records, indent=2)+'\n')
                print(json.dumps(record), flush=True)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
    finally:
        session.close()


if __name__ == '__main__':
    main()
