"""Compare residual refinement batch policies on identical dictionary candidates.

Pass --reference-source pointing to core.py captured before the change. Only the
two residual refinement methods are restored; other CPU improvements stay active.
Source patterns/workflow/dictionary are read only. Fits update in-memory state.
"""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from types import MethodType
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core
from primary_refinement import adaptive_batches, initial_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=320)
    parser.add_argument('--offset', type=int, default=3000)
    parser.add_argument('--cores', type=int, default=10)
    parser.add_argument('--keep-n', type=int, default=5)
    parser.add_argument('--maxfev', type=int, default=25)
    args = parser.parse_args()
    names = ('refine_overlap_residual_indices', '_batch_refine_residual_points')
    source = ast.parse(args.reference_source.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'WorkflowSession')
    namespace = dict(vars(core), progress_batches=adaptive_batches, refinement_initial_batch=initial_batch)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference>', 'exec'), namespace)
    session = core.WorkflowSession()
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(records, indent=2)+'\n')

    try:
        print(session.restore_workflow_state(args.workflow), flush=True)
        indices = np.array(sorted(session.residual_point_results))[args.offset:args.offset+args.count]
        if len(indices) != args.count:
            raise ValueError('Insufficient residual fits for the requested selection.')
        session.index_overlap_residual_indices(indices, keep_n=args.keep_n, parallel_cores=args.cores)
        seeds = session.residual_candidate_eulers_rad.copy()
        metadata = {int(i): session._strip_residual_point_result(session.residual_point_results[int(i)]) for i in indices}
        # Prime common JIT/master setup before the paired measurements.
        session._refinement_master()
        session._simulate_pattern_for_euler(int(indices[0]), session.current_eulers_rad[indices[0]])
        expected = None
        records.append(dict(count=args.count, cores=args.cores, keep_n=args.keep_n,
                            maxfev=args.maxfev, full_resolution=True, map_count=session.data.count))
        for label in ('reference', 'deferred_images_original_batches', 'implemented', 'fixed_maximum_batch'):
            session.residual_point_results.update({i: replace(r) for i, r in metadata.items()})
            session.residual_candidate_eulers_rad = seeds.copy()
            session._residual_inspection_indices.clear()
            batches = []
            times = {}
            calls = {}
            def timed(name):
                method = getattr(session, name)
                def run(*a, **kw):
                    start = perf_counter()
                    try:
                        return method(*a, **kw)
                    finally:
                        times[name] = times.get(name, 0.) + perf_counter()-start
                        calls[name] = calls.get(name, 0) + 1
                        if name == '_batch_refine_residual_points':
                            batches.append(len(a[0]))
                return run
            with ExitStack() as stack:
                if label in ('reference', 'deferred_images_original_batches'):
                    stack.enter_context(patch.object(session, names[0], MethodType(namespace[names[0]], session)))
                if label == 'reference':
                    stack.enter_context(patch.object(session, names[1], MethodType(namespace[names[1]], session)))
                if label == 'fixed_maximum_batch':
                    original = core.progress_batches
                    def fixed(points, maximum, progress_callback=None, **kwargs):
                        return original(points, maximum)
                    stack.enter_context(patch.object(core, 'progress_batches', fixed))
                for name in ('_batch_refine_residual_points', '_refine_orientation_signal', '_simulate_pattern_for_euler'):
                    stack.enter_context(patch.object(session, name, side_effect=timed(name)))
                start = perf_counter()
                with redirect_stdout(io.StringIO()):
                    session.refine_overlap_residual_indices(
                        indices, trust_euler_deg=1.4, maxfev=args.maxfev, parallel_cores=args.cores,
                        selected_index=int(indices[0]), progress_callback=lambda value, message: None,
                    )
                    inspected = session.get_residual_point_result(int(indices[0]))
                seconds = perf_counter()-start
            result = (session.residual_eulers_rad[indices].copy(),
                      session.last_residual_scores_map.reshape(-1)[indices].copy(),
                      inspected.secondary_simulated.copy(), inspected.residual.copy())
            if expected is None:
                expected = result
            for actual, reference in zip(result, expected):
                np.testing.assert_array_equal(actual, reference)
            record = dict(label=label, seconds=seconds, batches=batches, times=times, calls=calls,
                          eulers_scores_and_selected_patterns_equal=True)
            records.append(record)
            save()
            print(json.dumps(record), flush=True)
    finally:
        session.close()
        save()


if __name__ == '__main__':
    main()
