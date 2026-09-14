"""Measure the work that per-batch index/refine fusion could eliminate.

Runs the real indexing/refinement stages on a restored workflow. The refinement
input preparation time is an optimistic ceiling on savings from retaining its
input in memory between stages. Records reconstruction calls to distinguish
cache reads from repeated residual fitting/projection. Source files are read only.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--cores', type=int, default=10)
    parser.add_argument('--keep-n', type=int, default=5)
    parser.add_argument('--maxfev', type=int, default=25)
    parser.add_argument('--binned-refinement', action='store_true')
    args = parser.parse_args()
    session = core.WorkflowSession()
    report = dict(count=args.count, cores=args.cores, keep_n=args.keep_n,
                  maxfev=args.maxfev, full_resolution=not args.binned_refinement)
    try:
        print(session.restore_workflow_state(args.workflow), flush=True)
        indices = np.array(sorted(session.residual_point_results))[3000:3000+args.count]
        if len(indices) != args.count:
            raise ValueError('Insufficient residual fits for the requested batch.')
        session.residual_pattern_output_path = None
        stages = {}
        for phase in ('index', 'refine'):
            times, calls, batches = {}, {}, []
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
            names = ('_residual_signal_from_indices', '_materialize_signal_batch',
                     '_processed_patterns_from_indices', '_simulate_patterns_for_eulers',
                     '_materialize_residual_point_result', '_batch_refine_residual_points')
            with ExitStack() as stack:
                for name in names:
                    stack.enter_context(patch.object(session, name, side_effect=timed(name)))
                start = perf_counter()
                with redirect_stdout(io.StringIO()):
                    if phase == 'index':
                        session.index_overlap_residual_indices(
                            indices, keep_n=args.keep_n, parallel_cores=args.cores,
                            progress_callback=lambda value, message: None,
                        )
                    else:
                        session.refine_overlap_residual_indices(
                            indices, maxfev=args.maxfev, trust_euler_deg=1.4,
                            use_full_resolution=not args.binned_refinement,
                            parallel_cores=args.cores, progress_callback=lambda value, message: None,
                        )
                seconds = perf_counter()-start
            stages[phase] = dict(seconds=seconds, times=times, calls=calls, batches=batches)
            print(phase, json.dumps(stages[phase]), flush=True)
        refine = stages['refine']
        assert refine['calls'].get('_simulate_patterns_for_eulers', 0) == 0
        assert refine['calls'].get('_processed_patterns_from_indices', 0) == 0
        assert refine['calls'].get('_materialize_residual_point_result', 0) == 0
        assert np.isfinite(session.residual_eulers_rad[indices]).all()
        assert np.isfinite(session.last_residual_scores_map.reshape(-1)[indices]).all()
        preparation = sum(refine['times'].get(name, 0.) for name in
                          ('_residual_signal_from_indices', '_materialize_signal_batch'))
        total = sum(stage['seconds'] for stage in stages.values())
        report.update(stages=stages, total_seconds=total,
                      refinement_input_seconds=preparation,
                      optimistic_saved_percent=100*preparation/total,
                      residuals_reconstructed_during_refinement=0)
    finally:
        session.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'stages'}), flush=True)


if __name__ == '__main__':
    main()
