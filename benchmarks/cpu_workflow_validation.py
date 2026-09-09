"""Validate CPU fits through residual indexing and Step 4 on a saved workflow.

Reads source data/dictionary; writes only the requested report and a temporary
workflow used for round-trip verification. The comparison fitter is extracted
from the repository's pre-optimization commit, not from the current wrappers.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core


def reference_fitter(name='_fit_overlap_mixture_pattern'):
    source = subprocess.run(['git', 'show', '301ba4d:multistep_overlap_ebsd/core.py'],
                            check=True, capture_output=True, text=True).stdout
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = vars(core).copy()
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pre-optimization mixture>', 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--pairs', type=Path, default=Path('/tmp/step3_workflow_pairs.npz'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/cpu_workflow_validation.json'))
    args = parser.parse_args()
    with np.load(args.pairs) as data:
        indices = data['indices'].copy()
    session = core.WorkflowSession()
    print(session.restore_workflow_state(args.workflow), flush=True)
    ui = session.restored_ui_state
    settings = dict(fit_maxiter=int(ui['gain_fit_maxiter']), fit_popsize=int(ui['gain_fit_popsize']),
                    fit_bounds=ui['primary_fit_bounds'])
    timings = {}
    records = []

    def progress(percent, message):
        print(f'{percent:.1f}% {message}', flush=True)

    for name, action in (
        ('residuals', lambda: session.compute_overlap_residual_indices(indices, **settings, fit_method='staged')),
        ('residual_indexing', lambda: session.index_overlap_residual_indices(indices, keep_n=5, progress_callback=progress)),
        ('residual_refinement', lambda: session.refine_overlap_residual_indices(
            indices, trust_euler_deg=float(ui['trust_euler']), maxfev=int(ui['maxfev']),
            use_full_resolution=True, progress_callback=progress)),
    ):
        start = perf_counter()
        print(action(), flush=True)
        timings[name] = perf_counter()-start
    baseline = reference_fitter()
    for index in indices:
        idx = int(index)
        experimental = session._processed_pattern_at(idx)
        primary = session._simulate_pattern_for_euler(idx, session.current_eulers_rad[idx])
        secondary = session._simulate_pattern_for_euler(idx, session.residual_eulers_rad[idx])
        for method in ('baseline', 'staged', 'staged_joint'):
            start = perf_counter()
            fitter = baseline if method == 'baseline' else core._fit_overlap_mixture_pattern
            fit = fitter(experimental, primary, secondary, session._overlap_weights(),
                         maxiter=settings['fit_maxiter'], popsize=settings['fit_popsize'],
                         seed=idx+7919, fit_bounds=settings['fit_bounds'],
                         **({} if method == 'baseline' else dict(fit_method=method)))
            record = dict(index=idx, method=method, seconds=perf_counter()-start,
                          ncc=fit.ncc_mixture, residual_rms=fit.residual_rms,
                          primary_fraction=fit.primary_fraction, secondary_fraction=fit.secondary_fraction,
                          sigma=fit.sigma, success=fit.success)
            assert fit.primary_coefficient >= 0 and fit.secondary_coefficient >= 0
            if np.isfinite(fit.primary_fraction):
                assert abs(fit.primary_fraction+fit.secondary_fraction-1) < 1e-12
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(dict(timings=timings, fits=records), indent=2)+'\n')

    start = perf_counter()
    print(session.compute_overlap_mixture_indices(indices, **settings, fit_method='staged_joint',
          parallel_cores=3, selected_index=int(indices[0]), progress_callback=progress), flush=True)
    timings['parallel_mixture_3cores'] = perf_counter()-start
    selected = session.get_overlap_mixture_result(int(indices[0]))
    assert selected.residual.shape == (session.data.h, session.data.w)
    with tempfile.TemporaryDirectory(prefix='overlap-cpu-validation-') as directory:
        workflow = str(Path(directory)/'workflow.npz')
        session.save_workflow_state(workflow, ui_state={**ui, 'fit_method': 'staged_joint'})
        reopened = core.WorkflowSession()
        try:
            reopened.restore_workflow_state(workflow)
            assert reopened.restored_ui_state['fit_method'] == 'staged_joint'
            for idx in indices:
                before = session.get_residual_point_result(int(idx))
                after = reopened.get_residual_point_result(int(idx))
                np.testing.assert_array_equal(before.residual, after.residual)
                a = session.get_overlap_mixture_result(int(idx))
                b = reopened.get_overlap_mixture_result(int(idx))
                np.testing.assert_array_equal(a.residual, b.residual)
                assert a.primary_fraction == b.primary_fraction
        finally:
            reopened.close()
    session.close()
    args.output.write_text(json.dumps(dict(timings=timings, fits=records,
                           validation='Residual indexing/refinement, parallel Step 4 and exact workflow reconstruction passed.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
