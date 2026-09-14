"""Compare implemented CPU indexing/refinement with f315c75 on a saved workflow.

Reads source files; updates only the in-memory session and this JSON report.
"""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import MethodType
from unittest.mock import patch

import dask
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core
from primary_refinement import adaptive_batches


def reference_methods():
    source = subprocess.run(['git', 'show', 'f315c75:multistep_overlap_ebsd/core.py'],
                            check=True, capture_output=True, text=True).stdout
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'WorkflowSession')
    names = ('_dictionary_index_kikuchipy_signal', '_dictionary_index_kikuchipy',
             '_dictionary_n_per_iteration', 'refine_orientations_indices')
    namespace = {**vars(core), 'KIKUCHIPY_PARALLEL_REFINEMENT_RECHUNK': True,
                 'progress_batches': adaptive_batches}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<f315c75>', 'exec'), namespace)
    return {name: namespace[name] for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=512)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = core.WorkflowSession()
    print(session.restore_workflow_state(args.workflow), flush=True)
    old = reference_methods()
    records = []

    def measure(label, action):
        start = perf_counter()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = action()
        record = dict(label=label, seconds=perf_counter()-start)
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        return result, record

    try:
        cache = session.dictionary_cache
        rng = np.random.default_rng(20260909)
        indices = np.sort(rng.choice(np.flatnonzero(session.current_phases == cache.phase_id), args.count, replace=False))
        signal = session._materialize_signal_batch(session._signal_from_indices(indices,
            software_binning=cache.software_binning, crop_extent=cache.crop_extent))
        mask = session._signal_mask_for_dictionary_cache(cache)
        kwargs = dict(cache=cache, keep_n=5, signal_mask=mask)
        with dask.config.set(scheduler='threads', num_workers=10):
            reference, _ = measure('dictionary_reference_512', lambda: old['_dictionary_index_kikuchipy_signal'](
                session, signal, n_per_iteration=old['_dictionary_n_per_iteration'](session, cache, mask), **kwargs))
            for label in ('dictionary_first_pass_512', 'dictionary_reused_statistics_512'):
                actual, record = measure(label, lambda: session._dictionary_index_kikuchipy_signal(signal, **kwargs))
                record['candidate_eulers_equal'] = bool(np.array_equal(actual[2], reference[2]))
                record['max_score_difference'] = float(np.max(np.abs(actual[3]-reference[3])))
                np.testing.assert_array_equal(actual[2], reference[2])
                np.testing.assert_allclose(actual[3], reference[3], atol=2e-6, rtol=0)
        progress = lambda value, message: None
        for old_code in (True, False):
            with ExitStack() as stack:
                if old_code:
                    for name in ('_dictionary_index_kikuchipy_signal', '_dictionary_index_kikuchipy', '_dictionary_n_per_iteration'):
                        stack.enter_context(patch.object(session, name, MethodType(old[name], session)))
                _, record = measure('gui_index_24_reference' if old_code else 'gui_index_24_implemented',
                    lambda: session.dictionary_index_indices(indices[:24], cache.phase_id, keep_n=5,
                        parallel_cores=10, progress_callback=progress))
        # Restore identical primary seeds/candidates between repeat-refinement runs.
        selected = indices[:80]
        seed_eulers = session.current_eulers_rad[selected].copy()
        session.indexed_candidate_eulers_rad[selected] = np.nan
        with dask.config.set(scheduler='threads', num_workers=10):
            measure('repeat_refine_80_reference', lambda: old['refine_orientations_indices'](
                session, selected, phase_id=cache.phase_id, trust_euler_deg=1.4, maxfev=25, progress_callback=progress))
        expected_eulers = session.current_eulers_rad[selected].copy()
        expected_scores = session.last_scores_map.reshape(-1)[selected].copy()
        session.current_eulers_rad[selected] = seed_eulers
        _, record = measure('repeat_refine_80_implemented', lambda: session.refine_orientations_indices(
            selected, phase_id=cache.phase_id, trust_euler_deg=1.4, maxfev=25, parallel_cores=10, progress_callback=progress))
        record['eulers_equal'] = bool(np.array_equal(session.current_eulers_rad[selected], expected_eulers))
        record['scores_equal'] = bool(np.array_equal(session.last_scores_map.reshape(-1)[selected], expected_scores))
        np.testing.assert_array_equal(session.current_eulers_rad[selected], expected_eulers)
        np.testing.assert_array_equal(session.last_scores_map.reshape(-1)[selected], expected_scores)
        # Actual residual/mixture pipeline and selected-point refinement.
        small = selected[:6]
        measure('residual_construction_6', lambda: session.compute_overlap_residual_indices(small, parallel_cores=1))
        measure('residual_indexing_6', lambda: session.index_overlap_residual_indices(small, keep_n=5, parallel_cores=10))
        measure('residual_refinement_6', lambda: session.refine_overlap_residual_indices(small, maxfev=25, parallel_cores=10))
        result = session.get_residual_point_result(int(small[0]))
        measure('selected_residual_refinement', lambda: session.refine_overlap_residual(result, maxfev=25, parallel_cores=10))
        measure('mixture_fitting_6', lambda: session.compute_overlap_mixture_indices(small, parallel_cores=1))
        for index in small:
            fit = session.get_overlap_mixture_result(int(index))
            assert np.isfinite(fit.ncc_mixture)
            assert abs(fit.primary_fraction+fit.secondary_fraction-1) < 1e-6
        measure('calibration_3', lambda: session.refine_indices(small[:3], cache.phase_id, maxfev=25, parallel_cores=3))
    finally:
        session.close()
        args.output.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
