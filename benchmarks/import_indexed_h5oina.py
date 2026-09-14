"""Validate exported residuals as indexed primary data in a fresh session.

Source files are read only. Computes and indexes a small next-generation residual
selection without calling primary DI, then closes the temporary session/cache.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd.core import WorkflowSession, GeometryConfig, H5OINA_NCC_DATASET
from multistep_overlap_ebsd.gui import MultiStepOverlapGUI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('master')
    parser.add_argument('dictionary')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    session = WorkflowSession()
    report = {}
    try:
        report['load_message'] = session.load_input(args.input, None, GeometryConfig())
        print(report['load_message'], flush=True)
        root = session.data.h5_analysis_root
        with h5py.File(args.input, 'r') as h5:
            source_scores = np.asarray(h5[f'{root}/{H5OINA_NCC_DATASET}'][()]).reshape(-1)
        mask = session.indexed_mask.copy()
        count = int(np.count_nonzero(mask))
        np.testing.assert_array_equal(session.last_scores_map.reshape(-1)[mask], source_scores[mask])
        report['primary_indexed_on_load'] = count
        assert session.master is None and session.residual_eulers_rad is None
        assert not session.residual_point_results and not session.overlap_mixture_results
        assert session.residual_pattern_output_path is None
        report['new_residual_state_empty'] = True
        before = session.last_scores_map.copy()
        with patch.object(session, 'dictionary_index_indices', side_effect=AssertionError('Primary DI must be skipped')):
            report['master_message'] = session.load_master(args.master)
            np.testing.assert_array_equal(session.last_scores_map, before)
            report['dictionary_message'] = session.load_dictionary(args.dictionary)
            np.testing.assert_array_equal(session.last_scores_map, before)
            np.testing.assert_array_equal(session.indexed_mask, mask)
            report['primary_indexed_after_setup'] = int(np.count_nonzero(session.indexed_mask))
            gui = SimpleNamespace(session=session, _residual_ncc_threshold=lambda: .15)
            excluded = MultiStepOverlapGUI._primary_threshold_mask(gui).reshape(-1)
            selected = np.flatnonzero(~excluded)[:8]
            assert len(selected) == 8
            loaded_patterns = session._processed_patterns_from_indices(selected).copy()
            with redirect_stdout(io.StringIO()):
                report['next_residual_message'] = session.compute_overlap_residual_indices(
                    selected, fit_blur_gain=False, parallel_cores=2)
                report['next_index_message'] = session.index_overlap_residual_indices(
                    selected, keep_n=5, parallel_cores=2)
            assert np.all(np.isfinite(session.residual_eulers_rad[selected]))
            assert np.all(np.isfinite(session.last_residual_scores_map.reshape(-1)[selected]))
            assert len(session.residual_point_results) == 8
            np.testing.assert_array_equal(session.last_scores_map, before)
            np.testing.assert_array_equal(session._processed_patterns_from_indices(selected), loaded_patterns)
            report.update(next_residual_indices=selected.tolist(), next_residuals_indexed=8,
                          primary_di_calls=0, primary_patterns_and_scores_unchanged=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        session.close()


if __name__ == '__main__':
    main()
