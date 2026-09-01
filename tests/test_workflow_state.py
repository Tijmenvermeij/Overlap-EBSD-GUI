from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from multistep_overlap_ebsd.core import (
    OverlapMixtureResult,
    OverlapPointResult,
    WorkflowSession,
    _output_path_with_single_suffix,
)


class WorkflowStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="overlap-ebsd-workflow-test-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _data() -> SimpleNamespace:
        return SimpleNamespace(
            pattern_path="/tmp/source.h5oina",
            orientation_path=None,
            sample_tilt_deg=70.0,
            detector_tilt_deg=0.0,
            azimuthal_deg=0.0,
            twist_deg=0.0,
            rows=2,
            cols=3,
            count=6,
            h=4,
            w=6,
            source_type="h5oina",
        )

    def test_complete_workflow_round_trip_preserves_steps_two_through_four(self) -> None:
        source = WorkflowSession()
        source.data = self._data()
        source.master = SimpleNamespace(path="/tmp/master.h5")
        source.initial_eulers_rad = np.zeros((6, 3), dtype=np.float64)
        source.current_eulers_rad = np.arange(18, dtype=np.float64).reshape(6, 3) / 100.0
        source.current_phases = np.ones(6, dtype=np.int32)
        source.current_pc_bruker = np.full((6, 3), 0.5, dtype=np.float64)
        source.current_pc_custom = np.full((6, 3), 0.6, dtype=np.float64)
        source.last_scores_map = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
        source.last_indexed_indices = np.array([4, 5], dtype=np.int64)
        source.indexed_mask = np.array([False, True, False, True, True, True])
        source.calibration_indices = [0, 5]
        source.calibrated_center_pc_bruker = np.array([0.4, 0.5, 0.6])
        source.calibrated_center_pc_custom = np.array([0.7, 0.8, 0.9])
        source.dictionary_settings = {
            "phase_id": 1,
            "resolution_deg": 1.2,
            "pc_bruker": np.array([0.4, 0.5, 0.6]),
            "software_binning": 2,
            "crop_extent": np.array([0, 4, 0, 6]),
        }

        source.residual_eulers_rad = np.full((6, 3), np.nan, dtype=np.float64)
        source.residual_eulers_rad[4] = [0.1, 0.2, 0.3]
        source.residual_phases = np.ones(6, dtype=np.int32)
        source.last_residual_scores_map = np.full((2, 3), np.nan, dtype=np.float32)
        source.last_residual_scores_map.reshape(-1)[4] = 0.77
        source.last_residual_indexed_indices = np.array([4], dtype=np.int64)
        residual = OverlapPointResult(
            index=4,
            row=1,
            col=1,
            ncc_es=0.8,
            scale=0.75,
            ncc_residual_sim=0.1,
            experimental=np.ones((2, 2)),
            simulated=np.ones((2, 2)),
            residual=np.zeros((2, 2)),
            fitted_sigma=1.25,
            gain_params=(1.0, 2.0, 3.0),
            ellipse_params=(1.0, 1.0, 0.0, 0.0),
            secondary_ncc_kp=0.77,
            secondary_euler_rad=np.array([0.1, 0.2, 0.3]),
            secondary_refined=True,
        )
        source.residual_point_results[4] = residual
        source.last_overlap = residual

        source.overlap_primary_fraction_map = np.full((2, 3), np.nan, dtype=np.float32)
        source.overlap_secondary_fraction_map = np.full((2, 3), np.nan, dtype=np.float32)
        source.overlap_mixture_ncc_map = np.full((2, 3), np.nan, dtype=np.float32)
        source.overlap_primary_fraction_map.reshape(-1)[4] = 0.65
        source.overlap_secondary_fraction_map.reshape(-1)[4] = 0.35
        source.overlap_mixture_ncc_map.reshape(-1)[4] = 0.91
        mixture = OverlapMixtureResult(
            index=4,
            row=1,
            col=1,
            primary_fraction=0.65,
            secondary_fraction=0.35,
            primary_coefficient=0.7,
            secondary_coefficient=0.3,
            ncc_mixture=0.91,
            residual_rms=0.05,
            old_primary_ncc=0.8,
            old_secondary_ncc=0.77,
            experimental=np.ones((2, 2)),
            primary_simulated=np.ones((2, 2)),
            secondary_simulated=np.ones((2, 2)),
            combined_simulated=np.ones((2, 2)),
            residual=np.zeros((2, 2)),
            fitted_sigma=1.1,
            gain_params=(1.0, 2.0, 3.0),
            ellipse_params=(1.0, 1.0, 0.0, 0.0),
            primary_euler_rad=np.array([0.2, 0.3, 0.4]),
            secondary_euler_rad=np.array([0.1, 0.2, 0.3]),
            orientation_refined=True,
        )
        source.overlap_mixture_results[4] = mixture
        source.last_overlap_mixture = mixture

        workflow_path = self.root / "map_overlap_workflow.npz"
        ui_state = {"roi_r0": 1, "roi_c0": 1, "roi_nrows": 1, "roi_ncols": 2, "selected_workflow_tab": 3}
        source.save_workflow_state(str(workflow_path), ui_state=ui_state)

        restored = WorkflowSession()

        def load_input(_pattern_path, _orientation_path, _geom):
            restored.data = self._data()
            return "Loaded fake input."

        def load_master(master_path):
            restored.master = SimpleNamespace(path=master_path)
            return "Loaded fake master."

        restored.load_input = load_input  # type: ignore[method-assign]
        restored.load_master = load_master  # type: ignore[method-assign]
        restored.restore_workflow_state(str(workflow_path))

        np.testing.assert_array_equal(restored.initial_eulers_rad, source.initial_eulers_rad)
        np.testing.assert_array_equal(restored.current_eulers_rad, source.current_eulers_rad)
        np.testing.assert_array_equal(restored.indexed_mask, source.indexed_mask)
        np.testing.assert_array_equal(restored.last_residual_indexed_indices, np.array([4]))
        self.assertEqual(restored.restored_ui_state, ui_state)
        self.assertIn(4, restored.residual_point_results)
        self.assertTrue(restored.residual_point_results[4].secondary_refined)
        self.assertIsNone(restored.residual_point_results[4].residual)
        self.assertIn(4, restored.overlap_mixture_results)
        self.assertAlmostEqual(restored.overlap_mixture_results[4].primary_fraction, 0.65)
        self.assertAlmostEqual(float(restored.overlap_secondary_fraction_map.reshape(-1)[4]), 0.35)

    def test_repeated_h5oina_suffix_is_collapsed(self) -> None:
        path = _output_path_with_single_suffix(
            str(self.root / "primary_roi.h5oina.h5oina"),
            ".h5oina",
        )
        self.assertEqual(path.name, "primary_roi.h5oina")


if __name__ == "__main__":
    unittest.main()
