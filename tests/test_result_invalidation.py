from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from multistep_overlap_ebsd.core import OverlapPointResult, ResidualBatchPayload, WorkflowSession, _compute_residual_roi_batch


class ResultInvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = WorkflowSession()
        self.session.data = SimpleNamespace(
            rows=1, cols=2, count=2, h=4, w=4, source_type="up_ang",
            pc_output_convention="bruker", sample_tilt_deg=70.0,
            detector_tilt_deg=0.0, azimuthal_deg=0.0, twist_deg=0.0,
        )
        self.session.current_eulers_rad = np.zeros((2, 3), dtype=np.float64)
        self.session.current_phases = np.ones(2, dtype=np.int32)
        self.session.current_pc_custom = np.full((2, 3), 0.5)
        self.session.current_pc_bruker = np.full((2, 3), 0.5)
        self.session.last_scores_map = np.array([[0.9, 0.95]])
        self.session.indexed_mask = np.ones(2, dtype=bool)
        self.session.last_indexed_indices = np.array([0, 1])
        self.session.indexed_candidate_eulers_rad = np.ones((2, 2, 3))
        self.session._ensure_residual_state()
        self.session.residual_eulers_rad[:] = 0.2
        self.session.last_residual_scores_map[:] = 0.7
        self.session.last_residual_indexed_indices = np.array([0, 1])
        self.session.residual_candidate_eulers_rad = np.ones((2, 2, 3))
        for index in range(2):
            self.session.residual_point_results[index] = self._result(index)
            self.session.overlap_mixture_results[index] = SimpleNamespace(index=index)
        self.session.last_overlap = self.session.residual_point_results[0]
        self.session.last_overlap_mixture = self.session.overlap_mixture_results[0]
        self.session._ensure_overlap_mixture_state()
        self.session.overlap_primary_fraction_map[:] = 0.6
        self.session.overlap_secondary_fraction_map[:] = 0.4
        self.session.overlap_mixture_ncc_map[:] = 0.99
        self.session.residual_pattern_output_path = "/tmp/old-residuals.up1"

    @staticmethod
    def _result(index: int) -> OverlapPointResult:
        return OverlapPointResult(
            index=index, row=0, col=index, ncc_es=0.9, scale=0.9,
            ncc_residual_sim=0.0, experimental=np.ones((4, 4)),
            simulated=np.ones((4, 4)), residual=np.zeros((4, 4)),
            secondary_euler_rad=np.full(3, 0.2), secondary_ncc_kp=0.7,
            secondary_simulated=np.ones((4, 4)),
        )

    def _prepare_residual_simulation(self) -> None:
        s = self.session
        s.master = SimpleNamespace(kind="legacy")
        experimental = np.arange(16, dtype=np.float32).reshape(4, 4)
        simulation = experimental.copy()
        simulation[1, 1] += 3.0
        s._processed_pattern_at = Mock(return_value=experimental)
        s._simulate_pattern_for_euler = Mock(return_value=simulation)

    def test_edit_invalidates_only_the_changed_point_and_its_scores(self) -> None:
        s = self.session
        other_residual = s.residual_point_results[1]
        other_mixture = s.overlap_mixture_results[1]
        s.set_point_state(0, euler_deg=(90.0, 90.0, 90.0))

        self.assertIsNone(s.get_primary_index_ncc(0))
        self.assertEqual(s.get_primary_index_ncc(1), 0.95)
        np.testing.assert_array_equal(s.indexed_mask, [False, True])
        np.testing.assert_array_equal(s.last_indexed_indices, [1])
        np.testing.assert_array_equal(s.last_residual_indexed_indices, [1])
        self.assertTrue(np.isnan(s.indexed_candidate_eulers_rad[0]).all())
        self.assertTrue(np.isfinite(s.indexed_candidate_eulers_rad[1]).all())
        self.assertTrue(np.isnan(s.residual_eulers_rad[0]).all())
        self.assertTrue(np.isnan(s.last_residual_scores_map[0, 0]))
        self.assertNotIn(0, s.residual_point_results)
        self.assertNotIn(0, s.overlap_mixture_results)
        self.assertIsNone(s.last_overlap)
        self.assertIsNone(s.last_overlap_mixture)
        self.assertIs(s.residual_point_results[1], other_residual)
        self.assertIs(s.overlap_mixture_results[1], other_mixture)
        self.assertAlmostEqual(float(s.overlap_mixture_ncc_map[0, 1]), 0.99)
        self.assertIsNone(s.residual_pattern_output_path)

    def test_invalid_edit_is_atomic_and_unchanged_apply_keeps_results(self) -> None:
        s = self.session
        original = s.current_eulers_rad.copy()
        with self.assertRaises(ValueError):
            s.set_point_state(0, euler_deg=(90.0, 90.0, 90.0), pc_custom=(0.5, 0.5, -1.0))
        np.testing.assert_array_equal(s.current_eulers_rad, original)
        self.assertEqual(s.get_primary_index_ncc(0), 0.9)
        result = s.residual_point_results[0]
        message = s.set_point_state(0, euler_deg=(0.0, 0.0, 0.0), pc_custom=(0.5, 0.5, 0.5))
        self.assertIn("No changes", message)
        self.assertIs(s.residual_point_results[0], result)

    def test_pc_edit_discards_score_and_dictionary_but_keeps_other_point(self) -> None:
        s = self.session
        s.dictionary_cache = SimpleNamespace(owns_storage=False, storage_path="dictionary.h5")
        s.calibrated_center_pc_bruker = np.full(3, 0.5)
        s.calibrated_center_pc_custom = np.full(3, 0.5)
        s.set_point_state(0, pc_custom=(0.4, 0.5, 0.6))
        self.assertIsNone(s.dictionary_cache)
        self.assertIsNone(s.get_primary_index_ncc(0))
        self.assertEqual(s.get_primary_index_ncc(1), 0.95)
        self.assertIn(1, s.residual_point_results)
        np.testing.assert_allclose(s.calibrated_center_pc_bruker, s.current_pc_bruker[0])

    def test_rebuilt_residual_clears_old_indexing_and_mixture(self) -> None:
        self._prepare_residual_simulation()
        s = self.session
        new = s.analyze_overlap_point(0, fit_blur_gain=False)
        self.assertIsNone(new.secondary_euler_rad)
        self.assertIs(s.get_residual_point_result(0), new)
        self.assertIs(s.last_overlap, new)
        self.assertTrue(np.isnan(s.residual_eulers_rad[0]).all())
        self.assertTrue(np.isnan(s.last_residual_scores_map[0, 0]))
        self.assertTrue(np.isnan(s.residual_candidate_eulers_rad[0]).all())
        self.assertNotIn(0, s.overlap_mixture_results)
        self.assertIn(1, s.overlap_mixture_results)
        self.assertAlmostEqual(float(s.last_residual_scores_map[0, 1]), 0.7)
        self.assertEqual(s.get_primary_index_ncc(0), 0.9)
        self.assertIsNone(s.residual_pattern_output_path)

    def test_partial_roi_working_file_cannot_supply_unwritten_residual_rows(self) -> None:
        self._prepare_residual_simulation()
        s = self.session
        writer = SimpleNamespace(output_path="/tmp/new-residuals.up1", write=Mock(), close=Mock())
        with patch("multistep_overlap_ebsd.core.ResidualPatternWriter.create", return_value=writer):
            s.compute_overlap_residual_indices(
                np.array([0]), fit_blur_gain=False, write_patterns=True,
                residual_output_path=writer.output_path, parallel_cores=1,
            )
        writer.write.assert_called_once()
        writer.close.assert_called_once()
        self.assertIsNone(s.residual_pattern_output_path)
        self.assertTrue(np.isnan(s.residual_eulers_rad[0]).all())
        self.assertTrue(np.isnan(s.last_residual_scores_map[0, 0]))
        self.assertNotIn(0, s.overlap_mixture_results)
        self.assertIn(1, s.overlap_mixture_results)
        self.assertEqual(s.get_primary_index_ncc(0), 0.9)

    def test_roi_working_file_is_reused_when_it_covers_all_retained_results(self) -> None:
        self._prepare_residual_simulation()
        s = self.session
        writer = SimpleNamespace(output_path="/tmp/new-residuals.up1", write=Mock(), close=Mock())
        with patch("multistep_overlap_ebsd.core.ResidualPatternWriter.create", return_value=writer):
            s.compute_overlap_residual_indices(
                np.array([0, 1]), fit_blur_gain=False, write_patterns=True,
                residual_output_path=writer.output_path, parallel_cores=1,
            )
        self.assertEqual(writer.write.call_count, 2)
        self.assertEqual(s.residual_pattern_output_path, writer.output_path)

    def test_conditioning_change_invalidates_all_scores_and_candidates(self) -> None:
        s = self.session
        s.set_pattern_mask_option(0)
        self.assertFalse(s.indexed_mask.any())
        self.assertTrue(np.isnan(s.last_scores_map).all())
        self.assertIsNone(s.last_indexed_indices)
        self.assertIsNone(s.indexed_candidate_eulers_rad)
        self.assertIsNone(s.residual_eulers_rad)
        self.assertFalse(s.residual_point_results)
        self.assertFalse(s.overlap_mixture_results)
        s.set_pattern_mask_option(-1)
        self.assertIsNone(s.get_primary_index_ncc(0))

    def test_manual_blur_is_applied_by_serial_and_parallel_residual_code(self) -> None:
        self._prepare_residual_simulation()
        s = self.session
        s.compute_overlap_residual_indices(np.array([0]), blur_sigma=1.25, fit_blur_gain=False)
        self.assertEqual(s.residual_point_results[0].fitted_sigma, 1.25)
        payload = ResidualBatchPayload(
            indices=np.array([0]), experimental=np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            eulers_rad=np.zeros((1, 3)), pc_bruker=np.full((1, 3), 0.5), pc_custom=np.full((1, 3), 0.5),
        )
        state = dict(weights=np.ones((4, 4)), fit_blur_gain=False, fit_maxiter=1,
                     fit_popsize=1, fit_bounds=None, cols=2, master_kind="legacy", blur_sigma=1.25)
        with patch("multistep_overlap_ebsd.core._RESIDUAL_ROI_WORKER_STATE", state), patch(
            "multistep_overlap_ebsd.core._residual_roi_worker_simulated_pattern",
            return_value=s._simulate_pattern_for_euler(0, np.zeros(3)),
        ):
            results = _compute_residual_roi_batch(payload)
        self.assertEqual(results[0].fitted_sigma, 1.25)
        np.testing.assert_allclose(results[0].residual, s.get_residual_point_result(0).residual, atol=1e-6)

    def test_cancelled_residual_writer_aborts_and_keeps_completed_fit_valid(self) -> None:
        self._prepare_residual_simulation()
        s = self.session
        untouched = s.residual_point_results[1]
        writer = SimpleNamespace(output_path="/tmp/cancelled-residuals.up1", write=Mock(), close=Mock())

        def progress(value: float, _message: str) -> None:
            if value > 0.0:
                raise InterruptedError("Cancelled")

        with patch("multistep_overlap_ebsd.core.ResidualPatternWriter.create", return_value=writer):
            with self.assertRaises(InterruptedError):
                s.compute_overlap_residual_indices(
                    np.array([0, 1]), fit_blur_gain=False, write_patterns=True,
                    residual_output_path=writer.output_path, parallel_cores=1, progress_callback=progress,
                )
        writer.close.assert_called_once_with(commit=False)
        self.assertIsNone(s.residual_pattern_output_path)
        self.assertIsNone(s.residual_point_results[0].secondary_euler_rad)
        self.assertTrue(np.isnan(s.last_residual_scores_map[0, 0]))
        self.assertIs(s.residual_point_results[1], untouched)
        self.assertEqual(s.get_primary_index_ncc(0), 0.9)

    def test_parallel_cancellation_does_not_restart_residual_or_mixture_work(self) -> None:
        s = self.session
        s.data.cols = 4
        s.data.count = 4
        s.data.rot_sd = np.eye(3)
        s.data.direction_cosines = None
        s.current_eulers_rad = np.zeros((4, 3))
        s.current_phases = np.ones(4, dtype=np.int32)
        s.current_pc_custom = np.full((4, 3), 0.5)
        s.current_pc_bruker = np.full((4, 3), 0.5)
        s._ensure_overlap_mixture_state()
        s.master = SimpleNamespace(kind="legacy", path="unused", energy_kv=None,
                                   energy_mode="highest", energy_values_kv=None, energy_weights=None)
        s._parallel_worker_count = Mock(return_value=2)
        s.analyze_overlap_point = Mock()
        s.fit_overlap_mixture_point = Mock()
        selected = np.arange(4)
        s._overlap_mixture_inputs_for_indices = Mock(return_value=(
            selected, np.zeros((4, 3)), np.full(4, 0.9), np.full(4, 0.7), 0,
        ))
        pool = Mock()
        pool.__enter__ = Mock(side_effect=InterruptedError("Cancelled"))
        pool.__exit__ = Mock(return_value=False)
        with patch("multistep_overlap_ebsd.core.ProcessPoolExecutor", return_value=pool):
            with self.assertRaises(InterruptedError):
                s.compute_overlap_residual_indices(selected, parallel_cores=2)
            with self.assertRaises(InterruptedError):
                s.compute_overlap_mixture_indices(selected, parallel_cores=2)
        s.analyze_overlap_point.assert_not_called()
        s.fit_overlap_mixture_point.assert_not_called()

    def _prepare_dictionary_backend(self) -> None:
        s = self.session
        s.master = SimpleNamespace(kind="kikuchipy", mp_signal=object(), phase=object(), energy_kv=20.0)
        s.dictionary_cache = SimpleNamespace(
            phase_id=1, software_binning=1, crop_extent=(0, 4, 0, 4),
            rotation_count=2, owns_storage=False,
        )
        s._signal_mask_for_dictionary_cache = Mock(return_value=None)
        s._dictionary_n_per_iteration = Mock(return_value=None)
        s._dictionary_index_batch_size = Mock(return_value=1)
        s._signal_from_indices = Mock(return_value=object())
        s._residual_signal_from_indices = Mock(return_value=object())
        s._dictionary_index_kikuchipy_signal = Mock(return_value=(
            np.full((2, 3), 0.1), np.array([0.8, 0.8]), np.full((2, 1, 3), 0.1), np.full((2, 1), 0.8),
        ))

    @staticmethod
    def _cancel_after_completed_work(value: float, _message: str) -> None:
        if value > 0.0:
            raise InterruptedError("Cancelled after completed batch")

    def test_cancelled_primary_indexing_commits_completed_batch_validity(self) -> None:
        self._prepare_dictionary_backend()
        s = self.session
        s.indexed_mask[:] = False
        s.last_scores_map[:] = np.nan
        s.last_indexed_indices = None
        s._orientation_color_cache["IPF-Z"] = np.zeros((1, 2, 3))
        with self.assertRaises(InterruptedError):
            s.dictionary_index_indices(
                np.array([0, 1]), phase_id=1, keep_n=1,
                progress_callback=self._cancel_after_completed_work,
            )
        np.testing.assert_array_equal(s.indexed_mask, [True, False])
        np.testing.assert_array_equal(s.last_indexed_indices, [0])
        self.assertEqual(s.get_primary_index_ncc(0), 0.8)
        self.assertIsNone(s.get_primary_index_ncc(1))
        self.assertNotIn("IPF-Z", s._orientation_color_cache)
        self.assertNotIn(0, s.residual_point_results)
        self.assertIn(1, s.residual_point_results)
        s._dictionary_index_kikuchipy_signal.assert_called_once()

    def test_cancelled_legacy_indexing_commits_completed_point_validity(self) -> None:
        s = self.session
        pattern = np.arange(16, dtype=np.float32).reshape(4, 4)
        s.master = SimpleNamespace(kind="legacy", projector=SimpleNamespace(project=Mock(return_value=pattern)))
        s.data.rot_sd = np.eye(3)
        s.data.direction_cosines = None
        s._processed_pattern_at = Mock(return_value=pattern)
        s.indexed_mask[:] = False
        s.last_scores_map[:] = np.nan
        s.last_indexed_indices = None
        s._orientation_color_cache["IPF-Z"] = np.zeros((1, 2, 3))
        rotations = SimpleNamespace(to_euler=lambda: np.array([[0.1, 0.2, 0.3]]))
        with patch("orix.sampling.get_sample_fundamental", return_value=rotations):
            with self.assertRaises(InterruptedError):
                s.dictionary_index_indices(
                    np.array([0, 1]), phase_id=1, keep_n=1,
                    progress_callback=self._cancel_after_completed_work,
                )
        np.testing.assert_array_equal(s.indexed_mask, [True, False])
        np.testing.assert_array_equal(s.last_indexed_indices, [0])
        self.assertGreater(s.get_primary_index_ncc(0), 0.99)
        self.assertIsNone(s.get_primary_index_ncc(1))
        self.assertNotIn("IPF-Z", s._orientation_color_cache)

    def test_cancelled_residual_indexing_records_completed_batch(self) -> None:
        self._prepare_dictionary_backend()
        s = self.session
        s.last_residual_indexed_indices = None
        s._orientation_color_cache["RES-IPF-Z"] = np.zeros((1, 2, 3))
        with self.assertRaises(InterruptedError):
            s.index_overlap_residual_indices(
                np.array([0, 1]), keep_n=1, progress_callback=self._cancel_after_completed_work,
            )
        np.testing.assert_array_equal(s.last_residual_indexed_indices, [0])
        self.assertAlmostEqual(float(s.last_residual_scores_map[0, 0]), 0.8)
        self.assertAlmostEqual(float(s.last_residual_scores_map[0, 1]), 0.7)
        self.assertNotIn("RES-IPF-Z", s._orientation_color_cache)
        self.assertNotIn(0, s.overlap_mixture_results)
        self.assertIn(1, s.overlap_mixture_results)
        s._dictionary_index_kikuchipy_signal.assert_called_once()

    def test_cancelled_residual_refinement_commits_cache_and_index_updates(self) -> None:
        self._prepare_dictionary_backend()
        s = self.session
        result = self._result(0)
        result.secondary_ncc_kp = 0.85
        result.secondary_refined = True
        s._batch_refine_residual_points = Mock(return_value=[result])
        s._kikuchipy_refinement_binning_settings = Mock(return_value=(1, (0, 4, 0, 4), None, "full"))
        s.last_residual_indexed_indices = None
        s._orientation_color_cache["RES-IPF-Z"] = np.zeros((1, 2, 3))
        with self.assertRaises(InterruptedError):
            s.refine_overlap_residual_indices(
                np.array([0]), progress_callback=self._cancel_after_completed_work,
            )
        np.testing.assert_array_equal(s.last_residual_indexed_indices, [0])
        self.assertAlmostEqual(float(s.last_residual_scores_map[0, 0]), 0.85)
        self.assertNotIn("RES-IPF-Z", s._orientation_color_cache)
        self.assertNotIn(0, s.overlap_mixture_results)
        self.assertIn(1, s.overlap_mixture_results)

    def test_index_refine_residual_sequence_retains_new_primary_validity(self) -> None:
        s = self.session
        s.master = SimpleNamespace(kind="kikuchipy", mp_signal=object(), phase=object(), energy_kv=20.0)
        s.dictionary_cache = SimpleNamespace(
            phase_id=1, software_binning=1, crop_extent=(0, 4, 0, 4),
            rotation_count=2, owns_storage=False,
        )
        s._signal_mask_for_dictionary_cache = Mock(return_value=None)
        s._dictionary_n_per_iteration = Mock(return_value=None)
        s._dictionary_index_batch_size = Mock(return_value=2)
        refined = SimpleNamespace(
            rotations=SimpleNamespace(to_euler=lambda: np.full((2, 3), 0.15)),
            prop={"scores": np.array([0.85, 0.85])},
        )
        signal = SimpleNamespace(refine_orientation=Mock(return_value=refined))
        s._signal_from_indices = Mock(return_value=signal)
        s._dictionary_index_kikuchipy_signal = Mock(return_value=(
            np.full((2, 3), 0.1), np.array([0.8, 0.8]), np.full((2, 1, 3), 0.1), np.full((2, 1), 0.8),
        ))
        s.dictionary_index_indices(np.array([0]), phase_id=1, keep_n=1)
        self.assertEqual(s.get_primary_index_ncc(0), 0.8)
        self.assertIn(1, s.residual_point_results)
        s.residual_point_results[0] = self._result(0)
        s._kikuchipy_refinement_binning_settings = Mock(return_value=(1, (0, 4, 0, 4), None, "full"))
        s._phase_list_for_current_master = Mock(return_value=None)
        s._kikuchipy_detector_for_indices = Mock(return_value=object())
        with patch("orix.crystal_map.CrystalMap", return_value=object()):
            s.refine_orientations_indices(np.array([0]), phase_id=1)
        self.assertEqual(s.get_primary_index_ncc(0), 0.85)
        self.assertEqual(s.get_primary_index_ncc(1), 0.95)
        self.assertTrue(s.indexed_mask[0])
        np.testing.assert_array_equal(s.last_indexed_indices, [0])
        self.assertNotIn(0, s.residual_point_results)
        self.assertIn(1, s.residual_point_results)
        self._prepare_residual_simulation()
        s.compute_overlap_residual_indices(np.array([0]), fit_blur_gain=False, parallel_cores=1)
        self.assertEqual(s.get_primary_index_ncc(0), 0.85)
        self.assertIn(0, s.residual_point_results)
        s.master.kind = "kikuchipy"
        s._residual_signal_from_indices = Mock(return_value=signal)
        s.index_overlap_residual_indices(np.array([0]), keep_n=1)
        self.assertTrue(np.isfinite(s.residual_eulers_rad[0]).all())
        self.assertAlmostEqual(float(s.last_residual_scores_map[0, 0]), 0.8)
        self.assertEqual(s.get_primary_index_ncc(0), 0.85)
        self.assertIn(1, s.overlap_mixture_results)


if __name__ == "__main__":
    unittest.main()
