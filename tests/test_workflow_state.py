from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from multistep_overlap_ebsd.core import (
    GeometryConfig,
    MASTER_ENERGY_MODE_GLOBAL,
    OverlapMixtureResult,
    OverlapPointResult,
    ResidualPatternWriter,
    WorkflowSession,
    _copy_h5oina_for_map_export,
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
        source.master = SimpleNamespace(
            path="/tmp/master.h5",
            energy_mode=MASTER_ENERGY_MODE_GLOBAL,
            energy_values_kv=np.asarray([18.0, 19.0, 20.0]),
            energy_weights=np.asarray([0.15, 0.35, 0.5]),
            energy_reference_pc_bruker=np.asarray([0.4, 0.5, 0.6]),
        )
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
        linked_dictionary = self.root / "saved_dictionary.h5"
        linked_dictionary.touch()
        source.dictionary_cache = SimpleNamespace(
            owns_storage=False,
            storage_path=str(linked_dictionary),
        )
        source.indexed_candidate_eulers_rad = np.full((6, 2, 3), 0.25, dtype=np.float64)

        source.residual_eulers_rad = np.full((6, 3), np.nan, dtype=np.float64)
        source.residual_eulers_rad[4] = [0.1, 0.2, 0.3]
        source.residual_phases = np.ones(6, dtype=np.int32)
        source.last_residual_scores_map = np.full((2, 3), np.nan, dtype=np.float32)
        source.last_residual_scores_map.reshape(-1)[4] = 0.77
        source.last_residual_indexed_indices = np.array([4], dtype=np.int64)
        source.residual_candidate_eulers_rad = np.full((6, 2, 3), 0.5, dtype=np.float64)
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

        restored_master_kwargs: dict[str, object] = {}
        restored_dictionary_paths: list[str] = []

        def load_master(master_path, **kwargs):
            restored_master_kwargs.update(kwargs)
            restored.master = SimpleNamespace(path=master_path)
            return "Loaded fake master."

        restored.load_input = load_input  # type: ignore[method-assign]
        restored.load_master = load_master  # type: ignore[method-assign]

        def load_dictionary(dictionary_path: str) -> str:
            restored_dictionary_paths.append(dictionary_path)
            # Real dictionary loading reapplies the master-energy model and
            # invalidates these caches. Restore must load it before replaying
            # the workflow's saved Steps 2-4 state.
            restored._invalidate_residual_cache()
            restored.indexed_candidate_eulers_rad = None
            restored.dictionary_cache = SimpleNamespace(
                owns_storage=False,
                storage_path=dictionary_path,
            )
            return "Loaded fake dictionary."

        restored.load_dictionary = load_dictionary  # type: ignore[method-assign]
        restore_message = restored.restore_workflow_state(str(workflow_path))

        np.testing.assert_array_equal(restored.initial_eulers_rad, source.initial_eulers_rad)
        np.testing.assert_array_equal(restored.current_eulers_rad, source.current_eulers_rad)
        np.testing.assert_array_equal(restored.indexed_mask, source.indexed_mask)
        np.testing.assert_array_equal(restored.last_residual_indexed_indices, np.array([4]))
        np.testing.assert_array_equal(restored.indexed_candidate_eulers_rad, source.indexed_candidate_eulers_rad)
        np.testing.assert_array_equal(restored.residual_candidate_eulers_rad, source.residual_candidate_eulers_rad)
        self.assertEqual(restored.restored_ui_state, ui_state)
        self.assertIn(4, restored.residual_point_results)
        self.assertTrue(restored.residual_point_results[4].secondary_refined)
        self.assertIsNone(restored.residual_point_results[4].residual)
        self.assertIn(4, restored.overlap_mixture_results)
        self.assertAlmostEqual(restored.overlap_mixture_results[4].primary_fraction, 0.65)
        self.assertAlmostEqual(float(restored.overlap_secondary_fraction_map.reshape(-1)[4]), 0.35)
        self.assertEqual(restored_master_kwargs["energy_mode"], MASTER_ENERGY_MODE_GLOBAL)
        np.testing.assert_allclose(restored_master_kwargs["energy_weights"], [0.15, 0.35, 0.5])
        np.testing.assert_allclose(
            restored_master_kwargs["energy_reference_pc_bruker"],
            [0.4, 0.5, 0.6],
        )
        self.assertEqual(restored_dictionary_paths, [str(linked_dictionary.resolve())])
        self.assertIn("Restored 1 residual fit(s), 1 residual solution(s), and 1 Step-4 mixture result(s).", restore_message)

    def test_repeated_h5oina_suffix_is_collapsed(self) -> None:
        path = _output_path_with_single_suffix(
            str(self.root / "primary_roi.h5oina.h5oina"),
            ".h5oina",
        )
        self.assertEqual(path.name, "primary_roi.h5oina")
        mixed = _output_path_with_single_suffix(
            str(self.root / "primary_roi.h5oina.ang"),
            ".h5oina",
        )
        self.assertEqual(mixed.name, "primary_roi.h5oina")

    def test_pattern_bearing_h5oina_copy_keeps_every_pattern_without_filters(self) -> None:
        source = self.root / "source.h5oina"
        output = self.root / "residual_export.h5oina"
        pattern_path = "1/EBSD/Data/Processed Patterns"
        patterns = np.arange(6 * 3 * 4, dtype=np.uint8).reshape(6, 3, 4)
        with h5py.File(source, "w") as h5:
            h5.attrs["Manufacturer"] = "Oxford Instruments"
            dataset = h5.create_dataset(
                pattern_path,
                data=patterns,
                chunks=(1, 3, 4),
                compression="lzf",
                shuffle=True,
            )
            dataset.attrs["Meaning"] = "complete scan"
            h5.create_dataset("1/EBSD/Data/Unprocessed Patterns", data=patterns)
            h5.create_dataset("1/EBSD/Data/Phase", data=np.ones(6, dtype=np.int32))

        _copy_h5oina_for_map_export(
            source,
            output,
            included_processed_path=pattern_path,
        )

        with h5py.File(output, "r") as h5:
            copied = h5[pattern_path]
            np.testing.assert_array_equal(copied[()], patterns)
            self.assertEqual(copied.attrs["Meaning"], "complete scan")
            self.assertIsNone(copied.compression)
            self.assertFalse(copied.shuffle)
            self.assertNotIn("1/EBSD/Data/Unprocessed Patterns", h5)

    def test_loading_gui_export_restores_dictionary_indexed_ncc_state(self) -> None:
        source = self.root / "gui_primary_export.h5oina"
        scores = np.array([0.8, 0.7, 0.0, 0.0, 0.6, 0.0], dtype=np.float32)
        roi_mask = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)
        phases = np.array([1, 1, 0, 0, 1, 0], dtype=np.int32)
        with h5py.File(source, "w") as h5:
            h5.create_dataset("1/EBSD/Data/Processed Patterns", data=np.zeros((2, 3, 3, 4), dtype=np.uint8))
            h5.create_dataset("1/Data Processing/Data/Euler", data=np.zeros((6, 3), dtype=np.float64))
            h5.create_dataset("1/Data Processing/Data/Phase", data=phases)
            h5.create_dataset(
                "1/Data Processing/Pattern Matching/Data/Cross Correlation Coefficient",
                data=scores,
            )
            export_group = h5.require_group("1/Data Processing/Overlap EBSD Indexing/Data")
            export_group.attrs["Export Type"] = "primary"
            export_group.create_dataset("ROI Mask", data=roi_mask)
            export_group.create_dataset("Primary NCC", data=scores)

        detector = SimpleNamespace(
            sample_tilt=70.0,
            tilt=0.0,
            azimuthal=0.0,
            twist=0.0,
            px_size=1.0,
            binning=1.0,
            pc_bruker=lambda: np.full((6, 3), 0.5, dtype=np.float64),
            pc_oxford=lambda: np.full((6, 3), 0.6, dtype=np.float64),
        )
        axis = SimpleNamespace(scale=1.0, units="um")
        signal = SimpleNamespace(
            data=np.zeros((2, 3, 3, 4), dtype=np.uint8),
            detector=detector,
            axes_manager=SimpleNamespace(navigation_axes=[axis, axis]),
            xmap=None,
        )
        session = WorkflowSession()
        with patch("kikuchipy.load", return_value=signal):
            message = session.load_input(str(source), None, GeometryConfig())

        np.testing.assert_array_equal(
            session.indexed_mask,
            np.array([True, True, False, False, True, False]),
        )
        np.testing.assert_array_equal(session.last_indexed_indices, np.array([0, 1, 4]))
        np.testing.assert_allclose(session.last_scores_map.reshape(-1)[[0, 1, 4]], [0.8, 0.7, 0.6])
        self.assertTrue(np.all(np.isnan(session.last_scores_map.reshape(-1)[[2, 3, 5]])))
        self.assertIn("Restored 3 dictionary-indexed point(s)", message)

    def test_residual_writer_preserves_points_without_residual_and_commits_atomically(self) -> None:
        source = self.root / "source.h5oina"
        output = self.root / "residuals.h5oina"
        pattern_path = "1/EBSD/Data/Processed Patterns"
        patterns = np.arange(6 * 3 * 4, dtype=np.uint8).reshape(6, 3, 4)
        with h5py.File(source, "w") as h5:
            h5.create_dataset(
                pattern_path,
                data=patterns,
                chunks=(1, 3, 4),
                compression="lzf",
                shuffle=True,
            )
            h5.create_dataset("1/EBSD/Data/Phase", data=np.ones(6, dtype=np.int32))

        source_data = SimpleNamespace(
            source_type="h5oina",
            pattern_path=str(source),
            h5_analysis_root="1",
        )
        writer = ResidualPatternWriter.create(source_data, str(output))
        self.assertFalse(output.exists())
        writer.write(2, np.full((3, 4), 3.0, dtype=np.float32))
        writer.close()

        self.assertTrue(output.exists())
        with h5py.File(output, "r") as h5:
            saved = h5[pattern_path]
            np.testing.assert_array_equal(saved[0], patterns[0])
            np.testing.assert_array_equal(saved[1], patterns[1])
            np.testing.assert_array_equal(saved[2], np.full((3, 4), 255, dtype=np.uint8))
            np.testing.assert_array_equal(saved[3:], patterns[3:])
            self.assertIsNone(saved.compression)
            marker_group = h5["1/Data Processing/Overlap EBSD Indexing/Data"]
            self.assertEqual(
                marker_group.attrs["Residual Pattern Encoding"],
                "overlap-ebsd-residual-patterns-v1",
            )
            np.testing.assert_array_equal(
                marker_group["Residual Pattern Available"][()],
                np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8),
            )

        aborted_output = self.root / "aborted_residuals.h5oina"
        with self.assertRaisesRegex(RuntimeError, "stop writing"):
            with ResidualPatternWriter.create(source_data, str(aborted_output)):
                raise RuntimeError("stop writing")
        self.assertFalse(aborted_output.exists())

    def test_residual_export_recomputes_patterns_from_unverified_old_store(self) -> None:
        source = self.root / "source.h5oina"
        old_store = self.root / "old_residuals.h5oina"
        output = self.root / "exported_residuals.h5oina"
        pattern_path = "1/EBSD/Data/Processed Patterns"
        primary_patterns = np.arange(6 * 3 * 4, dtype=np.uint8).reshape(6, 3, 4)
        with h5py.File(source, "w") as h5:
            h5.create_dataset(pattern_path, data=primary_patterns)
            h5.create_dataset("1/Data Processing/Data/Euler", data=np.zeros((6, 3)))
            h5.create_dataset("1/Data Processing/Data/Phase", data=np.ones(6, dtype=np.int32))
        # This represents an old/incomplete residual file: it contains only
        # copied primary patterns and has no verified residual-store marker.
        _copy_h5oina_for_map_export(source, old_store, included_processed_path=pattern_path)

        session = WorkflowSession()
        session.data = SimpleNamespace(
            pattern_path=str(source),
            source_type="h5oina",
            h5_analysis_root="1",
            rows=2,
            cols=3,
            count=6,
            h=3,
            w=4,
            phase_euler_corrections_rad={},
        )
        session.current_eulers_rad = np.zeros((6, 3), dtype=np.float64)
        session.current_phases = np.ones(6, dtype=np.int32)
        session.last_scores_map = np.ones((2, 3), dtype=np.float32)
        session.residual_eulers_rad = np.zeros((6, 3), dtype=np.float64)
        session.residual_phases = np.ones(6, dtype=np.int32)
        session.last_residual_scores_map = np.ones((2, 3), dtype=np.float32)
        session.residual_pattern_output_path = str(old_store)
        session.residual_point_results[2] = OverlapPointResult(
            index=2,
            row=0,
            col=2,
            ncc_es=0.8,
            scale=0.7,
            ncc_residual_sim=0.1,
            experimental=None,
            simulated=None,
            residual=None,
        )
        materialized = SimpleNamespace(residual=np.full((3, 4), 3.0, dtype=np.float32))
        progress_updates: list[tuple[float, str]] = []
        with patch.object(session, "_materialize_residual_point_result", return_value=materialized) as regenerate:
            session.export_residual_roi_results(
                (0, 0, 2, 3),
                str(output),
                include_residual_patterns=True,
                progress_callback=lambda value, message: progress_updates.append((value, message)),
            )

        regenerate.assert_called_once()
        self.assertEqual(progress_updates[0][0], 0.0)
        self.assertTrue(any("Copying source pattern stack" in message for _, message in progress_updates))
        self.assertTrue(any("Writing residual patterns: 1/1" in message for _, message in progress_updates))
        self.assertEqual(progress_updates[-1][0], 99.0)
        self.assertTrue(all(0.0 <= value <= 100.0 for value, _ in progress_updates))
        with h5py.File(output, "r") as h5:
            exported = h5[pattern_path]
            np.testing.assert_array_equal(exported[2], np.full((3, 4), 255, dtype=np.uint8))
            np.testing.assert_array_equal(exported[0], primary_patterns[0])
            marker_group = h5["1/Data Processing/Overlap EBSD Indexing/Data"]
            self.assertEqual(
                marker_group.attrs["Residual Pattern Encoding"],
                "overlap-ebsd-residual-patterns-v1",
            )

        primary_output = self.root / "exported_primary.h5oina"
        session.indexed_mask = np.ones(6, dtype=bool)
        session.export_primary_roi_results(
            (0, 0, 2, 3),
            str(primary_output),
            include_primary_patterns=True,
        )
        with h5py.File(primary_output, "r") as h5:
            np.testing.assert_array_equal(h5[pattern_path][()], primary_patterns)
            export_group = h5["1/Data Processing/Overlap EBSD Indexing/Data"]
            self.assertEqual(int(export_group.attrs["Primary Patterns Included"]), 1)

    def test_primary_ang_export_can_include_companion_patterns_and_forces_ang_suffix(self) -> None:
        source_up = self.root / "source.up1"
        source_up.write_bytes(b"UP-pattern-payload")
        source_ang = self.root / "source.ang"
        source_ang.write_text(
            "# synthetic ANG\n"
            "0 0 0 0 0 1 0.1 1\n"
            "0 0 0 1 0 1 0.2 1\n",
            encoding="utf-8",
        )
        reader = SimpleNamespace(
            path=str(source_up),
            dtype=np.dtype(np.uint8),
            pattern_offset=0,
            n_patterns=2,
            h=1,
            w=1,
        )
        session = WorkflowSession()
        session.data = SimpleNamespace(
            # The file extension remains authoritative even if stale state
            # reports the wrong source type.
            source_type="h5oina",
            pattern_path=str(source_up),
            orientation_path=str(source_ang),
            rows=1,
            cols=2,
            count=2,
            ang_angles_were_degrees=False,
            up_pattern_reader=reader,
        )
        session.current_eulers_rad = np.zeros((2, 3), dtype=np.float64)
        session.current_phases = np.ones(2, dtype=np.int32)
        session.last_scores_map = np.array([[0.8, 0.7]], dtype=np.float32)
        session.indexed_mask = np.ones(2, dtype=bool)

        requested = self.root / "primary_result.h5oina"
        session.export_primary_roi_results(
            (0, 0, 1, 2),
            str(requested),
            include_primary_patterns=True,
        )

        actual_ang = self.root / "primary_result.ang"
        actual_up = self.root / "primary_result.up1"
        self.assertTrue(actual_ang.is_file())
        self.assertFalse(requested.exists())
        self.assertEqual(actual_up.read_bytes(), source_up.read_bytes())

    def test_parallel_core_count_zero_means_all_available(self) -> None:
        with patch("multistep_overlap_ebsd.core.os.cpu_count", return_value=8):
            self.assertEqual(WorkflowSession._parallel_worker_count(0, 20), 8)
            self.assertEqual(WorkflowSession._parallel_worker_count(4, 20), 4)
            self.assertEqual(WorkflowSession._parallel_worker_count(16, 20), 8)
            self.assertEqual(WorkflowSession._parallel_worker_count(0, 3), 3)
            with self.assertRaises(ValueError):
                WorkflowSession._parallel_worker_count(-1, 20)

    def test_dictionary_batch_size_scales_with_available_memory(self) -> None:
        session = WorkflowSession()
        cache = SimpleNamespace(
            pattern_shape=(64, 78),
            rotation_count=486_755,
        )
        with patch.object(WorkflowSession, "_available_memory_bytes", return_value=8 * 1024**3):
            self.assertEqual(
                session._dictionary_index_batch_size(
                    cache,
                    900,
                    n_per_iteration=26_886,
                ),
                900,
            )
            large_batch = session._dictionary_index_batch_size(
                cache,
                10_000,
                n_per_iteration=26_886,
            )
        with patch.object(WorkflowSession, "_available_memory_bytes", return_value=256 * 1024**2):
            constrained_batch = session._dictionary_index_batch_size(
                cache,
                10_000,
                n_per_iteration=26_886,
            )
        self.assertGreater(large_batch, 512)
        self.assertLessEqual(large_batch, 4096)
        self.assertLess(constrained_batch, large_batch)

    def test_step4_hdf5_export_keeps_full_scan_shape_for_small_roi(self) -> None:
        session = WorkflowSession()
        session.data = SimpleNamespace(
            **vars(self._data()),
            x_coords=np.arange(6, dtype=np.float64),
            y_coords=np.repeat(np.arange(2, dtype=np.float64), 3),
            scan_unit="um",
            step_x=0.5,
            step_y=0.75,
            pc_output_convention="oxford",
        )
        session.master = SimpleNamespace(path="/tmp/master.h5")
        session.current_eulers_rad = np.zeros((6, 3), dtype=np.float64)
        session.residual_eulers_rad = np.full((6, 3), np.nan, dtype=np.float64)
        session.current_phases = np.ones(6, dtype=np.int32)
        session.residual_phases = np.ones(6, dtype=np.int32)
        session.current_pc_custom = np.full((6, 3), 0.5, dtype=np.float64)
        session.last_scores_map = np.full((2, 3), np.nan, dtype=np.float32)
        session.last_residual_scores_map = np.full((2, 3), np.nan, dtype=np.float32)

        def result(index: int, primary_fraction: float) -> OverlapMixtureResult:
            row, col = divmod(index, 3)
            return OverlapMixtureResult(
                index=index,
                row=row,
                col=col,
                primary_fraction=primary_fraction,
                secondary_fraction=1.0 - primary_fraction,
                primary_coefficient=0.7,
                secondary_coefficient=0.3,
                ncc_mixture=0.92,
                residual_rms=0.04,
                old_primary_ncc=0.8,
                old_secondary_ncc=0.75,
                experimental=None,
                primary_simulated=None,
                secondary_simulated=None,
                combined_simulated=None,
                residual=None,
                fitted_sigma=1.2,
                gain_params=(1.0, 2.0, 3.0),
                ellipse_params=(1.0, 1.1, 0.0, 0.1),
                component_correlation=0.2,
                primary_euler_rad=np.array([0.1, 0.2, 0.3]),
                secondary_euler_rad=np.array([0.4, 0.5, 0.6]),
                orientation_refined=True,
                orientation_refinement_note="accepted",
                initial_mixture_ncc=0.88,
                primary_euler_delta_deg=(0.1, 0.0, 0.0),
                secondary_euler_delta_deg=(0.0, 0.2, 0.0),
            )

        # One accumulated point lies outside the currently selected 1x1 ROI;
        # both must remain represented in the exported full-map datasets.
        session.overlap_mixture_results = {0: result(0, 0.6), 4: result(4, 0.7)}
        output = self.root / "step4.h5.h5"
        session.export_overlap_optimization_results(
            str(output),
            (1, 1, 1, 1),
            settings={"parallel_worker_cores": 0},
        )
        actual = self.root / "step4.h5"
        self.assertTrue(actual.exists())
        with h5py.File(actual, "r") as h5:
            self.assertEqual(h5.attrs["format"], "overlap-ebsd-step4-results-v1")
            self.assertEqual(h5["Maps/primary_fraction"].shape, (2, 3))
            self.assertEqual(h5["Maps/primary_euler_rad"].shape, (2, 3, 3))
            self.assertEqual(h5["Maps/gain_params"].shape, (2, 3, 3))
            np.testing.assert_array_equal(
                h5["Maps/computed_mask"][()],
                np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                h5["Scan/requested_roi_mask"][()],
                np.array([[0, 0, 0], [0, 1, 0]], dtype=np.uint8),
            )
            self.assertTrue(np.isnan(h5["Maps/primary_fraction"][0, 1]))
            self.assertEqual(h5["Point Results/index"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
