from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from multistep_overlap_ebsd.core import (
    GeometryConfig, WorkflowSession, _compute_direction_cosines_kikuchipy,
    _h5_detector_pixel_geometry, _init_residual_roi_worker, _residual_roi_worker_simulated_pattern,
)
from multistep_overlap_ebsd.legacy_projector import XProjector


class PCGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="overlap-pc-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def session(self) -> WorkflowSession:
        up = self.root / "input.up1"
        up.write_bytes(np.asarray([1, 8, 6, 16], dtype=np.uint32).tobytes()
                       + np.arange(12 * 6 * 8, dtype=np.uint8).tobytes())
        ang = self.root / "input.ang"
        header = ["# x-star 0.5", "# y-star 0.4", "# z-star 0.6", "# NROWS 3",
                  "# NCOLS_ODD 4", "# NCOLS_EVEN 4", "# XSTEP 10", "# YSTEP 10"]
        ang.write_text("\n".join(header) + "\n" + "\n".join(
            f"0.1 0.2 0.3 {col * 10} {row * 10} 1 0.8 1"
            for row in range(3) for col in range(4)) + "\n")
        session = WorkflowSession()
        session.load_input(str(up), str(ang), GeometryConfig())
        return session

    def profile(self, mode, shape, acquired=None):
        path = self.root / "metadata.h5"
        with h5py.File(path, "w") as h5:
            if mode is not None:
                h5.create_dataset("1/EBSD/Header/Camera Mode", data=mode)
            if acquired is not None:
                h5.create_dataset("1/EBSD/Header/Acquired Pattern Height", data=acquired[0])
                h5.create_dataset("1/EBSD/Header/Acquired Pattern Width", data=acquired[1])
            return _h5_detector_pixel_geometry(h5, ["1"], shape)

    def test_known_oxford_profiles_have_expected_effective_pitch(self):
        for mode, shape, effective in (
            ("Resolution (1244x1024 px)", (1024, 1244), 20),
            ("Sensitivity (622x512 px)", (512, 622), 40),
            ("Speed 1 (622x512 px)", (512, 622), 40),
            ("Speed 2 (156x128 px)", (128, 156), 160),
            ("Speed 3 (156x88 px)", (88, 156), 160),
        ):
            with self.subTest(mode=mode):
                geometry = self.profile(mode, shape)
                self.assertEqual(geometry["effective_detector_px_size_um"], effective)
                self.assertEqual(geometry["detector_px_size"], 20)
                self.assertIn("assumed 20", geometry["detector_pixel_size_source"])

    def test_stored_reduction_is_separate_from_acquisition_binning(self):
        geometry = self.profile("Sensitivity (622x512 px)", (128, 156), (512, 622))
        self.assertEqual(geometry["effective_detector_px_size_um"], 160)
        self.assertEqual(geometry["detector_binning"], 2)
        self.assertEqual(geometry["detector_px_size"], 20)

    def test_unknown_or_cropped_camera_geometry_stays_unknown(self):
        for mode, shape in ((None, (128, 156)), ("Imported (156x128 px)", (128, 156)),
                            ("Sensitivity (622x512 px)", (256, 250))):
            with self.subTest(mode=mode):
                self.assertIsNone(self.profile(mode, shape)["effective_detector_px_size_um"])

    def test_up_requires_scale_before_scan_correction(self):
        session = self.session()
        self.assertIsNone(session.data.effective_detector_px_size_um)
        self.assertIsNone(session.data.detector_px_size)
        session.calibration_indices = [0, 11]
        for effective in (None, 0, float("nan"), -1):
            with self.subTest(effective=effective), self.assertRaisesRegex(ValueError, "positive detector pixel size"):
                session.apply_average_calibration_pc(effective_detector_px_size_um=effective)

    def test_scan_plane_matches_physical_increments_and_is_idempotent(self):
        session = self.session()
        session.calibration_indices = [0, 11]
        initial_mean = session.current_pc_bruker[[0, 11]].mean(axis=0)
        session.apply_average_calibration_pc(effective_detector_px_size_um=100)
        first = session.current_pc_bruker.copy()
        pc = first.reshape(3, 4, 3)
        alpha = np.deg2rad(20)
        np.testing.assert_allclose(pc[0, 1] - pc[0, 0], [-10 / (100 * 8), 0, 0], atol=1e-14)
        np.testing.assert_allclose(pc[1, 0] - pc[0, 0],
                                   [0, -10 * np.cos(alpha) / (100 * 6), 10 * np.sin(alpha) / (100 * 6)], atol=1e-14)
        np.testing.assert_allclose(first[[0, 11]].mean(axis=0), initial_mean)
        session.apply_average_calibration_pc()
        np.testing.assert_allclose(session.current_pc_bruker, first, atol=1e-14)
        self.assertEqual(session.data.effective_detector_px_size_um, 100)
        self.assertEqual(session.data.detector_pixel_size_source, "user supplied")

    def test_legacy_unbinned_pitch_and_binning_arguments_are_retained(self):
        session = self.session()
        session.calibration_indices = [0, 11]
        session.apply_average_calibration_pc(detector_px_size=50, detector_binning=2)
        self.assertEqual(session.data.detector_px_size, 50)
        self.assertEqual(session.data.detector_binning, 2)
        self.assertEqual(session.data.effective_detector_px_size_um, 100)

    def test_unsupported_scan_axes_are_not_silently_extrapolated(self):
        session = self.session()
        session.calibration_indices = [0]
        session.data.scan_pc_correction_issue = "Nonzero Scan Rotation requires a calibrated PC plane"
        with self.assertRaisesRegex(ValueError, "Scan Rotation"):
            session.apply_average_calibration_pc(effective_detector_px_size_um=100)

    def test_dictionary_uses_live_pc_after_calibration_and_manual_map_apply(self):
        session = self.session()
        session.calibration_indices = [0, 11]
        session.apply_average_calibration_pc(effective_detector_px_size_um=100)
        session.set_point_state(0, pc_custom=(0.65, 0.4, 0.6))
        session.apply_point_pc_to_full_map(0)
        np.testing.assert_allclose(session.calibrated_center_pc_bruker, session.current_pc_bruker[0])
        # Even an old workflow containing a stale compatibility cache must not win.
        session.calibrated_center_pc_bruker[:] = 0.2
        session.master = SimpleNamespace(kind="kikuchipy")
        cache = SimpleNamespace(rotation_count=1, pattern_shape=(6, 8), pattern_dtype="uint8", software_binning=1)
        with patch.object(session, "_get_or_build_kikuchipy_dictionary", return_value=cache) as generate:
            session.generate_dictionary(phase_id=1)
            np.testing.assert_allclose(generate.call_args.kwargs["pc_bruker"], session.current_pc_bruker[0])

    def test_ang_sidecar_round_trip_keeps_spatial_pc_and_pitch(self):
        session = self.session()
        session.calibration_indices = [0, 11]
        session.apply_average_calibration_pc(effective_detector_px_size_um=100)
        output = self.root / "result.ang"
        session.export_reindexed_results(str(output))
        restored = WorkflowSession()
        message = restored.load_input(session.data.pattern_path, str(output), GeometryConfig())
        np.testing.assert_allclose(restored.current_pc_bruker, session.current_pc_bruker)
        np.testing.assert_allclose(restored.current_pc_custom, session.current_pc_custom)
        self.assertEqual(restored.data.effective_detector_px_size_um, 100)
        self.assertIn("Restored per-point PCs", message)

    def test_sidecar_wrong_detector_shape_is_rejected(self):
        session = self.session()
        sidecar = Path(session.data.orientation_path).with_suffix(".pc_map.npz")
        np.savez(sidecar, pc_bruker=session.current_pc_bruker.reshape(3, 4, 3), pattern_shape=[3, 16])
        with self.assertRaisesRegex(ValueError, "different detector pattern shape"):
            WorkflowSession().load_input(session.data.pattern_path, session.data.orientation_path, GeometryConfig())

    def test_up_to_h5_does_not_invent_pitch_and_persists_manual_geometry(self):
        for effective in (None, 100):
            with self.subTest(effective=effective):
                session = self.session()
                if effective:
                    session.calibration_indices = [0, 11]
                    session.apply_average_calibration_pc(effective_detector_px_size_um=effective)
                output = self.root / f"export-{effective}.h5oina"
                session._create_h5oina_from_up_ang(output, include_patterns=True)
                restored = WorkflowSession()
                restored.load_input(str(output), None, GeometryConfig())
                self.assertEqual(restored.data.effective_detector_px_size_um, effective)
                np.testing.assert_allclose(restored.current_pc_bruker, session.current_pc_bruker, atol=1e-6)

    def test_workflow_round_trip_persists_manual_effective_pitch(self):
        session = self.session()
        session.calibration_indices = [0, 11]
        session.apply_average_calibration_pc(effective_detector_px_size_um=100)
        output = self.root / "workflow.npz"
        session.save_workflow_state(str(output))
        restored = WorkflowSession()
        restored.restore_workflow_state(str(output))
        self.assertEqual(restored.data.effective_detector_px_size_um, 100)
        self.assertEqual(restored.data.detector_pixel_size_source, "user supplied")
        np.testing.assert_allclose(restored.current_pc_bruker, session.current_pc_bruker)

    def test_h5_loading_uses_profile_and_detects_rotated_scan(self):
        session = self.session()
        output = self.root / "camera-profile.h5oina"
        session._create_h5oina_from_up_ang(output, include_patterns=True)
        with h5py.File(output, "r+") as h5:
            del h5["1/Data Processing/Overlap EBSD Indexing/Detector Geometry"]
            del h5["1/EBSD/Header/Camera Mode"]
            h5.create_dataset("1/EBSD/Header/Camera Mode", data="Resolution (8x6 px)")
            h5.create_dataset("1/EBSD/Header/Scan Rotation", data=np.pi / 2)
        restored = WorkflowSession()
        restored.load_input(str(output), None, GeometryConfig())
        self.assertEqual(restored.data.effective_detector_px_size_um, 20)
        self.assertEqual(restored.data.detector_binning, 1)
        self.assertIn("Scan Rotation", restored.data.scan_pc_correction_issue)

    def test_legacy_up_projector_matches_original_rays_and_responds_to_pc(self):
        rng = np.random.default_rng(3)
        pair = (rng.random((41, 41)), rng.random((41, 41)))
        euler = np.array([0.1, 0.2, 0.3])
        pc1, pc2 = (0.5, 0.4, 0.6), (0.6, 0.3, 0.7)
        dc, _ = _compute_direction_cosines_kikuchipy(6, 8, pc1, "oxford", 70, 0, 0, 0)
        for projection in ("stereographic", "lambert"):
            with self.subTest(projection=projection):
                hemis = dict(up=pair[0], lo=pair[1], projection=projection)
                old = XProjector(hemis, 6, 8)
                updated = XProjector(hemis, 6, 8, detector_geometry=dict(
                    convention="oxford", sample_tilt=70, tilt=0, azimuthal=0, twist=0))
                original = old.project(euler, pc1, np.eye(3), direction_cosines=dc)
                first = updated.project(euler, pc1, np.eye(3), direction_cosines=dc)
                changed = updated.project(euler, pc2, np.eye(3), direction_cosines=dc)
                np.testing.assert_allclose(first, original, atol=1e-6)
                self.assertGreater(float(np.max(np.abs(first - changed))), 0.01)

    def test_legacy_up_master_loading_enables_current_pc_geometry(self):
        session = self.session()
        master = self.root / "master.legacy"
        master.touch()
        hemis = (np.ones((11, 11)), np.ones((11, 11)))
        with patch("kikuchipy.load", side_effect=ValueError("legacy")), patch(
            "multistep_overlap_ebsd.core.load_master_hemis", return_value=[hemis]
        ):
            session.load_master(str(master))
        self.assertEqual(session.master.projector.detector_geometry["convention"], "oxford")

    def test_legacy_parallel_worker_responds_to_current_pc(self):
        rng = np.random.default_rng(4)
        hemis = (rng.random((41, 41)), rng.random((41, 41)))
        pc1, pc2 = (0.5, 0.4, 0.6), (0.6, 0.3, 0.7)
        dc, _ = _compute_direction_cosines_kikuchipy(6, 8, pc1, "oxford", 70, 0, 0, 0)
        with patch("multistep_overlap_ebsd.core._RESIDUAL_ROI_WORKER_STATE", None), patch(
            "multistep_overlap_ebsd.core.load_master_hemis", return_value=[hemis]
        ):
            _init_residual_roi_worker(
                master_kind="legacy", master_path="unused", master_energy_kv=None,
                master_energy_mode="highest", master_energy_values_kv=None, master_energy_weights=None,
                h=6, w=8, cols=4, rot_sd=np.eye(3), direction_cosines=dc,
                sample_tilt_deg=70, detector_tilt_deg=0, azimuthal_deg=0, twist_deg=0,
                kikuchipy_frame_active=False, weights=np.ones((6, 8)), fit_blur_gain=False,
                fit_maxiter=5, fit_popsize=5, fit_bounds=None)
            first = _residual_roi_worker_simulated_pattern(np.array([0.1, 0.2, 0.3]), np.array(pc1))
            changed = _residual_roi_worker_simulated_pattern(np.array([0.1, 0.2, 0.3]), np.array(pc2))
        self.assertGreater(float(np.max(np.abs(first - changed))), 0.01)


if __name__ == "__main__":
    unittest.main()
