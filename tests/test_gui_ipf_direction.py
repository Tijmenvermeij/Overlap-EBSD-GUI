from __future__ import annotations

import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from multistep_overlap_ebsd.gui import RESIDUAL_PATTERN_CMAP, MultiStepOverlapGUI


class IpfDirectionSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = tk.Tcl()
        self.variable = tk.StringVar(master=self.interpreter, value="Z")
        self.gui_stub = SimpleNamespace(ipf_direction_var=self.variable)

    def test_x_y_z_are_normalized_for_core_and_labels(self) -> None:
        for value, expected in (
            ("X", ("x", "IPF-X")),
            ("y", ("y", "IPF-Y")),
            (" Z ", ("z", "IPF-Z")),
        ):
            self.variable.set(value)
            self.assertEqual(MultiStepOverlapGUI._selected_ipf_direction(self.gui_stub), expected)

    def test_invalid_direction_falls_back_to_z(self) -> None:
        self.variable.set("invalid")
        self.assertEqual(MultiStepOverlapGUI._selected_ipf_direction(self.gui_stub), ("z", "IPF-Z"))
        self.assertEqual(self.variable.get(), "Z")

    def test_close_waits_for_active_worker(self) -> None:
        gui_stub = SimpleNamespace(
            busy=True,
            _worker_thread=None,
            session=Mock(),
            destroy=Mock(),
        )
        with patch("multistep_overlap_ebsd.gui.messagebox.showinfo") as showinfo:
            MultiStepOverlapGUI._on_close(gui_stub)
        showinfo.assert_called_once()
        gui_stub.session._clear_dictionary_cache.assert_not_called()
        gui_stub.destroy.assert_not_called()

    def test_idle_close_cleans_temporary_dictionary_cache(self) -> None:
        gui_stub = SimpleNamespace(
            busy=False,
            _worker_thread=None,
            session=Mock(),
            destroy=Mock(),
        )
        MultiStepOverlapGUI._on_close(gui_stub)
        gui_stub.session._clear_dictionary_cache.assert_called_once()
        gui_stub.destroy.assert_called_once()

    def test_primary_map_mask_hides_source_orientations(self) -> None:
        gui_stub = SimpleNamespace(
            session=SimpleNamespace(
                data=SimpleNamespace(rows=2, cols=2, count=4),
                indexed_mask=np.array([False, True, False, True]),
                last_scores_map=np.full((2, 2), 0.8, dtype=np.float32),
            ),
            _residual_ncc_threshold=lambda: 0.0,
        )
        mask = MultiStepOverlapGUI._primary_threshold_mask(gui_stub)
        np.testing.assert_array_equal(mask, np.array([[True, False], [True, False]]))

    def test_repeated_source_extensions_are_removed_from_default_stem(self) -> None:
        stem = MultiStepOverlapGUI._source_stem(Path("map.h5oina.h5oina"))
        self.assertEqual(stem, "map")

    def test_residual_patterns_use_grayscale(self) -> None:
        self.assertEqual(RESIDUAL_PATTERN_CMAP, "gray")
        gui_source = Path(__file__).parents[1] / "multistep_overlap_ebsd" / "gui.py"
        source = gui_source.read_text(encoding="utf-8")
        self.assertNotIn('cmap="bwr"', source)

    def test_automated_roi_analysis_runs_only_the_requested_stages(self) -> None:
        def variable(value):
            return SimpleNamespace(get=lambda value=value: value)

        call_order: list[str] = []
        selected_by_stage: dict[str, np.ndarray] = {}

        def operation(name: str):
            def run(*args, **kwargs):
                call_order.append(name)
                indices = kwargs.get("indices", args[0] if args else np.empty(0, dtype=np.int64))
                selected_by_stage[name] = np.asarray(indices, dtype=np.int64).copy()
                callback = kwargs.get("progress_callback")
                if callback is not None:
                    callback(100.0, f"{name} done")
                return f"{name} result"

            return run

        residual_result = SimpleNamespace(index=0)
        mixture_result = SimpleNamespace(index=0)
        mixture_operation = Mock(side_effect=operation("overlap optimization"))
        session = SimpleNamespace(
            data=SimpleNamespace(rows=1, cols=2),
            dictionary_cache=object(),
            roi_indices=lambda *_bounds: np.array([0, 1], dtype=np.int64),
            dictionary_index_indices=operation("primary indexing"),
            refine_orientations_indices=operation("primary refinement"),
            get_primary_index_ncc=lambda index: {0: 0.8, 1: 0.2}[int(index)],
            compute_overlap_residual_indices=operation("residual generation"),
            index_overlap_residual_indices=operation("residual indexing"),
            refine_overlap_residual_indices=operation("residual refinement"),
            compute_overlap_mixture_indices=mixture_operation,
            refine_overlap_mixture_orientations=Mock(),
            get_residual_point_result=lambda _index: residual_result,
            get_overlap_mixture_result=lambda _index: mixture_result,
        )
        complete_progress: list[tuple[float, str]] = []
        refreshed_views: list[int] = []
        outcome: list[str] = []
        gui_stub = SimpleNamespace(
            session=session,
            index_var=variable(0),
            phase_id_var=variable(1),
            di_res_deg_var=variable(1.2),
            dictionary_keep_n_var=variable(2),
            trust_euler_var=variable(1.0),
            maxfev_var=variable(25),
            refine_full_resolution_var=variable(False),
            fit_blur_gain_var=variable(True),
            gain_fit_maxiter_var=variable(40),
            gain_fit_popsize_var=variable(8),
            residual_trust_euler_var=variable(2.0),
            residual_maxfev_var=variable(25),
            residual_refine_full_resolution_var=variable(False),
            step3_parallel_cores_var=variable(0),
            step4_parallel_cores_var=variable(0),
            write_residual_patterns_var=variable(False),
            residual_pattern_path_var=variable("residuals.h5"),
            last_overlap=None,
            last_overlap_mixture=None,
            _roi_bounds=lambda: (0, 0, 1, 2),
            _primary_fit_bounds=lambda: [(0.0, 1.0)],
            _residual_ncc_threshold=lambda: 0.5,
            _overlap_mixture_residual_ncc_threshold=lambda: 0.6,
            _overlap_mixture_residual_ncc_for_index=lambda _index: 0.75,
            _sync_residual_keep_n_to_dictionary=lambda _keep_n: None,
            _set_complete_analysis_progress=lambda value, message: complete_progress.append(
                (float(value), str(message))
            ),
            _set_reindex_progress=lambda _value, _message: None,
            _set_refinement_progress=lambda _value, _message: None,
            _set_overlap_progress=lambda _value, _message: None,
            _set_overlap_optimization_progress=lambda _value, _message: None,
            _refresh_complete_analysis_maps=lambda view_index: refreshed_views.append(
                int(view_index)
            ),
            _log=lambda _message: None,
            after=lambda _delay, callback: callback(),
            _run_threaded=lambda action: outcome.append(action()),
        )

        with patch("multistep_overlap_ebsd.gui.messagebox.showerror") as showerror:
            MultiStepOverlapGUI._run_complete_roi_analysis(gui_stub)

        showerror.assert_not_called()
        session.refine_overlap_mixture_orientations.assert_not_called()
        self.assertEqual(
            call_order,
            [
                "primary indexing",
                "primary refinement",
                "residual generation",
                "residual indexing",
                "residual refinement",
                "overlap optimization",
            ],
        )
        np.testing.assert_array_equal(selected_by_stage["primary indexing"], np.array([0, 1]))
        np.testing.assert_array_equal(selected_by_stage["residual generation"], np.array([0]))
        np.testing.assert_array_equal(selected_by_stage["overlap optimization"], np.array([0]))
        self.assertIs(gui_stub.last_overlap, residual_result)
        self.assertIs(gui_stub.last_overlap_mixture, mixture_result)
        self.assertEqual(refreshed_views, [1, 1, 2, 2, 2, 3])
        self.assertEqual(complete_progress[-1], (100.0, "Complete ROI analysis finished successfully."))
        self.assertIn("Complete ROI analysis finished for 2 point(s)", outcome[0])

        call_order.clear()
        selected_by_stage.clear()
        complete_progress.clear()
        refreshed_views.clear()
        outcome.clear()
        mixture_operation.reset_mock()
        session.refine_overlap_mixture_orientations.reset_mock()
        gui_stub.last_overlap = None
        gui_stub.last_overlap_mixture = mixture_result
        gui_stub.step4_parallel_cores_var = SimpleNamespace(
            get=Mock(side_effect=AssertionError("Steps 2–3 must not read Step 4 settings"))
        )
        gui_stub._overlap_mixture_residual_ncc_threshold = Mock(
            side_effect=AssertionError("Steps 2–3 must not read Step 4 settings")
        )

        with patch("multistep_overlap_ebsd.gui.messagebox.showerror") as showerror:
            MultiStepOverlapGUI._run_complete_roi_analysis(gui_stub, include_step4=False)

        showerror.assert_not_called()
        mixture_operation.assert_not_called()
        session.refine_overlap_mixture_orientations.assert_not_called()
        self.assertEqual(
            call_order,
            [
                "primary indexing",
                "primary refinement",
                "residual generation",
                "residual indexing",
                "residual refinement",
            ],
        )
        np.testing.assert_array_equal(selected_by_stage["primary indexing"], np.array([0, 1]))
        np.testing.assert_array_equal(selected_by_stage["residual generation"], np.array([0]))
        self.assertIs(gui_stub.last_overlap, residual_result)
        self.assertIsNone(gui_stub.last_overlap_mixture)
        self.assertEqual(refreshed_views, [1, 1, 2, 2, 2])
        self.assertEqual(
            complete_progress[-1],
            (100.0, "Steps 2–3 ROI analysis finished successfully."),
        )
        self.assertIn("Steps 2–3 ROI analysis finished for 2 point(s)", outcome[0])
        self.assertIn("Step 4 mixture fitting was not run", outcome[0])


if __name__ == "__main__":
    unittest.main()
