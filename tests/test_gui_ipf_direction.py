from __future__ import annotations

import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.figure import Figure

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

    def test_inspection_marker_is_black_on_ipf_and_red_on_quality_maps(self) -> None:
        figure = Figure()
        ipf_ax, quality_ax = figure.subplots(1, 2)

        ipf_marker = MultiStepOverlapGUI._draw_inspection_marker(
            ipf_ax,
            row=1,
            col=2,
            ipf=True,
        )
        quality_marker = MultiStepOverlapGUI._draw_inspection_marker(
            quality_ax,
            row=1,
            col=2,
            ipf=False,
        )

        np.testing.assert_allclose(ipf_marker.get_edgecolors()[0, :3], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(quality_marker.get_edgecolors()[0, :3], (1.0, 0.0, 0.0))
        self.assertTrue(getattr(ipf_ax, "_overlap_ebsd_scan_map"))
        self.assertTrue(getattr(quality_ax, "_overlap_ebsd_scan_map"))

    def test_clicking_any_tagged_map_moves_the_inspection_point(self) -> None:
        figure = Figure()
        axes = figure.subplots(2, 4)
        residual_score_ax = axes[1, 3]
        pattern_ax = axes[0, 2]
        MultiStepOverlapGUI._draw_inspection_marker(
            residual_score_ax,
            row=0,
            col=0,
            ipf=False,
        )
        row_updates: list[int] = []
        col_updates: list[int] = []
        sync_index = Mock()
        gui_stub = SimpleNamespace(
            session=SimpleNamespace(data=SimpleNamespace(rows=5, cols=6)),
            axes=axes,
            row_var=SimpleNamespace(set=lambda value: row_updates.append(int(value))),
            col_var=SimpleNamespace(set=lambda value: col_updates.append(int(value))),
            _activate_plot_view=lambda _view_index: None,
            _sync_index_from_row_col=sync_index,
        )

        MultiStepOverlapGUI._on_plot_click(
            gui_stub,
            SimpleNamespace(inaxes=residual_score_ax, xdata=3.2, ydata=1.7),
            view_index=2,
        )
        MultiStepOverlapGUI._on_plot_click(
            gui_stub,
            SimpleNamespace(inaxes=pattern_ax, xdata=1.0, ydata=1.0),
            view_index=2,
        )

        self.assertEqual(row_updates, [2])
        self.assertEqual(col_updates, [3])
        sync_index.assert_called_once_with()

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

    def test_roi_export_suffix_follows_pattern_file_even_if_state_is_stale(self) -> None:
        h5_gui = SimpleNamespace(
            session=SimpleNamespace(
                data=SimpleNamespace(pattern_path="/tmp/map.h5oina", source_type="up_ang")
            )
        )
        up_gui = SimpleNamespace(
            session=SimpleNamespace(
                data=SimpleNamespace(pattern_path="/tmp/map.up1", source_type="h5oina")
            )
        )
        up_h5_gui = SimpleNamespace(
            session=up_gui.session,
            roi_export_format_var=SimpleNamespace(get=lambda: "H5OINA"),
        )
        self.assertEqual(MultiStepOverlapGUI._roi_export_suffix(h5_gui), ".h5oina")
        self.assertEqual(MultiStepOverlapGUI._roi_export_suffix(up_gui), ".ang")
        self.assertEqual(MultiStepOverlapGUI._roi_export_suffix(up_h5_gui), ".h5oina")
        self.assertEqual(
            MultiStepOverlapGUI._path_with_single_suffix("/tmp/map.h5oina.ang", ".h5oina").name,
            "map.h5oina",
        )

    def test_primary_and_residual_save_dialogs_do_not_double_h5oina_extension(self) -> None:
        def path_var(default: str):
            value = [default]
            return SimpleNamespace(get=lambda: value[0], set=lambda new: value.__setitem__(0, new))

        gui_stub = SimpleNamespace(
            session=SimpleNamespace(data=SimpleNamespace(source_type="h5oina")),
            _default_roi_export_path=lambda *, residual: (
                "/tmp/map_residual_roi.h5oina" if residual else "/tmp/map_primary_roi.h5oina"
            ),
            _roi_export_suffix=lambda: ".h5oina",
            _path_with_single_suffix=MultiStepOverlapGUI._path_with_single_suffix,
        )
        # A restored workflow may contain stale ANG paths. The loaded H5OINA
        # source must remain authoritative for both save dialogs.
        primary_var = path_var("/tmp/map_primary_roi.ang")
        residual_var = path_var("/tmp/map_residual_roi.ang")

        with patch("multistep_overlap_ebsd.gui.filedialog.asksaveasfilename") as save_dialog:
            save_dialog.side_effect = [
                "/tmp/map_primary_roi.ang",
                "/tmp/map_residual_roi.ang",
            ]
            primary_path = MultiStepOverlapGUI._browse_roi_export(gui_stub, primary_var, residual=False)
            residual_path = MultiStepOverlapGUI._browse_roi_export(gui_stub, residual_var, residual=True)

        self.assertEqual(save_dialog.call_args_list[0].kwargs["initialfile"], "map_primary_roi")
        self.assertEqual(save_dialog.call_args_list[1].kwargs["initialfile"], "map_residual_roi")
        self.assertEqual(save_dialog.call_args_list[0].kwargs["defaultextension"], ".h5oina")
        self.assertEqual(save_dialog.call_args_list[1].kwargs["defaultextension"], ".h5oina")
        self.assertEqual(Path(primary_var.get()).name, "map_primary_roi.h5oina")
        self.assertEqual(Path(residual_var.get()).name, "map_residual_roi.h5oina")
        self.assertEqual(Path(primary_path).name, "map_primary_roi.h5oina")
        self.assertEqual(Path(residual_path).name, "map_residual_roi.h5oina")

    def test_up_ang_save_dialog_accepts_explicit_h5oina_output(self) -> None:
        value = ["/tmp/map_primary_roi.ang"]
        path_var = SimpleNamespace(get=lambda: value[0], set=lambda new: value.__setitem__(0, new))
        gui_stub = SimpleNamespace(
            session=SimpleNamespace(
                data=SimpleNamespace(pattern_path="/tmp/map.up1", source_type="up_ang")
            ),
            _default_roi_export_path=lambda *, residual: "/tmp/map_primary_roi.ang",
            _roi_export_suffix=lambda: ".ang",
            _path_with_single_suffix=MultiStepOverlapGUI._path_with_single_suffix,
        )

        with patch(
            "multistep_overlap_ebsd.gui.filedialog.asksaveasfilename",
            return_value="/tmp/map_primary_roi.h5oina",
        ):
            output = MultiStepOverlapGUI._browse_roi_export(gui_stub, path_var, residual=False)

        self.assertEqual(Path(output).name, "map_primary_roi.h5oina")
        self.assertEqual(Path(path_var.get()).name, "map_primary_roi.h5oina")

    def test_up_ang_path_sync_preserves_selected_h5oina_format(self) -> None:
        def path_var(default: str):
            value = [default]
            return SimpleNamespace(get=lambda: value[0], set=lambda new: value.__setitem__(0, new))

        primary_var = path_var("/tmp/map_primary_roi.h5oina")
        residual_var = path_var("/tmp/map_residual_roi.h5oina")
        gui_stub = SimpleNamespace(
            session=SimpleNamespace(
                data=SimpleNamespace(pattern_path="/tmp/map.up1", source_type="up_ang")
            ),
            primary_roi_export_path_var=primary_var,
            residual_roi_export_path_var=residual_var,
            _roi_export_suffix=lambda: ".ang",
            _default_roi_export_path=lambda *, residual: (
                f"/tmp/map_{'residual' if residual else 'primary'}_roi.ang"
            ),
            _path_with_single_suffix=MultiStepOverlapGUI._path_with_single_suffix,
        )

        MultiStepOverlapGUI._sync_roi_export_paths_to_source(gui_stub)

        self.assertEqual(Path(primary_var.get()).suffix, ".h5oina")
        self.assertEqual(Path(residual_var.get()).suffix, ".h5oina")

    def test_workflow_restore_paths_follow_loaded_source_type(self) -> None:
        def path_var(default: str):
            value = [default]
            return SimpleNamespace(get=lambda: value[0], set=lambda new: value.__setitem__(0, new))

        for source_type, expected_suffix, stale_suffix in (
            ("h5oina", ".h5oina", ".ang"),
            ("up_ang", ".ang", ".h5oina"),
        ):
            primary_var = path_var(f"/tmp/map_primary_roi{stale_suffix}")
            residual_var = path_var(f"/tmp/map_residual_roi{stale_suffix}")
            gui_stub = SimpleNamespace(
                session=SimpleNamespace(data=SimpleNamespace(source_type=source_type)),
                primary_roi_export_path_var=primary_var,
                residual_roi_export_path_var=residual_var,
                _roi_export_suffix=lambda suffix=expected_suffix: suffix,
                _default_roi_export_path=lambda *, residual: (
                    f"/tmp/map_{'residual' if residual else 'primary'}_roi{expected_suffix}"
                ),
                _path_with_single_suffix=MultiStepOverlapGUI._path_with_single_suffix,
            )

            MultiStepOverlapGUI._sync_roi_export_paths_to_source(gui_stub)

            self.assertEqual(Path(primary_var.get()).suffix, expected_suffix)
            self.assertEqual(Path(residual_var.get()).suffix, expected_suffix)

    def test_step4_save_dialog_does_not_double_h5_extension(self) -> None:
        value = ["/tmp/map_step4_results.h5"]
        path_var = SimpleNamespace(get=lambda: value[0], set=lambda new: value.__setitem__(0, new))
        gui_stub = SimpleNamespace(
            overlap_optimization_export_path_var=path_var,
            _default_overlap_optimization_export_path=lambda: "/tmp/map_step4_results.h5",
            _path_with_single_suffix=MultiStepOverlapGUI._path_with_single_suffix,
        )

        with patch(
            "multistep_overlap_ebsd.gui.filedialog.asksaveasfilename",
            return_value="/tmp/map_step4_results.h5.h5",
        ) as save_dialog:
            output_path = MultiStepOverlapGUI._browse_overlap_optimization_export(gui_stub)

        self.assertEqual(save_dialog.call_args.kwargs["initialfile"], "map_step4_results")
        self.assertEqual(save_dialog.call_args.kwargs["defaultextension"], ".h5")
        self.assertEqual(Path(output_path).name, "map_step4_results.h5")
        self.assertEqual(Path(path_var.get()).name, "map_step4_results.h5")

    def test_map_exports_use_one_button_and_prompt_during_export(self) -> None:
        gui_source = Path(__file__).parents[1] / "multistep_overlap_ebsd" / "gui.py"
        source = gui_source.read_text(encoding="utf-8")
        self.assertNotIn('text="Browse", command=self._browse_primary_roi_export', source)
        self.assertNotIn('text="Browse", command=self._browse_residual_roi_export', source)
        self.assertNotIn('text="Browse", command=self._browse_overlap_optimization_export', source)

        session = SimpleNamespace(
            data=SimpleNamespace(source_type="h5oina"),
            export_primary_roi_results=Mock(return_value="exported"),
        )
        picker = Mock(return_value="/tmp/map_primary_roi.h5oina")
        gui_stub = SimpleNamespace(
            session=session,
            include_primary_patterns_export_var=SimpleNamespace(get=lambda: True),
            _roi_bounds=Mock(return_value=(0, 0, 2, 3)),
            _browse_primary_roi_export=picker,
            _set_overlap_progress=Mock(),
            after=lambda _delay, callback: callback(),
            _run_threaded=lambda action: action(),
        )

        MultiStepOverlapGUI._export_primary_roi_map(gui_stub)

        picker.assert_called_once_with()
        call = session.export_primary_roi_results.call_args
        self.assertEqual(call.args, ((0, 0, 2, 3), "/tmp/map_primary_roi.h5oina"))
        self.assertTrue(call.kwargs["include_primary_patterns"])
        self.assertTrue(callable(call.kwargs["progress_callback"]))

    def test_residual_export_forwards_progress_to_overlap_progress_bar(self) -> None:
        def export(*_args, **kwargs):
            kwargs["progress_callback"](52.0, "Writing residual patterns: 100/200...")
            return "exported"

        session = SimpleNamespace(
            data=SimpleNamespace(source_type="h5oina"),
            export_residual_roi_results=Mock(side_effect=export),
        )
        progress = Mock()
        gui_stub = SimpleNamespace(
            session=session,
            include_residual_patterns_export_var=SimpleNamespace(get=lambda: True),
            _roi_bounds=Mock(return_value=(0, 0, 2, 3)),
            _residual_ncc_threshold=Mock(return_value=0.2),
            _browse_residual_roi_export=Mock(return_value="/tmp/map_residual_roi.h5oina"),
            _set_overlap_progress=progress,
            after=lambda _delay, callback: callback(),
            _run_threaded=lambda action: action(),
        )

        MultiStepOverlapGUI._export_residual_roi_map(gui_stub)

        self.assertIn(
            ((52.0, "Writing residual patterns: 100/200..."), {}),
            [(call.args, call.kwargs) for call in progress.call_args_list],
        )
        self.assertEqual(progress.call_args_list[-1].args, (100.0, "Residual ROI export complete."))

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
