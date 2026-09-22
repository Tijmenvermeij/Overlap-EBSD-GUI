from __future__ import annotations

import threading
import traceback
import warnings
import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .gui_controls import GUIControls
from .cpu_fitting import FIT_METHOD_DEFAULT, FIT_METHOD_LABELS, validate_fit_method
from .live_updates import LiveUpdateGate
from .workflow_autosave import save_checkpoint
from .version import __version__

from .core import (
    MASTER_ENERGY_MODE_GLOBAL,
    MASTER_ENERGY_MODE_HIGHEST,
    ORIENTATION_LAYER_LABEL,
    ORIENTATION_LAYER_LABELS,
    GeometryConfig,
    OverlapMixtureResult,
    OverlapPointResult,
    WorkflowSession,
)

PLOT_TITLE_FONTSIZE = 9
PLOT_TEXT_FONTSIZE = 8
PLOT_TICK_FONTSIZE = 7
PLOT_INSTRUCTION_FONTSIZE = 10
RESIDUAL_PATTERN_CMAP = "gray"
_TIGHT_LAYOUT_WARNING = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect."
MASTER_ENERGY_MODE_LABELS = {
    MASTER_ENERGY_MODE_HIGHEST: "Highest available energy",
    MASTER_ENERGY_MODE_GLOBAL: "Global MC-weighted (EMsoft)",
}


def _selected_fit_method(gui) -> str:
    """Read the shared fit choice before launching a worker."""
    variable = getattr(gui, "fit_method_var", None)
    value = variable.get() if variable is not None else FIT_METHOD_DEFAULT
    for method, label in FIT_METHOD_LABELS.items():
        if value == label:
            return method
    return validate_fit_method(value)


class MultiStepOverlapGUI(GUIControls, tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Overlap EBSD/TKD Indexing — v{__version__}")
        width = min(1480, max(1000, self.winfo_screenwidth() - 40))
        height = min(920, max(680, self.winfo_screenheight() - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(1000, 680)

        self.session = WorkflowSession()
        self.last_overlap: OverlapPointResult | None = None
        self.last_overlap_mixture: OverlapMixtureResult | None = None
        self.busy = False
        self._worker_thread: threading.Thread | None = None
        self._suspend_point_trace = False
        self._live_refresh_after_id: str | None = None
        self._job_update_gate = None
        self._job_result_views = set()
        self.live_update_status_var = tk.StringVar(value="Speed priority: maps refresh between batches, less often when drawing is expensive.")
        self._residual_colorbar = None
        self._euler_step_deg = 0.01
        self._pc_step = 0.001
        self._left_canvas: tk.Canvas | None = None
        self._left_controls_frame: ttk.Frame | None = None
        self._left_canvas_window_id: int | None = None
        self.info_text: tk.Text | None = None
        self.info_texts: list[tk.Text] = []
        self.log_texts: list[tk.Text] = []
        self._plot_views: dict[int, dict[str, object]] = {}
        self.overlap_inspection_window: tk.Toplevel | None = None
        self.overlap_inspection_figure: Figure | None = None
        self.overlap_inspection_axes = None
        self.overlap_inspection_canvas: FigureCanvasTkAgg | None = None
        self._pending_overlap_inspection: OverlapPointResult | None = None
        self.btn_refine_roi: ttk.Button | None = None
        self.btn_index_roi: ttk.Button | None = None
        self.btn_refine_indexed: ttk.Button | None = None
        self.btn_steps_2_3_analysis: ttk.Button | None = None
        self.btn_complete_analysis: ttk.Button | None = None
        self.workflow_notebook: ttk.Notebook | None = None
        self._pending_restore_path: str | None = None

        cwd = Path.cwd()
        self.pattern_path_var = tk.StringVar(value="")
        self.orientation_path_var = tk.StringVar(value="")
        self.master_path_var = tk.StringVar(value="")
        self.source_type_var = tk.StringVar(value="H5OINA")
        self.pattern_input_label_var = tk.StringVar(value="Patterns + orientations (.h5oina)")
        self.detector_pixel_source_var = tk.StringVar(value="Load data to read detector calibration.")
        self.context_summary_var = tk.StringVar(value="No data loaded · No master pattern · No dictionary")
        self.phase_summary_var = tk.StringVar(value="Indexing phase comes from the loaded master pattern.")
        self.master_energy_mode_var = tk.StringVar(
            value=MASTER_ENERGY_MODE_LABELS[MASTER_ENERGY_MODE_HIGHEST]
        )
        self.export_path_var = tk.StringVar(value=str((cwd / "reindexed_output.h5oina").resolve()))
        initial_workflow_path = str((cwd / "overlap_workflow.npz").resolve())
        self.workflow_path_var = tk.StringVar(value=initial_workflow_path)
        self._auto_workflow_path: str | None = initial_workflow_path

        self.sample_tilt_var = tk.DoubleVar(value=70.0)
        self.detector_tilt_var = tk.DoubleVar(value=0.0)

        self.phase_id_var = tk.IntVar(value=1)
        self.index_var = tk.IntVar(value=0)
        self.row_var = tk.IntVar(value=0)
        self.col_var = tk.IntVar(value=0)
        self.roi_r0_var = tk.IntVar(value=0)
        self.roi_c0_var = tk.IntVar(value=0)
        self.roi_nrows_var = tk.IntVar(value=0)
        self.roi_ncols_var = tk.IntVar(value=0)
        self.roi_r1_var = self.roi_nrows_var
        self.roi_c1_var = self.roi_ncols_var
        self.euler1_deg_var = tk.DoubleVar(value=0.0)
        self.euler2_deg_var = tk.DoubleVar(value=0.0)
        self.euler3_deg_var = tk.DoubleVar(value=0.0)
        self.pcx_var = tk.DoubleVar(value=0.0)
        self.pcy_var = tk.DoubleVar(value=0.0)
        self.pcz_var = tk.DoubleVar(value=0.0)
        self.pc_conv_label_var = tk.StringVar(value="PC convention: -")

        self.trust_euler_var = tk.DoubleVar(value=1.5)
        self.trust_pc_var = tk.DoubleVar(value=0.03)
        self.maxfev_var = tk.IntVar(value=25)
        self.refine_full_resolution_var = tk.BooleanVar(value=True)
        self.auto_refine_var = tk.BooleanVar(value=True)
        self.follow_dictionary_trust_var = tk.BooleanVar(value=True)
        self.calibration_trust_euler_var = tk.DoubleVar(value=1.0)
        self.calibration_maxfev_var = tk.IntVar(value=25)

        self.di_res_deg_var = tk.DoubleVar(value=1.5)
        self.di_binning_var = tk.StringVar(value="2")
        self.dictionary_binned_size_var = tk.StringVar(value="Load input data to see the binned pattern size.")
        self.dictionary_keep_n_var = tk.IntVar(value=5)
        self.dictionary_status_var = tk.StringVar(value="No dictionary generated or loaded.")
        self.dictionary_progress_var = tk.DoubleVar(value=0.0)
        self.reindex_progress_var = tk.DoubleVar(value=0.0)
        self.reindex_progress_status_var = tk.StringVar(value="Re-indexing not started.")
        self.complete_analysis_progress_var = tk.DoubleVar(value=0.0)
        self.complete_analysis_status_var = tk.StringVar(value="Complete ROI analysis not started.")
        self.overlap_progress_var = tk.DoubleVar(value=0.0)
        self.overlap_progress_status_var = tk.StringVar(value="Residual ROI workflow not started.")
        self.overlap_optimization_progress_var = tk.DoubleVar(value=0.0)
        self.overlap_optimization_status_var = tk.StringVar(value="Overlap optimization not started.")
        self.refinement_progress_var = tk.DoubleVar(value=0.0)
        self.refinement_progress_status_var = tk.StringVar(value="Orientation refinement not started.")
        initial_dictionary_path = str((cwd / "dictionary_bin2_1p5deg.h5").resolve())
        self._auto_dictionary_path: str | None = initial_dictionary_path
        self.dictionary_path_var = tk.StringVar(value=initial_dictionary_path)
        self.primary_roi_export_path_var = tk.StringVar(value=str((cwd / "primary_roi_map.h5oina").resolve()))
        self.residual_roi_export_path_var = tk.StringVar(value=str((cwd / "residual_roi_map.h5oina").resolve()))
        self.blur_sigma_var = tk.DoubleVar(value=0.0)
        self.fit_blur_gain_var = tk.BooleanVar(value=True)
        self.fit_method_var = tk.StringVar(value=FIT_METHOD_LABELS[FIT_METHOD_DEFAULT])
        self.gain_fit_maxiter_var = tk.IntVar(value=80)
        self.gain_fit_popsize_var = tk.IntVar(value=15)
        self.residual_trust_euler_var = self.trust_euler_var
        self.residual_maxfev_var = self.maxfev_var
        self.residual_refine_full_resolution_var = self.refine_full_resolution_var
        self.residual_keep_n_var = self.dictionary_keep_n_var
        self.parallel_cores_var = tk.IntVar(value=min(6, os.cpu_count() or 1))
        self.step3_parallel_cores_var = self.parallel_cores_var
        self.overlap_mixture_trust_euler_var = tk.DoubleVar(value=1.0)
        self.overlap_mixture_maxfev_var = tk.IntVar(value=80)
        self.step4_parallel_cores_var = self.parallel_cores_var
        self.overlap_min_ncc_var = tk.StringVar(value="0.15")
        self.residual_ipf_ncc_var = tk.StringVar(value="0.15")
        self.overlap_mixture_residual_ncc_var = tk.StringVar(value=self.residual_ipf_ncc_var.get())
        self.write_residual_patterns_var = tk.BooleanVar(value=False)
        self.include_primary_patterns_export_var = tk.BooleanVar(value=False)
        self.include_residual_patterns_export_var = tk.BooleanVar(value=False)
        self.roi_export_format_var = tk.StringVar(value="H5OINA")
        self.residual_pattern_path_var = tk.StringVar(value=str((cwd / "residual_patterns.h5oina").resolve()))
        self.overlap_optimization_export_path_var = tk.StringVar(
            value=str((cwd / "overlap_optimization_results.h5").resolve())
        )
        self.primary_fit_bound_specs = [
            ("Gaussian sigma", tk.DoubleVar(value=0.1), tk.DoubleVar(value=5.0)),
            ("Gain min", tk.DoubleVar(value=-1.5), tk.DoubleVar(value=4.5)),
            ("Gain max", tk.DoubleVar(value=0.0), tk.DoubleVar(value=12.5)),
            ("Gain power", tk.DoubleVar(value=0.1), tk.DoubleVar(value=10.0)),
            ("Ellipse a scale", tk.DoubleVar(value=0.6), tk.DoubleVar(value=1.4)),
            ("Ellipse b scale", tk.DoubleVar(value=0.6), tk.DoubleVar(value=1.4)),
            ("Ellipse y offset", tk.DoubleVar(value=-0.15), tk.DoubleVar(value=0.15)),
            ("Ellipse x offset", tk.DoubleVar(value=-0.15), tk.DoubleVar(value=0.15)),
        ]
        self.use_scan_pc_shift_var = tk.BooleanVar(value=False)
        self.detector_px_size_var = tk.StringVar(value="")
        self.detector_binning_var = tk.DoubleVar(value=1.0)
        self.calibration_summary_var = tk.StringVar(value="No calibration points selected.")
        self.calibration_statistics_var = tk.StringVar(value="")
        self._completed_calibration_report = None
        self.calibration_apply_status_var = tk.StringVar(value="Optimize first, then apply the average PC to the map.")
        self._calibration_apply_state = "none"
        self._calibration_apply_session = self.session
        self._applied_calibration_settings = None
        self._roi_drag_view_index: int | None = None
        self._roi_drag_axis = None
        self._roi_drag_start: tuple[float, float] | None = None
        self._roi_drag_last: tuple[float, float] | None = None
        self._roi_drag_patch: Rectangle | None = None
        self.pattern_mask_option_var = tk.IntVar(value=-1)
        self.pattern_mask_mode_var = tk.StringVar(value="Automatic")
        self.pattern_mask_status_var = tk.StringVar(value=f"Mask: {self.session.pattern_mask_description()}")
        self.dynamic_bg_enabled_var = tk.BooleanVar(value=False)
        self.dynamic_bg_std_var = tk.StringVar(value="0")
        self.dynamic_bg_status_var = tk.StringVar(value=f"Dynamic BG: {self.session.dynamic_background_description()}")

        self.map_layer_var = tk.StringVar(value=ORIENTATION_LAYER_LABEL)
        self.ipf_direction_var = tk.StringVar(value="Z")
        self.solution_map_var = tk.StringVar(value="IPF maps")
        self.index_quality_layer_var = tk.StringVar(value="CI")
        self.status_var = tk.StringVar(value="Load data to begin.")
        self.refinement_progress_bar: ttk.Progressbar | None = None
        self.overlap_progress_bar: ttk.Progressbar | None = None
        self._progress_pulse_after_ids: dict[str, str] = {}
        self._progress_pulse_active: set[str] = set()

        self._build_ui()
        self.di_res_deg_var.trace_add("write", self._sync_refinement_settings)
        self.di_binning_var.trace_add("write", self._update_dictionary_binned_size)
        self.follow_dictionary_trust_var.trace_add("write", self._sync_refinement_settings)
        self.pattern_mask_option_var.trace_add("write", self._sync_mask_mode)
        self.use_scan_pc_shift_var.trace_add("write", self._update_calibration_application_controls)
        self.detector_px_size_var.trace_add("write", self._update_calibration_application_controls)
        self._sync_refinement_settings()
        self._update_dictionary_binned_size()
        self._sync_input_type_controls()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._attach_point_value_traces()
        self._attach_threshold_value_traces()
        self._attach_entry_commit_handlers()
        self._update_mode_controls()

    # ---------------------------- UI ---------------------------- #


    def _build_plot_area(
        self,
        parent: ttk.Frame,
        view_index: int,
        *,
        single_map: bool = False,
        fixed_ipf: bool = False,
    ) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ipf_combo = None
        if view_index == 1:
            ttk.Label(top, text="IPF direction").pack(side=tk.LEFT)
            ipf_combo = ttk.Combobox(
                top,
                textvariable=self.ipf_direction_var,
                values=("X", "Y", "Z"),
                state="readonly",
                width=4,
            )
            ipf_combo.pack(side=tk.LEFT, padx=4)
            ipf_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
            combo = ipf_combo
        elif fixed_ipf:
            ttk.Label(top, text="Primary and residual maps").pack(side=tk.LEFT)
            ttk.Label(top, text="IPF direction").pack(side=tk.LEFT, padx=(14, 0))
            ipf_combo = ttk.Combobox(
                top,
                textvariable=self.ipf_direction_var,
                values=("X", "Y", "Z"),
                state="readonly",
                width=4,
            )
            ipf_combo.pack(side=tk.LEFT, padx=4)
            ipf_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
            combo = ipf_combo
        elif single_map:
            ttk.Label(top, text="Re-indexed orientation map").pack(side=tk.LEFT)
            combo = ttk.Combobox(top, textvariable=self.map_layer_var, values=[ORIENTATION_LAYER_LABEL], state="disabled", width=1)
        else:
            ttk.Label(top, text="Map layer").pack(side=tk.LEFT)
            combo = ttk.Combobox(
                top,
                textvariable=self.map_layer_var,
                values=[*ORIENTATION_LAYER_LABELS, "Phase"],
                state="readonly",
                width=20,
            )
            combo.pack(side=tk.LEFT, padx=4)
            combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
        if view_index in (1, 2, 3):
            map_kind = ttk.Combobox(top, textvariable=self.solution_map_var,
                                   values=("IPF maps", "Phase maps"), state="readonly", width=12)
            map_kind.pack(side=tk.LEFT, padx=4)
            map_kind.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
        ttk.Button(top, text="Refresh", command=self._refresh_plot).pack(side=tk.LEFT, padx=4)
        if fixed_ipf:
            figure = Figure(figsize=(13.6, 8.8), dpi=100)
            axes = figure.subplots(2, 4, gridspec_kw={"wspace": 0.06, "hspace": 0.18})
        elif view_index == 1:
            figure = Figure(figsize=(14.0, 8.9), dpi=100)
            axes = figure.subplots(2, 3, gridspec_kw={"wspace": 0.06, "hspace": 0.18})
        else:
            figure = Figure(figsize=(11.5, 8.5), dpi=100)
            axes = np.asarray([[figure.subplots()]]) if single_map else figure.subplots(2, 3)
        for axis in axes.flat:
            axis.set_axis_off()
        self._safe_tight_layout(figure)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        phase_legend = None
        if view_index in (1, 2, 3):
            phase_legend = tk.Canvas(top, height=25, width=1, highlightthickness=0,
                                     background=ttk.Style().lookup("TFrame", "background") or "#ffffff")
            phase_legend.bind("<MouseWheel>", lambda event, legend=phase_legend:
                              legend.xview_scroll(-1 if event.delta > 0 else 1, "units"))
        canvas.mpl_connect("button_press_event", lambda event, i=view_index: self._on_plot_click(event, i))
        canvas.mpl_connect("motion_notify_event", lambda event, i=view_index: self._on_plot_motion(event, i))
        canvas.mpl_connect("button_release_event", lambda event, i=view_index: self._on_plot_release(event, i))
        self._plot_views[view_index] = {
            "figure": figure,
            "axes": axes,
            "canvas": canvas,
            "combo": combo,
            "ipf_combo": ipf_combo,
            "colorbar": None,
            "toolbar": toolbar,
            "phase_legend": phase_legend,
        }
        if view_index == 0:
            self.map_layer_combo = combo

    def _activate_plot_view(self, index: int) -> None:
        view = self._plot_views[int(index)]
        self.figure = view["figure"]
        self.axes = view["axes"]
        self.canvas = view["canvas"]
        self.map_layer_combo = view["combo"]

    def _on_workspace_changed(self, _event=None) -> None:
        if self.workflow_notebook is None:
            return
        index = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        self._activate_plot_view(index)
        self._update_calibration_application_controls()
        if not self.busy:
            self._refresh_plot()


    # ------------------------ UI handlers ------------------------ #

    def _on_close(self) -> None:
        worker = self._worker_thread
        if self.busy or (worker is not None and worker.is_alive()):
            messagebox.showinfo(
                "Operation in progress",
                "Wait for the current operation to finish before closing the application.",
            )
            return
        # An active worker can still use the dictionary and lossless residual
        # cache. Release both only after the existing busy/worker guard passes.
        self.session.close()
        self.destroy()

    def _set_busy(self, flag: bool) -> None:
        self.busy = bool(flag)
        if self.busy:
            self._busy_widget_states = []
            def disable_controls(parent):
                for widget in parent.winfo_children():
                    if widget.winfo_class() in {
                        "TButton", "TCheckbutton", "TRadiobutton", "TEntry", "TCombobox",
                        "TSpinbox", "TScale", "Button", "Checkbutton", "Entry", "Spinbox", "Scale",
                    }:
                        try:
                            state = str(widget.cget("state"))
                            self._busy_widget_states.append((widget, state))
                            widget.configure(state="disabled")
                        except tk.TclError:
                            pass
                    disable_controls(widget)
            disable_controls(self)
            cancel = getattr(self, "btn_cancel", None)
            if cancel is not None:
                cancel.configure(state=tk.NORMAL)
        else:
            for widget, state in getattr(self, "_busy_widget_states", []):
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass
            self._busy_widget_states = []
            for kind in ("refinement", "overlap"):
                self._stop_progress_pulse(kind)
            self._update_mode_controls()

    def _cancel_current_action(self) -> None:
        if self.busy:
            self._cancel_requested.set()
            self.status_var.set("Stopping at the next safe boundary...")
            cancel = getattr(self, "btn_cancel", None)
            if cancel is not None:
                cancel.configure(state=tk.DISABLED)

    def _check_job_cancelled(self) -> None:
        requested = getattr(self, "_cancel_requested", None)
        if requested is not None and requested.is_set():
            raise InterruptedError("Operation cancelled at a safe boundary; completed results remain available.")

    def _guarded_action(fn):
        """Reject concurrent actions and report invalid entries before launching a job."""
        needs_current_results = fn.__name__ in {
            "_refine_last_indexed", "_run_complete_roi_analysis", "_run_residual_roi_analysis",
            "_index_overlap_residual", "_refine_overlap_residual", "_compute_overlap_residual_roi",
            "_index_overlap_residual_roi", "_refine_overlap_residual_roi",
            "_fit_overlap_mixture", "_refine_overlap_mixture_orientations", "_fit_overlap_mixture_roi",
            "_export_fitted_primary_patterns", "_export_fitted_residual_patterns",
        }
        def guarded(self, *args, **kwargs):
            if getattr(self, "busy", False):
                return None
            try:
                if needs_current_results:
                    self._sync_pattern_conditioning_settings()
                return fn(self, *args, **kwargs)
            except (ValueError, TypeError, RuntimeError, tk.TclError) as exc:
                messagebox.showerror("Invalid settings", str(exc))
                return None
        guarded.__name__ = fn.__name__
        guarded.__doc__ = fn.__doc__
        return guarded

    def _post_ui(self, callback) -> None:
        """Enqueue callbacks without calling Tk from numerical worker threads."""
        self._ui_callback_queue.put(callback)
        gate = getattr(self, "_job_update_gate", None)
        if gate is not None and threading.current_thread() is self._worker_thread:
            gate.at_boundary(self._ui_callback_queue.put, lambda: self._refresh_job_maps())

    def _refresh_job_maps(self) -> None:
        # The coordinator waits at a progress boundary for this callback.
        # No numerical worker writes session maps while they are being drawn.
        if not self.busy or self.session.data is None or not self._job_result_views:
            return
        active = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        views = tuple(sorted(self._job_result_views))
        self._job_result_views.clear()
        try:
            self._populate_point_vars()
            if 0 in views:
                self.calibration_summary_var.set(self.session.calibration_point_summary(include_statistics=False))
            # Hidden tabs will refresh when opened. Draw the visible view once
            # even if several stages have produced results since the last draw.
            self._refresh_complete_analysis_maps(active)
            from time import strftime
            self.live_update_status_var.set(
                f"Live maps updated {strftime('%H:%M:%S')} — speed priority; completed batches."
            )
        except Exception:
            self.live_update_status_var.set("Live map refresh failed; processing continues.")
            self._log(f"Live map refresh failed:\n{traceback.format_exc()}")
        finally:
            self._activate_plot_view(active)

    def _drain_ui_callbacks(self) -> None:
        import queue
        try:
            for _ in range(100):
                callback = self._ui_callback_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        if self.busy:
            self.after(25, self._drain_ui_callbacks)

    def _progress_bar_for_kind(self, kind: str) -> ttk.Progressbar | None:
        if kind == "refinement":
            return self.refinement_progress_bar
        if kind == "overlap":
            return self.overlap_progress_bar
        return None

    def _cancel_progress_pulse_timer(self, kind: str) -> None:
        after_id = self._progress_pulse_after_ids.pop(kind, None)
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass

    def _start_progress_pulse(self, kind: str) -> None:
        self._progress_pulse_after_ids.pop(kind, None)
        if not self.busy:
            return
        bar = self._progress_bar_for_kind(kind)
        if bar is None or kind in self._progress_pulse_active:
            return
        try:
            bar.configure(mode="indeterminate")
            bar.start(12)
            self._progress_pulse_active.add(kind)
        except Exception:
            pass

    def _stop_progress_pulse(self, kind: str) -> None:
        self._cancel_progress_pulse_timer(kind)
        bar = self._progress_bar_for_kind(kind)
        if kind in self._progress_pulse_active and bar is not None:
            try:
                bar.stop()
            except Exception:
                pass
        self._progress_pulse_active.discard(kind)
        if bar is not None:
            try:
                bar.configure(mode="determinate")
            except Exception:
                pass

    def _schedule_progress_pulse(self, kind: str, *, delay_ms: int = 600) -> None:
        self._cancel_progress_pulse_timer(kind)
        bar = self._progress_bar_for_kind(kind)
        if bar is None:
            return
        self._progress_pulse_after_ids[kind] = self.after(
            int(delay_ms),
            lambda kind=kind: self._start_progress_pulse(kind),
        )

    def _set_progress_state(
        self,
        kind: str,
        variable: tk.DoubleVar,
        status_var: tk.StringVar,
        value: float,
        message: str,
    ) -> None:
        clipped = float(np.clip(value, 0.0, 100.0))
        self._stop_progress_pulse(kind)
        variable.set(clipped)
        status_var.set(message)
        # Keep refinement / overlap progress bars determinate so the displayed
        # value tracks the reported percentage instead of switching to a pulse.
        self.update_idletasks()

    def _attach_point_value_traces(self) -> None:
        vars_to_watch = (
            self.euler1_deg_var,
            self.euler2_deg_var,
            self.euler3_deg_var,
            self.pcx_var,
            self.pcy_var,
            self.pcz_var,
        )
        for var in vars_to_watch:
            var.trace_add("write", self._on_point_values_changed)
        for var in (
            self.roi_r0_var,
            self.roi_c0_var,
            self.roi_nrows_var,
            self.roi_ncols_var,
        ):
            var.trace_add("write", self._on_roi_values_changed)

    def _on_roi_values_changed(self, *_args) -> None:
        if self._suspend_point_trace:
            return
        self._refresh_context_summary()
        self._schedule_live_refresh(delay_ms=250)

    def _on_point_values_changed(self, *_args) -> None:
        self._schedule_live_refresh(delay_ms=250)

    def _attach_threshold_value_traces(self) -> None:
        vars_to_watch = (
            self.overlap_min_ncc_var,
            self.residual_ipf_ncc_var,
            self.overlap_mixture_residual_ncc_var,
        )
        for var in vars_to_watch:
            var.trace_add("write", self._on_threshold_values_changed)

    def _on_threshold_values_changed(self, *_args) -> None:
        self._schedule_live_refresh(delay_ms=250)

    def _schedule_live_refresh(self, *, delay_ms: int) -> None:
        if self._suspend_point_trace:
            return
        if self.busy or self.session.data is None:
            return
        if self._live_refresh_after_id is not None:
            try:
                self.after_cancel(self._live_refresh_after_id)
            except Exception:
                pass
        self._live_refresh_after_id = self.after(max(0, int(delay_ms)), self._run_live_refresh)

    def _attach_entry_commit_handlers(self, widget: tk.Misc | None = None) -> None:
        root = self if widget is None else widget
        for child in root.winfo_children():
            try:
                if child.winfo_class() in {"Entry", "TEntry", "Spinbox", "TSpinbox"}:
                    child.bind("<Return>", self._on_value_commit, add="+")
                    child.bind("<KP_Enter>", self._on_value_commit, add="+")
            except Exception:
                pass
            self._attach_entry_commit_handlers(child)

    def _on_value_commit(self, event: tk.Event | None = None) -> str | None:
        self._schedule_live_refresh(delay_ms=0)
        if event is not None and str(getattr(event, "keysym", "")) in {"Return", "KP_Enter"}:
            return "break"
        return None

    def _run_live_refresh(self) -> None:
        self._live_refresh_after_id = None
        if self.busy or self.session.data is None:
            return
        try:
            self._refresh_plot()
        except Exception:
            # Keep GUI responsive during intermediate invalid edits.
            return

    def _event_has_shift(self, event) -> bool:
        key = str(getattr(event, "key", "")).lower()
        return "shift" in key

    def _step2_roi_axes(self) -> tuple[object, ...]:
        axes = np.asarray(self.axes, dtype=object)
        if axes.shape == (2, 3):
            return tuple(axes.flat)
        return ()

    def _cancel_roi_drag(self) -> None:
        if self._roi_drag_patch is not None:
            try:
                self._roi_drag_patch.remove()
            except Exception:
                pass
        self._roi_drag_patch = None
        self._roi_drag_axis = None
        self._roi_drag_start = None
        self._roi_drag_last = None
        self._roi_drag_view_index = None

    @_guarded_action
    def _set_roi_bounds(self, r0: int, c0: int, nrows: int, ncols: int, *, source: str) -> str:
        if self.session.data is None:
            raise RuntimeError("Load input data first.")
        rows = int(self.session.data.rows)
        cols = int(self.session.data.cols)
        r0 = max(0, min(int(r0), rows - 1))
        c0 = max(0, min(int(c0), cols - 1))
        nrows = max(1, min(int(nrows), rows - r0))
        ncols = max(1, min(int(ncols), cols - c0))
        self._suspend_point_trace = True
        try:
            self.roi_r0_var.set(r0)
            self.roi_c0_var.set(c0)
            self.roi_nrows_var.set(nrows)
            self.roi_ncols_var.set(ncols)
        finally:
            self._suspend_point_trace = False
        msg = f"ROI set to r0={r0}, c0={c0}, nrows={nrows}, ncols={ncols} ({source})."
        self.status_var.set(msg)
        self._log(msg)
        self._refresh_plot()
        self._refresh_context_summary()
        return msg

    def _maybe_begin_roi_drag(self, event, view_index: int | None) -> bool:
        if self.session.data is None or view_index != 1 or event.button != 1:
            return False
        if not self._event_has_shift(event) or event.xdata is None or event.ydata is None:
            return False
        roi_axes = self._step2_roi_axes()
        if not roi_axes or event.inaxes not in roi_axes:
            return False
        self._cancel_roi_drag()
        self._roi_drag_view_index = 1
        self._roi_drag_axis = event.inaxes
        self._roi_drag_start = (float(event.xdata), float(event.ydata))
        self._roi_drag_last = self._roi_drag_start
        rect = Rectangle(
            (float(event.xdata), float(event.ydata)),
            0.0,
            0.0,
            fill=False,
            edgecolor="magenta",
            linewidth=1.5,
            linestyle="--",
        )
        event.inaxes.add_patch(rect)
        self._roi_drag_patch = rect
        self.canvas.draw_idle()
        return True

    def _update_roi_drag(self, event) -> None:
        if self._roi_drag_patch is None or self._roi_drag_axis is None or self._roi_drag_start is None:
            return
        if event.inaxes is not self._roi_drag_axis or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._roi_drag_start
        x1, y1 = float(event.xdata), float(event.ydata)
        self._roi_drag_last = (x1, y1)
        left = min(x0, x1)
        bottom = min(y0, y1)
        width = max(abs(x1 - x0), 1e-9)
        height = max(abs(y1 - y0), 1e-9)
        self._roi_drag_patch.set_x(left)
        self._roi_drag_patch.set_y(bottom)
        self._roi_drag_patch.set_width(width)
        self._roi_drag_patch.set_height(height)
        self.canvas.draw_idle()

    def _finish_roi_drag(self, event) -> bool:
        if self._roi_drag_view_index != 1 or self._roi_drag_start is None:
            return False
        end = self._roi_drag_last if self._roi_drag_last is not None else self._roi_drag_start
        if event.xdata is not None and event.ydata is not None and (
            self._roi_drag_axis is None or event.inaxes is self._roi_drag_axis
        ):
            end = (float(event.xdata), float(event.ydata))
        x0, y0 = self._roi_drag_start
        x1, y1 = end
        self._cancel_roi_drag()
        c0 = int(np.clip(round(min(x0, x1)), 0, self.session.data.cols - 1))
        c1 = int(np.clip(round(max(x0, x1)), 0, self.session.data.cols - 1))
        r0 = int(np.clip(round(min(y0, y1)), 0, self.session.data.rows - 1))
        r1 = int(np.clip(round(max(y0, y1)), 0, self.session.data.rows - 1))
        return bool(self._set_roi_bounds(r0, c0, r1 - r0 + 1, c1 - c0 + 1, source="shift-drag"))

    def _log(self, text: str) -> None:
        targets = self.log_texts or [self.log_text]
        for target in targets:
            target.insert(tk.END, text + "\n")
            target.see(tk.END)

    def _set_info_lines(self, lines: list[str]) -> None:
        targets = self.info_texts or ([self.info_text] if self.info_text is not None else [])
        if not targets:
            return
        for target in targets:
            target.configure(state=tk.NORMAL)
            target.delete("1.0", tk.END)
            target.insert("1.0", "\n".join(lines).strip() + "\n")
            target.configure(state=tk.DISABLED)

    def _browse_patterns(self) -> str | None:
        fn = filedialog.askopenfilename(filetypes=[("Pattern files", "*.h5oina *.up1 *.up2"), ("All files", "*.*")])
        if fn:
            self.pattern_path_var.set(str(Path(fn).resolve()))
            suffix = Path(fn).suffix.lower()
            if suffix in {".up1", ".up2", ".h5oina"}:
                self.source_type_var.set("H5OINA" if suffix == ".h5oina" else "UP + ANG")
                self._sync_input_type_controls(reset_up_tilt=True)
            self._refresh_default_workflow_path()
            return str(Path(fn).resolve())

    def _browse_orientation(self) -> str | None:
        fn = filedialog.askopenfilename(filetypes=[("ANG files", "*.ang"), ("All files", "*.*")])
        if fn:
            self.orientation_path_var.set(str(Path(fn).resolve()))
            return str(Path(fn).resolve())

    def _browse_master(self) -> str | None:
        fn = filedialog.askopenfilename(filetypes=[("Master patterns", "*.h5 *.hdf5 *.sdf5"), ("All files", "*.*")])
        if fn:
            self.master_path_var.set(str(Path(fn).resolve()))
            return str(Path(fn).resolve())

    @_guarded_action
    def _choose_and_load_input(self) -> None:
        path = self._browse_patterns()
        if not path:
            return
        if Path(path).suffix.lower() in {".up1", ".up2"} and not self._browse_orientation():
            return
        self._load_input()

    @_guarded_action
    def _choose_and_load_master(self) -> None:
        if self._browse_master():
            self._load_master()

    def _browse_export(self) -> None:
        fn = filedialog.asksaveasfilename(defaultextension=".h5oina", filetypes=[("All files", "*.*")])
        if fn:
            self.export_path_var.set(str(Path(fn).resolve()))

    def _default_workflow_path(self) -> str:
        pattern = self.pattern_path_var.get().strip()
        source = Path(pattern).expanduser().resolve() if pattern else Path.cwd() / "overlap_ebsd"
        stem = self._source_stem(source)
        return str(source.with_name(f"{stem}_overlap_workflow.npz"))

    def _refresh_default_workflow_path(self, *, force: bool = False) -> None:
        current = self.workflow_path_var.get().strip()
        if not force and (self._auto_workflow_path is None or current != self._auto_workflow_path):
            return
        suggested = self._default_workflow_path()
        self.workflow_path_var.set(suggested)
        self._auto_workflow_path = suggested

    @staticmethod
    def _source_stem(path: Path) -> str:
        stem = path.stem
        while Path(stem).suffix.lower() in {".h5oina", ".ang", ".up1", ".up2"}:
            stem = Path(stem).stem
        return stem

    @staticmethod
    def _path_with_single_suffix(path: str | Path, suffix: str) -> Path:
        out = Path(path).expanduser().resolve()
        suffix = suffix.lower()
        name = out.name
        known_export_suffixes = {".ang", ".h5oina", suffix}
        while Path(name).suffix.lower() in known_export_suffixes:
            name = Path(name).stem
        return out.with_name(name + suffix)

    def _default_residual_pattern_path(self) -> str:
        pattern = self.pattern_path_var.get().strip()
        if pattern:
            src = Path(pattern).expanduser().resolve()
            ext = src.suffix.lower()
            if ext in {".h5oina", ".up1", ".up2"}:
                return str(src.with_name(f"{self._source_stem(src)}_residuals{ext}"))
        return str((Path.cwd() / "residual_patterns.h5oina").resolve())

    def _browse_residual_pattern_output(self) -> None:
        current = Path(self.residual_pattern_path_var.get().strip() or self._default_residual_pattern_path())
        ext = current.suffix.lower()
        if ext not in {".h5oina", ".up1", ".up2"}:
            ext = ".h5oina"
        fn = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialfile=current.name,
            initialdir=str(current.parent),
            filetypes=[
                ("Pattern files", "*.h5oina *.up1 *.up2"),
                ("H5OINA files", "*.h5oina"),
                ("EDAX pattern files", "*.up1 *.up2"),
                ("All files", "*.*"),
            ],
        )
        if fn:
            self.residual_pattern_path_var.set(str(Path(fn).resolve()))

    def _default_roi_export_path(self, *, residual: bool = False) -> str:
        tag = "residual" if residual else "primary"
        suffix = ".h5oina"
        source: Path | None = None
        if self.session.data is not None:
            pattern_suffix = Path(self.session.data.pattern_path).suffix.lower()
            if pattern_suffix == ".h5oina":
                source = Path(self.session.data.pattern_path)
                suffix = ".h5oina"
            elif pattern_suffix in {".up1", ".up2"}:
                source_path = self.session.data.orientation_path or self.orientation_path_var.get().strip()
                source = Path(source_path) if source_path else None
                suffix = self._roi_export_suffix()
        if source is not None and source.name:
            return str(source.with_name(f"{self._source_stem(source)}_{tag}_roi{suffix}").resolve())
        return str((Path.cwd() / f"{tag}_roi{suffix}").resolve())

    def _roi_export_suffix(self) -> str:
        data = self.session.data
        if data is None:
            return ".h5oina"
        pattern_suffix = Path(data.pattern_path).suffix.lower()
        if pattern_suffix == ".h5oina":
            return ".h5oina"
        if pattern_suffix in {".up1", ".up2"}:
            format_var = getattr(self, "roi_export_format_var", None)
            if format_var is not None and str(format_var.get()).strip().upper() == "H5OINA":
                return ".h5oina"
            return ".ang"
        return ".ang" if data.source_type == "up_ang" else ".h5oina"

    def _sync_roi_export_paths_to_source(self) -> None:
        default_suffix = self._roi_export_suffix()
        data = self.session.data
        allow_h5oina = bool(
            data is not None
            and Path(getattr(data, "pattern_path", "")).suffix.lower() in {".up1", ".up2"}
        )
        explicit_format = getattr(self, "roi_export_format_var", None) is not None
        for variable, residual in (
            (self.primary_roi_export_path_var, False),
            (self.residual_roi_export_path_var, True),
        ):
            current = variable.get().strip() or self._default_roi_export_path(residual=residual)
            suffix = Path(current).suffix.lower()
            if explicit_format or suffix not in ({".ang", ".h5oina"} if allow_h5oina else {default_suffix}):
                suffix = default_suffix
            variable.set(str(self._path_with_single_suffix(current, suffix)))

    def _default_overlap_optimization_export_path(self) -> str:
        source_path = (
            self.session.data.pattern_path
            if self.session.data is not None
            else self.pattern_path_var.get().strip()
        )
        source = Path(source_path).expanduser().resolve() if source_path else Path.cwd() / "overlap_ebsd"
        return str(source.with_name(f"{self._source_stem(source)}_step4_results.h5"))

    def _browse_overlap_optimization_export(self) -> str | None:
        current = Path(
            self.overlap_optimization_export_path_var.get().strip()
            or self._default_overlap_optimization_export_path()
        )
        current = self._path_with_single_suffix(current, ".h5")
        fn = filedialog.asksaveasfilename(
            defaultextension=".h5",
            initialfile=current.name[: -len(".h5")],
            initialdir=str(current.parent),
            filetypes=[("Step 4 HDF5 results", "*.h5"), ("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not fn:
            return None
        output_path = str(self._path_with_single_suffix(fn, ".h5"))
        self.overlap_optimization_export_path_var.set(output_path)
        return output_path

    def _browse_roi_export(self, var: tk.StringVar, *, residual: bool) -> str | None:
        default_ext = self._roi_export_suffix()
        current = Path(var.get().strip() or self._default_roi_export_path(residual=residual))
        data = self.session.data
        allow_h5oina = bool(
            data is not None
            and Path(getattr(data, "pattern_path", "")).suffix.lower() in {".up1", ".up2"}
        )
        allowed_extensions = {".ang", ".h5oina"} if allow_h5oina else {default_ext}
        ext = current.suffix.lower()
        if getattr(self, "roi_export_format_var", None) is not None or ext not in allowed_extensions:
            ext = default_ext
        current = self._path_with_single_suffix(current, ext)
        # Tk adds ``defaultextension`` to ``initialfile`` on some platforms,
        # notably the native macOS save dialog.  Supplying the suffix in both
        # options therefore presents e.g. ``map.h5oina.h5oina`` as the default.
        initialfile = current.name[: -len(ext)]
        format_labels = {".ang": "ANG files", ".h5oina": "H5OINA files"}
        format_extensions = [ext] + sorted(allowed_extensions - {ext})
        selected_format = tk.StringVar(master=self, value="")
        formats_title = " or ".join(format_extensions)
        fn = filedialog.asksaveasfilename(
            title=f"Save {'residual' if residual else 'primary'} results ({formats_title})",
            defaultextension=ext,
            initialfile=initialfile,
            initialdir=str(current.parent),
            filetypes=[(format_labels[suffix], f"*{suffix}") for suffix in format_extensions],
            typevariable=selected_format,
        )
        if not fn:
            return None
        selected_ext = Path(fn).suffix.lower()
        for suffix in format_extensions:
            if selected_format.get() == format_labels[suffix]:
                selected_ext = suffix
                break
        if selected_ext not in allowed_extensions:
            selected_ext = ext
        output_path = str(self._path_with_single_suffix(fn, selected_ext))
        var.set(output_path)
        return output_path

    def _browse_primary_roi_export(self) -> str | None:
        return self._browse_roi_export(self.primary_roi_export_path_var, residual=False)

    def _browse_residual_roi_export(self) -> str | None:
        return self._browse_roi_export(self.residual_roi_export_path_var, residual=True)

    def _dictionary_filetypes(self) -> list[tuple[str, str]]:
        return [("Binned EBSD dictionary", "*.h5 *.hdf5"), ("All files", "*.*")]

    def _selected_master_energy_mode(self) -> str:
        label = self.master_energy_mode_var.get()
        for mode, candidate in MASTER_ENERGY_MODE_LABELS.items():
            if label == candidate:
                return mode
        if label in MASTER_ENERGY_MODE_LABELS:
            return label
        return MASTER_ENERGY_MODE_HIGHEST

    def _sync_master_energy_mode_from_session(self) -> None:
        master = self.session.master
        mode = getattr(master, "energy_mode", MASTER_ENERGY_MODE_HIGHEST)
        self.master_energy_mode_var.set(
            MASTER_ENERGY_MODE_LABELS.get(str(mode), MASTER_ENERGY_MODE_LABELS[MASTER_ENERGY_MODE_HIGHEST])
        )

    def _refresh_default_dictionary_path(self, *, force: bool = False) -> None:
        current = self.dictionary_path_var.get().strip()
        if not force and (self._auto_dictionary_path is None or current != self._auto_dictionary_path):
            return
        master_stem = Path(self.master_path_var.get().strip() or "master_pattern").stem
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in master_stem).strip("_")
        safe_stem = safe_stem or "master_pattern"
        resolution = f"{float(self.di_res_deg_var.get()):g}".replace(".", "p")
        binning = max(1, int(self.di_binning_var.get()))
        energy_suffix = "_globalMC" if self._selected_master_energy_mode() == MASTER_ENERGY_MODE_GLOBAL else ""
        parent = Path(current).expanduser().resolve().parent if current else Path.cwd()
        suggested = str(
            (parent / f"{safe_stem}_dictionary{energy_suffix}_bin{binning}_{resolution}deg.h5").resolve()
        )
        self.dictionary_path_var.set(suggested)
        self._auto_dictionary_path = suggested

    def _dictionary_dialog_options(self, *, for_save: bool = False) -> dict[str, object]:
        options: dict[str, object] = {"filetypes": self._dictionary_filetypes()}
        raw = self.dictionary_path_var.get().strip()
        if not raw:
            return options
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        parent = candidate.parent
        if parent:
            options["initialdir"] = str(parent.resolve())
        if candidate.name:
            if for_save and candidate.suffix.lower() in {".h5", ".hdf5"}:
                options["initialfile"] = candidate.stem
            else:
                options["initialfile"] = candidate.name
        return options

    def _choose_dictionary_save_path(self) -> str | None:
        selected = filedialog.asksaveasfilename(
            defaultextension=".h5",
            **self._dictionary_dialog_options(for_save=True),
        )
        if not selected:
            return None
        path = str(Path(selected).resolve())
        self.dictionary_path_var.set(path)
        self._auto_dictionary_path = None
        return path

    def _roi_bounds(self) -> tuple[int, int, int, int]:
        if self.session.data is None:
            raise RuntimeError("Load input data first.")
        rows = int(self.session.data.rows)
        cols = int(self.session.data.cols)
        r0 = max(0, min(int(self.roi_r0_var.get()), rows - 1))
        c0 = max(0, min(int(self.roi_c0_var.get()), cols - 1))
        nrows = max(1, min(int(self.roi_nrows_var.get()), rows - r0))
        ncols = max(1, min(int(self.roi_ncols_var.get()), cols - c0))
        return r0, c0, nrows, ncols

    def _calibration_progress(self, value: float, message: str) -> None:
        self._check_job_cancelled()
        def update():
            self._job_result_views.add(0)
            self.status_var.set(message)
        self._post_ui(update)

    def _set_dictionary_progress(self, value: float, message: str) -> None:
        self.dictionary_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.dictionary_status_var.set(message)

    def _set_reindex_progress(self, value: float, message: str) -> None:
        if hasattr(self, "_job_result_views"):
            self._job_result_views.add(1)
        self.reindex_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.reindex_progress_status_var.set(message)

    def _set_complete_analysis_progress(self, value: float, message: str) -> None:
        self.complete_analysis_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.complete_analysis_status_var.set(message)

    def _set_refinement_progress(self, value: float, message: str) -> None:
        if hasattr(self, "_job_result_views"):
            self._job_result_views.add(1)
        self._set_progress_state(
            "refinement",
            self.refinement_progress_var,
            self.refinement_progress_status_var,
            value,
            message,
        )

    def _set_overlap_progress(self, value: float, message: str) -> None:
        if hasattr(self, "_job_result_views"):
            self._job_result_views.add(2)
        self._set_progress_state(
            "overlap",
            self.overlap_progress_var,
            self.overlap_progress_status_var,
            value,
            message,
        )

    def _set_overlap_optimization_progress(self, value: float, message: str) -> None:
        if hasattr(self, "_job_result_views"):
            self._job_result_views.add(3)
        self.overlap_optimization_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.overlap_optimization_status_var.set(message)

    def _update_pattern_mask_status(self) -> None:
        self.pattern_mask_status_var.set(f"Mask: {self.session.pattern_mask_description()}")

    def _update_dynamic_bg_status(self) -> None:
        self.dynamic_bg_status_var.set(f"Dynamic BG: {self.session.dynamic_background_description()}")

    def _sync_pattern_mask_setting(self) -> str:
        try:
            option = int(self.pattern_mask_option_var.get())
        except Exception as exc:
            raise ValueError("Pattern mask must be -1, 0, or a positive integer diameter.") from exc
        previous = int(self.session.pattern_mask_option)
        msg = self.session.set_pattern_mask_option(option)
        if option != previous:
            self.last_overlap = None
            self.last_overlap_mixture = None
        self._update_pattern_mask_status()
        return msg

    def _sync_dynamic_bg_setting(self) -> str:
        enabled = bool(self.dynamic_bg_enabled_var.get())
        raw_std = str(self.dynamic_bg_std_var.get()).strip()
        try:
            std_px = 0.0 if raw_std == "" else float(raw_std)
        except Exception as exc:
            if enabled:
                raise ValueError("Dynamic BG std must be 0 (auto) or a positive pixel value.") from exc
            std_px = 0.0
        if not enabled:
            std_px = 0.0
        previous = self.session.dynamic_bg_config
        msg = self.session.set_dynamic_background(enabled, std_px=std_px)
        if self.session.dynamic_bg_config != previous:
            self.last_overlap = None
            self.last_overlap_mixture = None
        self._update_dynamic_bg_status()
        return msg

    def _sync_pattern_conditioning_settings(self) -> list[str]:
        messages = [self._sync_pattern_mask_setting(), self._sync_dynamic_bg_setting()]
        return messages

    def _sync_pattern_conditioning_for_refresh(self) -> None:
        try:
            self._sync_pattern_conditioning_settings()
            self.session.last_action_note = ""
        except Exception:
            return

    def _overlay_pattern_mask(self, ax) -> None:
        if self.session.data is None:
            return
        try:
            signal_mask = self.session._signal_mask_for_full_pattern()
        except Exception:
            return
        if signal_mask is None:
            return
        mask = np.asarray(signal_mask, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            return
        overlay = np.ma.masked_where(~mask, np.ones(mask.shape, dtype=np.float32))
        ax.imshow(overlay, cmap="gray", alpha=0.28, origin="upper", vmin=0.0, vmax=1.0)
        if np.any(~mask):
            ax.contour((~mask).astype(np.float32), levels=[0.5], colors="cyan", linewidths=1.0)

    def _residual_ncc_threshold(self) -> float:
        try:
            return float(str(self.overlap_min_ncc_var.get()).strip())
        except Exception:
            return 0.15

    def _selected_ipf_direction(self) -> tuple[str, str]:
        direction = str(self.ipf_direction_var.get()).strip().upper()
        if direction not in {"X", "Y", "Z"}:
            direction = "Z"
            self.ipf_direction_var.set(direction)
        return direction.lower(), f"IPF-{direction}"

    @staticmethod
    def _draw_inspection_marker(ax, *, row: int, col: int, ipf: bool) -> object:
        """Mark an axis as scan-selectable and draw its current inspection point."""
        setattr(ax, "_overlap_ebsd_scan_map", True)
        return ax.scatter(
            [col],
            [row],
            marker="+",
            color="black" if ipf else "red",
            s=180,
            linewidths=2.0,
        )

    def _residual_ipf_ncc_threshold(self) -> float:
        try:
            return float(str(self.residual_ipf_ncc_var.get()).strip())
        except Exception:
            return 0.15

    def _overlap_mixture_residual_ncc_threshold(self) -> float:
        try:
            return float(str(self.overlap_mixture_residual_ncc_var.get()).strip())
        except Exception:
            return 0.15

    def _primary_threshold_mask(self) -> np.ndarray | None:
        if self.session.data is None:
            return None
        rows, cols = self.session.data.rows, self.session.data.cols
        indexed_mask = self.session.indexed_mask
        if indexed_mask is None or np.asarray(indexed_mask).shape != (self.session.data.count,):
            mask = np.ones((rows, cols), dtype=bool)
        else:
            mask = ~np.asarray(indexed_mask, dtype=bool).reshape(rows, cols)
        threshold = self._residual_ncc_threshold()
        if threshold <= 0.0:
            return mask
        score_map = self.session.last_scores_map
        if score_map is None or score_map.shape != (rows, cols):
            return mask
        low_score = ~np.isfinite(score_map) | (np.asarray(score_map, dtype=np.float32) < threshold)
        return np.logical_or(mask, low_score)

    def _index_quality_layer_choices(self) -> list[str]:
        if self.session.data is None:
            return ["CI", "BC", "IQ", "BS", "DP", "NCC", "MAD", "Phase"]
        unavailable = {*ORIENTATION_LAYER_LABELS, "X", "Y"}
        layers = [layer for layer in self.session.available_layers() if layer not in unavailable]
        if not layers:
            layers = ["Phase"]
        return layers

    def _default_index_quality_layer(self) -> str:
        choices = self._index_quality_layer_choices()
        preferred_order = ("CI", "BC", "IQ", "BS", "DP", "NCC", "MAD", "Phase")
        if self.session.data is not None:
            if self.session.data.source_type == "h5oina":
                preferred_order = ("BC", "CI", "IQ", "BS", "DP", "NCC", "MAD", "Phase")
            elif self.session.data.source_type == "up_ang":
                preferred_order = ("CI", "BC", "IQ", "BS", "DP", "NCC", "MAD", "Phase")
        for preferred in preferred_order:
            if preferred in choices:
                return preferred
        return choices[0]

    def _sync_index_quality_layer_choices(self) -> None:
        choices = self._index_quality_layer_choices()
        if self.index_quality_layer_var.get() not in choices:
            self.index_quality_layer_var.set(self._default_index_quality_layer())

    def _residual_threshold_mask(self) -> np.ndarray | None:
        threshold = self._residual_ipf_ncc_threshold()
        if threshold <= 0.0 or self.session.data is None:
            return None
        rows, cols = self.session.data.rows, self.session.data.cols
        score_map = self.session.last_residual_scores_map
        if score_map is not None and score_map.shape == (rows, cols):
            return np.isfinite(score_map) & (score_map < threshold)
        mask = np.zeros((rows, cols), dtype=bool)
        for idx, result in self.session.residual_point_results.items():
            score = result.secondary_ncc_kp
            if score is not None and score < threshold:
                row, col = self.session.row_col_from_index(int(idx))
                mask[row, col] = True
        return mask

    def _overlap_mixture_residual_ncc_for_index(self, index: int) -> float | None:
        if self.session.data is None:
            return None
        idx = int(index)
        score_map = self.session.last_residual_scores_map
        if score_map is not None and score_map.shape == (self.session.data.rows, self.session.data.cols):
            row, col = self.session.row_col_from_index(idx)
            score = float(score_map[row, col])
            if np.isfinite(score):
                return score
        result = self.session.residual_point_results.get(idx)
        if result is None:
            return None
        for value in (result.secondary_ncc_kp, result.secondary_ncc_full, result.secondary_dictionary_ncc_kp):
            if value is not None:
                score = float(value)
                if np.isfinite(score):
                    return score
        return None

    def _overlap_mixture_residual_threshold_mask(self) -> np.ndarray | None:
        threshold = self._overlap_mixture_residual_ncc_threshold()
        if threshold <= 0.0 or self.session.data is None:
            return None
        rows, cols = self.session.data.rows, self.session.data.cols
        score_map = self.session.last_residual_scores_map
        if score_map is not None and score_map.shape == (rows, cols):
            return np.isfinite(score_map) & (score_map < threshold)
        mask = np.zeros((rows, cols), dtype=bool)
        for idx in self.session.residual_point_results.keys():
            score = self._overlap_mixture_residual_ncc_for_index(int(idx))
            if score is not None and score < threshold:
                row, col = self.session.row_col_from_index(int(idx))
                mask[row, col] = True
        return mask

    def _filter_overlap_mixture_indices_by_residual_threshold(self, indices: np.ndarray) -> tuple[np.ndarray, int]:
        threshold = self._overlap_mixture_residual_ncc_threshold()
        selected = np.asarray(indices, dtype=np.int64).ravel()
        if threshold <= 0.0:
            return selected, 0
        filtered: list[int] = []
        skipped = 0
        for idx in selected.tolist():
            score = self._overlap_mixture_residual_ncc_for_index(int(idx))
            if score is not None and score >= threshold:
                filtered.append(int(idx))
            else:
                skipped += 1
        return np.asarray(filtered, dtype=np.int64), skipped

    @staticmethod
    def _apply_white_mask(image: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
        if image is None or mask is None:
            return image
        out = np.asarray(image, dtype=np.float32).copy()
        if out.shape[:2] != mask.shape:
            return out
        out[mask] = 1.0
        return out

    def _primary_fit_bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for label, low_var, high_var in self.primary_fit_bound_specs:
            low = float(low_var.get())
            high = float(high_var.get())
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError(f"{label} bounds must be finite.")
            if high <= low:
                raise ValueError(f"{label} upper bound must be greater than the lower bound.")
            bounds.append((low, high))
        return bounds

    def _selected_primary_ncc(self, index: int) -> float | None:
        score = self.session.get_primary_index_ncc(int(index))
        return None if score is None else float(score)

    def _filter_roi_indices_by_threshold(self, indices: np.ndarray) -> tuple[np.ndarray, int]:
        threshold = self._residual_ncc_threshold()
        filtered: list[int] = []
        skipped = 0
        for idx in np.asarray(indices, dtype=np.int64).ravel().tolist():
            score = self.session.get_primary_index_ncc(int(idx))
            if score is not None and (threshold <= 0.0 or score >= threshold):
                filtered.append(int(idx))
            else:
                skipped += 1
        return np.asarray(filtered, dtype=np.int64), skipped

    def _update_dictionary_status(self) -> None:
        self._sync_refinement_settings()
        cache = self.session.dictionary_cache
        if cache is None:
            self.dictionary_status_var.set("No dictionary generated or loaded.")
            self.dictionary_progress_var.set(0.0)
            return
        storage_note = (
            "temporary disk cache — save to keep"
            if cache.owns_storage
            else f"disk-backed file={Path(cache.storage_path).name}" if cache.storage_path else "in memory"
        )
        self.dictionary_status_var.set(
            f"Ready: {cache.rotation_count} patterns, {cache.pattern_shape[0]}x{cache.pattern_shape[1]}, "
            f"binning={cache.software_binning}, resolution={cache.resolution_deg:g}°, "
            f"MP energy={MASTER_ENERGY_MODE_LABELS.get(cache.master_energy_mode, cache.master_energy_mode)}, "
            f"dtype={cache.pattern_dtype}; {storage_note}"
        )
        self.dictionary_progress_var.set(100.0)

    def _sync_residual_keep_n_to_dictionary(self, keep_n: int) -> None:
        self.residual_keep_n_var.set(max(1, int(keep_n)))

    def _autosave_stage(self, stage: str) -> None:
        context = getattr(self, "_job_autosave", None)
        if context is None:
            return
        path, ui_state = context
        try:
            save_checkpoint(self.session, path, ui_state)
            message = f"Workflow autosaved after {stage}: {path}"
        except Exception as exc:
            message = f"Workflow autosave FAILED after {stage}: {exc}. Use Save to retry."
        self._last_autosave_message = message
        self._post_ui(lambda message=message: self._log(message))

    def _run_threaded(self, fn, *, on_success=None, sync_conditioning: bool = True,
                      autosave: bool = False) -> bool:
        if self.busy:
            return False
        try:
            if sync_conditioning:
                self._sync_pattern_conditioning_settings()
            self._job_autosave = None
            self._last_autosave_message = ""
            if autosave:
                path = Path(self.workflow_path_var.get().strip() or self._default_workflow_path()).expanduser().resolve()
                while path.suffix.lower() == ".npz":
                    path = path.with_suffix("")
                path = path.with_name(path.name + ".npz")
                self.workflow_path_var.set(str(path))
                self._job_autosave = (path, self._workflow_ui_state())
            self.session.last_action_note = ""
        except Exception as exc:
            self._pending_restore_path = None
            messagebox.showerror("Invalid pattern conditioning", str(exc))
            return False
        import queue
        self._ui_callback_queue = queue.SimpleQueue()
        self._cancel_requested = threading.Event()
        self._job_update_gate = LiveUpdateGate()
        self._job_result_views = set()
        self._set_busy(True)
        self.status_var.set("Running...")
        self._set_info_lines(["Running...", "Check the log for detailed step updates."])

        def finish(msg):
            try:
                if on_success is not None:
                    updated = on_success(msg)
                    if updated is not None:
                        msg = updated
                self._on_action_done(msg)
            except Exception as exc:
                self._on_action_error(exc, traceback.format_exc())

        def worker() -> None:
            try:
                self._check_job_cancelled()
                msg = fn()
                if autosave:
                    MultiStepOverlapGUI._autosave_stage(self, "completed analysis")
                    msg = f"{msg} {self._last_autosave_message}"
                self._post_ui(lambda msg=msg: finish(msg))
            except Exception as exc:
                detail = traceback.format_exc()
                self._post_ui(lambda exc=exc, detail=detail: self._on_action_error(exc, detail))
            finally:
                self._job_autosave = None

        self._worker_thread = threading.Thread(target=worker, daemon=False, name="overlap-ebsd-worker")
        self.after(25, self._drain_ui_callbacks)
        self._worker_thread.start()
        return True

    def _update_mode_controls(self) -> None:
        loaded = self.session.data is not None
        master = self.session.master is not None
        dictionary = self.session.dictionary_cache is not None
        ready = not self.busy
        conditions = {
            "btn_refine_roi": loaded and master and bool(self.session.calibration_indices),
            "btn_index_roi": loaded and master and dictionary,
            "btn_refine_indexed": loaded and master and self.session.last_indexed_indices is not None,
            "btn_steps_2_3_analysis": loaded and master and dictionary,
            "btn_complete_analysis": loaded and master and dictionary,
            "btn_steps_3_4_analysis": loaded and master and dictionary,
            "btn_save_dictionary": dictionary,
        }
        for name, allowed in conditions.items():
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=tk.NORMAL if ready and allowed else tk.DISABLED)
        self._update_calibration_application_controls()

    def _on_action_done(self, msg: str) -> None:
        self._worker_thread = None
        self._job_result_views = set()
        self.live_update_status_var.set("Speed priority: maps refresh between batches, less often when drawing is expensive.")
        self._set_busy(False)
        if self._pending_restore_path is not None:
            restore_path = self._pending_restore_path
            self._pending_restore_path = None
            self._finish_workflow_restore(restore_path)
        if self.session.last_action_note:
            msg = f"{msg} {self.session.last_action_note}"
            self.session.last_action_note = ""
        self.status_var.set(msg)
        self._log(msg)
        self._update_calibration_summary()
        self._update_dictionary_status()
        self._update_pattern_mask_status()
        self._update_dynamic_bg_status()
        self._populate_point_vars()
        self._refresh_plot()
        self._refresh_context_summary()

    def _on_action_error(self, exc: Exception, detail: str) -> None:
        self._worker_thread = None
        self._pending_restore_path = None
        self._job_result_views = set()
        self.live_update_status_var.set("Live maps: completed results retained.")
        self._set_busy(False)
        if isinstance(exc, InterruptedError):
            self.status_var.set(str(exc))
            self._log(str(exc))
            self._update_dictionary_status()
            self._populate_point_vars()
            self._refresh_plot()
            self._refresh_context_summary()
            return
        self.status_var.set(f"Error: {exc}")
        self._log(detail)
        self._set_info_lines([f"Error: {exc}", "See Log for traceback details."])
        messagebox.showerror("Error", str(exc))

    @_guarded_action
    def _load_input(self) -> None:
        geom = GeometryConfig(
            pc_convention="edax",
            sample_tilt_deg=float(self.sample_tilt_var.get()),
            detector_tilt_deg=float(self.detector_tilt_var.get()),
            azimuthal_deg=0.0,
            twist_deg=0.0,
            phi1_offset_deg=0.0,
        )
        if not np.isfinite([geom.sample_tilt_deg, geom.detector_tilt_deg]).all():
            raise ValueError("Sample and detector tilts must be finite angles.")
        pattern_path = self.pattern_path_var.get().strip()
        orientation_path = self.orientation_path_var.get().strip() or None
        loaded = WorkflowSession()
        loaded.set_pattern_mask_option(int(self.pattern_mask_option_var.get()))
        background_enabled = bool(self.dynamic_bg_enabled_var.get())
        background_std = float(str(self.dynamic_bg_std_var.get()).strip() or "0") if background_enabled else 0.0
        loaded.set_dynamic_background(background_enabled, std_px=background_std)
        def action() -> str:
            try:
                return loaded.load_input(pattern_path=pattern_path, orientation_path=orientation_path, geom=geom)
            except Exception:
                loaded._clear_dictionary_cache()
                raise
        def commit(message):
            previous = self.session
            self.session = loaded
            previous._clear_dictionary_cache()
            return self._finish_input_load(message)
        self._run_threaded(action, on_success=commit, sync_conditioning=False)

    @staticmethod
    def _default_input_phase_id(data) -> int:
        phases = np.asarray(data.phases).ravel()
        choices, counts = np.unique(phases, return_counts=True)
        eligible = choices > 0 if data.source_type == "h5oina" else choices >= 0
        if np.any(eligible):
            return int(choices[eligible][np.argmax(counts[eligible])])
        if data.source_type == "h5oina":
            metadata_ids = sorted(int(pid) for pid in getattr(data, "phase_symmetries", {}) if int(pid) > 0)
            return metadata_ids[0] if metadata_ids else 1
        return int(choices[0]) if choices.size else 1

    def _finish_input_load(self, msg: str) -> str:
        self.last_overlap = None
        self.last_overlap_mixture = None
        self.index_var.set(0)
        self.row_var.set(0)
        self.col_var.set(0)
        if self.session.data is not None:
            layers = self.session.available_layers() or ["Phase"]
            self._sync_index_quality_layer_choices()
            for view_index, plot_view in self._plot_views.items():
                if view_index == 0 and plot_view.get("combo") is not None:
                    plot_view["combo"]["values"] = layers
            if self.map_layer_var.get() not in layers:
                self.map_layer_var.set(layers[0])
            if self.index_quality_layer_var.get() not in self._index_quality_layer_choices():
                self.index_quality_layer_var.set(self._default_index_quality_layer())
            self.roi_r0_var.set(0)
            self.roi_c0_var.set(0)
            self.roi_nrows_var.set(int(self.session.data.rows))
            self.roi_ncols_var.set(int(self.session.data.cols))
            unique_phases = np.unique(self.session.data.phases)
            chosen = self._default_input_phase_id(self.session.data)
            self.phase_id_var.set(chosen)
            msg = (
                f"{msg} Auto-set phase ID to {chosen}. "
                f"Available phases: {unique_phases.tolist()}"
            )
            if self.session.data.source_type == "up_ang":
                msg = (
                    f"{msg} UP mode keeps patterns on disk and batches ROI indexing."
                )
                self.roi_export_format_var.set("ANG")
            else:
                self.roi_export_format_var.set("H5OINA")
            self.residual_pattern_path_var.set(self._default_residual_pattern_path())
            self.primary_roi_export_path_var.set(self._default_roi_export_path(residual=False))
            self.residual_roi_export_path_var.set(self._default_roi_export_path(residual=True))
            self.overlap_optimization_export_path_var.set(self._default_overlap_optimization_export_path())
            # A newly loaded scan must not inherit a restored or Save-as path
            # belonging to the previous scan (including for automatic saves).
            self._refresh_default_workflow_path(force=True)
            imported_count = int(np.count_nonzero(self.session.indexed_mask))
            if imported_count and not self.session.indexed_mask[0]:
                first_indexed = int(np.flatnonzero(self.session.indexed_mask)[0])
                row, col = divmod(first_indexed, int(self.session.data.cols))
                self.index_var.set(first_indexed)
                self.row_var.set(row)
                self.col_var.set(col)
            self._set_reindex_progress(
                100.0 if imported_count else 0.0,
                (f"Loaded {imported_count}/{self.session.data.count} indexed primary point(s); DI can be skipped."
                 if imported_count else "Ready for primary dictionary indexing."),
            )
            self._update_mode_controls()
            self._update_calibration_summary()
            self._populate_point_vars()
        self._sync_loaded_geometry_controls()
        self._refresh_context_summary()
        return f"{msg} Load the master pattern for this input."

    def _sync_loaded_geometry_controls(self) -> None:
        data = self.session.data
        if data is None:
            return
        self.source_type_var.set("UP + ANG" if data.source_type == "up_ang" else "H5OINA")
        for variable, angle in (
            (self.sample_tilt_var, data.sample_tilt_deg),
            (self.detector_tilt_var, data.detector_tilt_deg),
        ):
            if data.source_type == "h5oina":
                # Float32 radians acquire insignificant digits on conversion to
                # degrees. These read-only fields show millidegrees; retain the
                # original full-precision geometry in the loaded data.
                value = round(float(angle), 3)
                variable.set(f"{value if value else 0:g}")
            else:
                variable.set(float(angle))
        pitch = getattr(data, "effective_detector_px_size_um", None)
        self.detector_px_size_var.set("" if pitch is None else f"{float(pitch):g}")
        self.detector_pixel_source_var.set(getattr(data, "detector_pixel_size_source", "Unknown"))
        self._sync_input_type_controls()
        self._update_dictionary_binned_size()

    def _finish_workflow_restore(self, restore_path: str) -> None:
        self.last_overlap = None
        self.last_overlap_mixture = None
        self.workflow_path_var.set(str(Path(restore_path).resolve()))
        self._auto_workflow_path = None
        if self.session.data is None:
            return
        self.pattern_path_var.set(self.session.data.pattern_path)
        self.orientation_path_var.set(self.session.data.orientation_path or "")
        self.master_path_var.set(self.session.master.path if self.session.master is not None else "")
        self.pattern_mask_option_var.set(int(self.session.pattern_mask_option))
        self.dynamic_bg_enabled_var.set(bool(self.session.dynamic_bg_config.enabled))
        self.dynamic_bg_std_var.set(f"{float(self.session.dynamic_bg_config.std_px):g}")
        layers = self.session.available_layers()
        for view_index, plot_view in self._plot_views.items():
            if view_index == 0 and plot_view.get("combo") is not None:
                plot_view["combo"]["values"] = layers
        if self.map_layer_var.get() not in layers and layers:
            self.map_layer_var.set(layers[0])
        self._sync_index_quality_layer_choices()
        self.roi_export_format_var.set(
            "ANG" if self.session.data.source_type == "up_ang" else "H5OINA"
        )
        self._sync_loaded_geometry_controls()
        self._apply_workflow_ui_state(self.session.restored_ui_state)
        # The loaded input determines the valid solution-map format, even if
        # a saved UI path came from another source type or an older workflow.
        self._sync_roi_export_paths_to_source()
        self._sync_master_energy_mode_from_session()
        cache = self.session.dictionary_cache
        if cache is not None:
            self.dictionary_path_var.set(str(cache.storage_path or ""))
            self.phase_id_var.set(int(cache.phase_id))
            self.di_res_deg_var.set(float(cache.resolution_deg))
            self.di_binning_var.set(int(cache.software_binning))
        else:
            settings = self.session.dictionary_settings or {}
            if "resolution_deg" in settings:
                self.di_res_deg_var.set(float(settings["resolution_deg"]))
            if "software_binning" in settings:
                self.di_binning_var.set(int(settings["software_binning"]))
        primary_candidates = self.session.indexed_candidate_eulers_rad
        residual_candidates = self.session.residual_candidate_eulers_rad
        candidate_count = max(
            int(primary_candidates.shape[1]) if primary_candidates is not None else 1,
            int(residual_candidates.shape[1]) if residual_candidates is not None else 1,
        )
        if "dictionary_keep_n" not in self.session.restored_ui_state:
            if "residual_keep_n" not in self.session.restored_ui_state:
                self.dictionary_keep_n_var.set(max(5, candidate_count))
        if not self.residual_pattern_path_var.get().strip():
            self.residual_pattern_path_var.set(
                self.session.residual_pattern_output_path or self._default_residual_pattern_path()
            )
        if (
            "step4_export_path" not in self.session.restored_ui_state
            or not self.overlap_optimization_export_path_var.get().strip()
        ):
            self.overlap_optimization_export_path_var.set(self._default_overlap_optimization_export_path())
        self._sync_refinement_settings()
        self._update_mode_controls()
        self._refresh_context_summary()

    @_guarded_action
    def _load_master(self) -> None:
        path = self.master_path_var.get().strip()
        def action() -> str:
            return self.session.load_master(master_path=path, energy_kv=None, energy_mode=MASTER_ENERGY_MODE_HIGHEST)
        self._run_threaded(action, on_success=lambda _msg: self._sync_master_energy_mode_from_session())

    @_guarded_action
    def _export_results(self) -> None:
        path = self.export_path_var.get().strip()
        self._run_threaded(lambda: self.session.export_reindexed_results(path), sync_conditioning=False)

    def _workflow_ui_state(self) -> dict[str, object]:
        variables = {
            "phase_id": self.phase_id_var,
            "master_energy_mode": self.master_energy_mode_var,
            "selected_index": self.index_var,
            "selected_row": self.row_var,
            "selected_col": self.col_var,
            "roi_r0": self.roi_r0_var,
            "roi_c0": self.roi_c0_var,
            "roi_nrows": self.roi_nrows_var,
            "roi_ncols": self.roi_ncols_var,
            "trust_euler": self.trust_euler_var,
            "trust_pc": self.trust_pc_var,
            "maxfev": self.maxfev_var,
            "refine_full_resolution": self.refine_full_resolution_var,
            "dictionary_resolution": self.di_res_deg_var,
            "dictionary_binning": self.di_binning_var,
            "dictionary_keep_n": self.dictionary_keep_n_var,
            "parallel_cores": self.parallel_cores_var,
            "auto_refine": self.auto_refine_var,
            "follow_dictionary_trust": self.follow_dictionary_trust_var,
            "calibration_trust_euler": self.calibration_trust_euler_var,
            "calibration_maxfev": self.calibration_maxfev_var,
            "blur_sigma": self.blur_sigma_var,
            "fit_blur_gain": self.fit_blur_gain_var,
            "gain_fit_maxiter": self.gain_fit_maxiter_var,
            "gain_fit_popsize": self.gain_fit_popsize_var,
            "overlap_min_ncc": self.overlap_min_ncc_var,
            "residual_ipf_ncc": self.residual_ipf_ncc_var,
            "write_residual_patterns": self.write_residual_patterns_var,
            "include_primary_patterns_export": self.include_primary_patterns_export_var,
            "include_residual_patterns_export": self.include_residual_patterns_export_var,
            "roi_export_format": self.roi_export_format_var,
            "residual_pattern_path": self.residual_pattern_path_var,
            "primary_roi_export_path": self.primary_roi_export_path_var,
            "residual_roi_export_path": self.residual_roi_export_path_var,
            "step4_export_path": self.overlap_optimization_export_path_var,
            "mixture_trust_euler": self.overlap_mixture_trust_euler_var,
            "mixture_maxfev": self.overlap_mixture_maxfev_var,
            "mixture_residual_ncc": self.overlap_mixture_residual_ncc_var,
            "use_scan_pc_shift": self.use_scan_pc_shift_var,
            "effective_detector_px_size_um": self.detector_px_size_var,
            "ipf_direction": self.ipf_direction_var,
            "solution_map": self.solution_map_var,
        }
        state = {key: variable.get() for key, variable in variables.items()}
        state["fit_method"] = _selected_fit_method(self)
        state["primary_fit_bounds"] = [
            [float(low.get()), float(high.get())] for _label, low, high in self.primary_fit_bound_specs
        ]
        if self.workflow_notebook is not None:
            state["selected_workflow_tab"] = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        refresh_calibration = getattr(self, "_update_calibration_application_controls", None)
        if refresh_calibration is not None:
            refresh_calibration()
        calibration_state = getattr(self, "_calibration_apply_state", "none")
        applied_settings = getattr(self, "_applied_calibration_settings", None)
        if (
            not isinstance(calibration_state, str)
            or calibration_state not in {"none", "pending", "applied"}
            or getattr(self, "_calibration_apply_session", None) is not self.session
        ):
            calibration_state = "none"
        state["calibration_apply_state"] = calibration_state
        state["applied_calibration_settings"] = (
            list(applied_settings)
            if calibration_state != "none" and isinstance(applied_settings, (tuple, list))
            else None
        )
        return state

    def _apply_workflow_ui_state(self, state: dict[str, object]) -> None:
        state = dict(state)
        try:
            fit_method = validate_fit_method(state.get("fit_method", FIT_METHOD_DEFAULT))
        except (TypeError, ValueError):
            fit_method = FIT_METHOD_DEFAULT
        self.fit_method_var.set(FIT_METHOD_LABELS[fit_method])
        if self.session.data is not None:
            state.setdefault("roi_r0", 0)
            state.setdefault("roi_c0", 0)
            state.setdefault("roi_nrows", int(self.session.data.rows))
            state.setdefault("roi_ncols", int(self.session.data.cols))
        for primary, legacy in (
            ("dictionary_keep_n", "residual_keep_n"),
            ("trust_euler", "residual_trust_euler"),
            ("maxfev", "residual_maxfev"),
            ("refine_full_resolution", "residual_refine_full_resolution"),
            ("parallel_cores", "step3_parallel_cores"),
        ):
            if primary not in state and legacy in state:
                state[primary] = state[legacy]
            state.pop(legacy, None)
        if "parallel_cores" not in state and "step4_parallel_cores" in state:
            state["parallel_cores"] = state["step4_parallel_cores"]
        state.pop("step4_parallel_cores", None)
        variables = {
            "phase_id": self.phase_id_var,
            "master_energy_mode": self.master_energy_mode_var,
            "selected_index": self.index_var,
            "selected_row": self.row_var,
            "selected_col": self.col_var,
            "roi_r0": self.roi_r0_var,
            "roi_c0": self.roi_c0_var,
            "roi_nrows": self.roi_nrows_var,
            "roi_ncols": self.roi_ncols_var,
            "trust_euler": self.trust_euler_var,
            "trust_pc": self.trust_pc_var,
            "maxfev": self.maxfev_var,
            "refine_full_resolution": self.refine_full_resolution_var,
            "dictionary_resolution": self.di_res_deg_var,
            "dictionary_binning": self.di_binning_var,
            "dictionary_keep_n": self.dictionary_keep_n_var,
            "parallel_cores": self.parallel_cores_var,
            "auto_refine": self.auto_refine_var,
            "follow_dictionary_trust": self.follow_dictionary_trust_var,
            "calibration_trust_euler": self.calibration_trust_euler_var,
            "calibration_maxfev": self.calibration_maxfev_var,
            "blur_sigma": self.blur_sigma_var,
            "fit_blur_gain": self.fit_blur_gain_var,
            "gain_fit_maxiter": self.gain_fit_maxiter_var,
            "gain_fit_popsize": self.gain_fit_popsize_var,
            "overlap_min_ncc": self.overlap_min_ncc_var,
            "residual_ipf_ncc": self.residual_ipf_ncc_var,
            "write_residual_patterns": self.write_residual_patterns_var,
            "include_primary_patterns_export": self.include_primary_patterns_export_var,
            "include_residual_patterns_export": self.include_residual_patterns_export_var,
            "roi_export_format": self.roi_export_format_var,
            "residual_pattern_path": self.residual_pattern_path_var,
            "primary_roi_export_path": self.primary_roi_export_path_var,
            "residual_roi_export_path": self.residual_roi_export_path_var,
            "step4_export_path": self.overlap_optimization_export_path_var,
            "mixture_trust_euler": self.overlap_mixture_trust_euler_var,
            "mixture_maxfev": self.overlap_mixture_maxfev_var,
            "mixture_residual_ncc": self.overlap_mixture_residual_ncc_var,
            "use_scan_pc_shift": self.use_scan_pc_shift_var,
            "effective_detector_px_size_um": self.detector_px_size_var,
            "ipf_direction": self.ipf_direction_var,
            "solution_map": self.solution_map_var,
        }
        self._suspend_point_trace = True
        try:
            if "follow_dictionary_trust" not in state and "trust_euler" in state:
                state["follow_dictionary_trust"] = False
            if "follow_dictionary_trust" in state:
                previous_follow = self.follow_dictionary_trust_var.get()
                try:
                    self.follow_dictionary_trust_var.set(state["follow_dictionary_trust"])
                    self.follow_dictionary_trust_var.get()
                except (ValueError, TypeError, tk.TclError):
                    self.follow_dictionary_trust_var.set(previous_follow)
            for key, variable in variables.items():
                if key in state:
                    previous = variable.get()
                    try:
                        variable.set(state[key])
                        value = variable.get()
                        if isinstance(variable, (tk.IntVar, tk.DoubleVar)) and not np.isfinite(float(value)):
                            raise ValueError(f"Non-finite saved setting: {key}")
                        if key in {
                            "dictionary_keep_n", "dictionary_binning", "dictionary_resolution",
                            "maxfev", "calibration_maxfev", "calibration_trust_euler", "trust_euler",
                            "parallel_cores", "gain_fit_maxiter", "gain_fit_popsize",
                        } and float(value) <= 0:
                            raise ValueError(f"Non-positive saved setting: {key}")
                    except (ValueError, TypeError, tk.TclError):
                        variable.set(previous)
            bounds = state.get("primary_fit_bounds")
            if isinstance(bounds, list):
                for (_label, low, high), pair in zip(self.primary_fit_bound_specs, bounds):
                    if isinstance(pair, list) and len(pair) == 2:
                        try:
                            lo, hi = float(pair[0]), float(pair[1])
                            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                                low.set(lo)
                                high.set(hi)
                        except (ValueError, TypeError):
                            pass
        finally:
            self._suspend_point_trace = False
        if self.session.data is not None:
            r0, c0, nrows, ncols = self._roi_bounds()
            self._suspend_point_trace = True
            try:
                self.roi_r0_var.set(r0)
                self.roi_c0_var.set(c0)
                self.roi_nrows_var.set(nrows)
                self.roi_ncols_var.set(ncols)
                index = max(0, min(int(self.index_var.get()), self.session.data.count - 1))
                row, col = self.session.row_col_from_index(index)
                self.index_var.set(index)
                self.row_var.set(row)
                self.col_var.set(col)
            finally:
                self._suspend_point_trace = False
        tab = state.get("selected_workflow_tab")
        if self.workflow_notebook is not None and tab is not None:
            try:
                self.workflow_notebook.select(max(0, min(int(tab), 3)))
            except Exception:
                pass
        # Old workflows do not establish whether their calibration was applied.
        # Restore this after setting the variables so their traces cannot treat
        # restoration as an edit to the previous session's application state.
        calibration_state = state.get("calibration_apply_state", "none")
        applied_settings = state.get("applied_calibration_settings")
        valid_settings = False
        if isinstance(applied_settings, (tuple, list)) and len(applied_settings) == 2:
            correction, pitch = applied_settings
            if isinstance(correction, bool):
                if correction:
                    valid_settings = (
                        isinstance(pitch, (int, float)) and not isinstance(pitch, bool)
                        and np.isfinite(pitch) and pitch > 0
                    )
                else:
                    valid_settings = pitch is None
        if (
            not isinstance(calibration_state, str)
            or calibration_state not in {"none", "pending", "applied"}
            or (calibration_state == "applied" and not valid_settings)
            or (applied_settings is not None and not valid_settings)
        ):
            calibration_state = "none"
        self._calibration_apply_state = calibration_state
        self._calibration_apply_session = self.session
        self._applied_calibration_settings = (
            tuple(applied_settings) if calibration_state != "none" and valid_settings else None
        )
        refresh_calibration = getattr(self, "_update_calibration_application_controls", None)
        if refresh_calibration is not None:
            refresh_calibration()

    @_guarded_action
    def _save_workflow(self, output_path: str | None = None) -> None:
        raw = output_path or self.workflow_path_var.get().strip() or self._default_workflow_path()
        path = Path(raw).expanduser().resolve()
        while path.suffix.lower() == ".npz":
            path = path.with_suffix("")
        path = path.with_name(path.name + ".npz")
        self.workflow_path_var.set(str(path))
        self._auto_workflow_path = None
        ui_state = self._workflow_ui_state()
        self._run_threaded(lambda: self.session.save_workflow_state(str(path), ui_state=ui_state), sync_conditioning=False)

    @_guarded_action
    def _save_workflow_as(self) -> None:
        current = Path(self.workflow_path_var.get().strip() or self._default_workflow_path())
        while current.suffix.lower() == ".npz":
            current = current.with_suffix("")
        fn = filedialog.asksaveasfilename(
            defaultextension=".npz",
            initialfile=current.name,
            initialdir=str(current.parent),
            filetypes=[("Overlap workflow", "*.npz"), ("All files", "*.*")],
        )
        if fn:
            self._save_workflow(fn)

    @_guarded_action
    def _restore_workflow(self) -> None:
        fn = filedialog.askopenfilename(filetypes=[("Overlap workflow", "*.npz"), ("All files", "*.*")])
        if not fn:
            return
        restore_path = str(Path(fn).resolve())
        restored = WorkflowSession()
        def action():
            try:
                return restored.restore_workflow_state(restore_path)
            except Exception:
                restored.close()
                raise
        def commit(_message):
            previous = self.session
            self.session = restored
            self._pending_restore_path = restore_path
            previous.close()
        self._run_threaded(action, on_success=commit, sync_conditioning=False)

    @_guarded_action
    def _center_roi_on_selected(self) -> None:
        if self.session.data is None:
            return
        rows = self.session.data.rows
        cols = self.session.data.cols
        height = max(1, min(int(self.roi_nrows_var.get()), rows))
        width = max(1, min(int(self.roi_ncols_var.get()), cols))
        row = int(self.row_var.get())
        col = int(self.col_var.get())
        r0 = max(0, min(row - height // 2, rows - height))
        c0 = max(0, min(col - width // 2, cols - width))
        self._set_roi_bounds(r0, c0, height, width, source="centered on selected point")

    @_guarded_action
    def _use_full_map_roi(self) -> None:
        if self.session.data is None:
            return
        self._set_roi_bounds(
            0,
            0,
            int(self.session.data.rows),
            int(self.session.data.cols),
            source="full map",
        )

    @_guarded_action
    def _sync_row_col_from_index(self) -> None:
        if self.session.data is None:
            return
        idx = max(0, min(int(self.index_var.get()), self.session.data.count - 1))
        self.index_var.set(idx)
        row, col = self.session.row_col_from_index(idx)
        self.row_var.set(int(row))
        self.col_var.set(int(col))
        self._populate_point_vars()
        self._refresh_plot()

    @_guarded_action
    def _sync_index_from_row_col(self) -> None:
        if self.session.data is None:
            return
        row = int(self.row_var.get())
        col = int(self.col_var.get())
        idx = self.session.index_from_row_col(row, col)
        self.index_var.set(int(idx))
        self._populate_point_vars()
        self._refresh_plot()

    def _load_selected_point_values(self) -> None:
        try:
            self._populate_point_vars()
            self.status_var.set("Loaded current Euler/PC values for selected point.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _quantize(self, value: float, step: float) -> float:
        step_f = float(step)
        if step_f <= 0:
            return float(value)
        return float(np.round(float(value) / step_f) * step_f)

    @_guarded_action
    def _apply_selected_point_values(self) -> None:
        idx = int(self.index_var.get())
        state = self.session.get_point_state(idx)
        eulers = np.asarray([float(var.get()) for var in
                             (self.euler1_deg_var, self.euler2_deg_var, self.euler3_deg_var)], dtype=np.float64)
        pc = np.asarray([float(var.get()) for var in
                         (self.pcx_var, self.pcy_var, self.pcz_var)], dtype=np.float64)
        changes = {}
        if not np.array_equal(eulers, np.asarray(state["euler_deg"]), equal_nan=True):
            if not np.all(np.isfinite(eulers)):
                raise ValueError("Edited Euler angles must be finite.")
            changes["euler_deg"] = tuple(float(value) for value in eulers)
        if not np.array_equal(pc, np.asarray(state["pc_custom"]), equal_nan=True):
            if not np.all(np.isfinite(pc)) or pc[2] <= 0.0:
                raise ValueError("Edited PC values must be finite, with positive detector distance.")
            changes["pc_custom"] = tuple(float(value) for value in pc)
        if not changes:
            self.status_var.set("No changes applied.")
            return
        if "pc_custom" in changes:
            self._mark_calibration_unapplied()
        self._run_threaded(lambda: self.session.set_point_state(idx, **changes), sync_conditioning=False)

    # -------------------------- Step 1 -------------------------- #

    def _calibration_application_settings(self) -> tuple[bool, float | None]:
        enabled = bool(self.use_scan_pc_shift_var.get())
        if not enabled:
            return False, None
        raw_pitch = self.detector_px_size_var.get().strip()
        pitch = float(raw_pitch) if raw_pitch else None
        if pitch is None or not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("Enter a positive effective detector pixel size in µm before enabling scan-position correction.")
        return True, pitch

    def _mark_calibration_unapplied(self) -> None:
        self._calibration_apply_session = self.session
        self._calibration_apply_state = "pending"
        self._applied_calibration_settings = None
        self._update_calibration_application_controls()

    def _update_calibration_application_controls(self, *_args) -> None:
        if self._calibration_apply_session is not self.session:
            self._calibration_apply_session = self.session
            self._calibration_apply_state = "none"
            self._applied_calibration_settings = None
        if self.session.data is None or not self.session.calibration_indices:
            self._calibration_apply_state = "none"
            self._applied_calibration_settings = None
        if self._calibration_apply_state == "applied":
            try:
                settings = self._calibration_application_settings()
                matches = settings == self._applied_calibration_settings
                if settings[0]:
                    actual_pitch = getattr(self.session.data, "effective_detector_px_size_um", None)
                    matches = matches and actual_pitch is not None and np.isclose(
                        actual_pitch, settings[1], rtol=1e-10, atol=1e-12,
                    )
            except (ValueError, TypeError, tk.TclError):
                matches = False
            if not matches:
                self._calibration_apply_state = "pending"
        state = self._calibration_apply_state
        if state == "pending":
            message = "Apply required — calibration changes are not yet applied to the entire map."
            text, style = "2. Apply average PC to map — required", "PendingCalibration.TButton"
            foreground, background = "#9f1239", "#fff1f2"
        elif state == "applied":
            message = "✓ Average PC applied to the entire map. Ready to continue."
            text, style = "✓ PC applied · Reapply average", "AppliedCalibration.TButton"
            foreground, background = "#166534", "#f0fdf4"
        else:
            message = "To recalibrate: 1. Optimize points → 2. Apply average PC."
            text, style = "2. Apply average PC to map", "TButton"
            foreground, background = "#374151", "#f3f4f6"
        self.calibration_apply_status_var.set(message)
        if not hasattr(self, "btn_apply_calibration"):
            return
        self._calibration_apply_status_label.configure(foreground=foreground, background=background)
        self.btn_apply_calibration.configure(
            text=text, style=style,
            state=tk.NORMAL if not self.busy and self.session.data is not None and self.session.calibration_indices else tk.DISABLED,
        )
        self._calibration_warning_button.configure(state=tk.DISABLED if self.busy else tk.NORMAL)
        selected_tab = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        if state == "pending" and selected_tab != 0:
            self._calibration_warning_banner.pack(fill=tk.X, before=self.workflow_notebook)
        else:
            self._calibration_warning_banner.pack_forget()

    def _finish_calibration_application(self, message: str, settings: tuple[bool, float | None]) -> str:
        self._calibration_apply_session = self.session
        self._calibration_apply_state = "applied"
        self._applied_calibration_settings = settings
        self._update_calibration_application_controls()
        return f"{message} Calibration applied to the entire map; ready to continue."

    def _update_calibration_summary(self) -> None:
        completed = self._completed_calibration_report
        if completed is not None and (
            completed[0] is not self.session
            or completed[1] != tuple(self.session.calibration_indices)
        ):
            self._completed_calibration_report = completed = None
        if completed is None:
            self.calibration_statistics_var.set("")
            self._calibration_statistics_label.pack_forget()
            self.calibration_summary_var.set(self.session.calibration_point_summary(include_statistics=False))
        else:
            self.calibration_statistics_var.set(completed[2])
            self._calibration_statistics_label.pack(fill=tk.X, pady=(3, 5), before=self._calibration_report_section)
            self.calibration_summary_var.set(completed[3])
        self._update_mode_controls()

    def _finish_calibration_optimization(self, message: str, indices: np.ndarray) -> str:
        self._mark_calibration_unapplied()
        if tuple(indices.tolist()) != tuple(self.session.calibration_indices):
            return message
        pcs = self.session.current_pc_custom[indices].copy()
        mean = np.mean(pcs, axis=0)
        std = np.std(pcs, axis=0, ddof=1 if len(pcs) > 1 else 0)
        convention = self.session.data.pc_output_convention
        statistics = (
            f"Last completed optimization ({convention}; x, y, z)\n"
            f"Mean PC: ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f})\n"
            f"PC std: ({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})"
        )
        self._completed_calibration_report = (
            self.session, tuple(indices.tolist()), statistics, self.session.calibration_point_summary(),
        )
        return f"{message} Next: use ‘2. Apply average PC to map’ before continuing to other tabs."

    @_guarded_action
    def _add_calibration_point(self) -> None:
        try:
            previous = tuple(self.session.calibration_indices)
            msg = self.session.add_calibration_point(int(self.index_var.get()))
            if self._calibration_apply_state != "none" and previous != tuple(self.session.calibration_indices):
                self._mark_calibration_unapplied()
            self.status_var.set(msg)
            self._log(msg)
            self._update_calibration_summary()
            self._refresh_plot()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    @_guarded_action
    def _remove_calibration_point(self) -> None:
        try:
            previous = tuple(self.session.calibration_indices)
            msg = self.session.remove_calibration_point(int(self.index_var.get()))
            if self._calibration_apply_state != "none" and previous != tuple(self.session.calibration_indices):
                self._mark_calibration_unapplied()
            self.status_var.set(msg)
            self._log(msg)
            self._update_calibration_summary()
            self._refresh_plot()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    @_guarded_action
    def _clear_calibration_points(self) -> None:
        msg = self.session.clear_calibration_points()
        self.status_var.set(msg)
        self._log(msg)
        self._update_calibration_summary()
        self._refresh_plot()

    @_guarded_action
    def _refine_selected_point(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        indices = np.array([int(self.index_var.get())], dtype=np.int64)
        if indices.size == 0:
            raise ValueError("Add calibration points from the map first.")
        phase_id = int(self.phase_id_var.get())
        trust_euler = float(self.calibration_trust_euler_var.get())
        trust_pc = float(self.trust_pc_var.get())
        maxfev = int(self.calibration_maxfev_var.get())
        if not np.isfinite(trust_euler) or trust_euler <= 0 or not np.isfinite(trust_pc) or trust_pc <= 0 or maxfev < 1:
            raise ValueError("Calibration trust regions and maximum evaluations must be positive.")
        self._completed_calibration_report = None
        self._update_calibration_summary()
        self._mark_calibration_unapplied()
        self._run_threaded(lambda: self.session.refine_indices(
            indices=indices, phase_id=phase_id, trust_euler_deg=trust_euler,
            trust_pc=trust_pc, maxfev=maxfev, progress_callback=self._calibration_progress,
            parallel_cores=indexing_cores,
        ), on_success=lambda msg: self._finish_calibration_optimization(msg, indices))

    @_guarded_action
    def _refine_calibration_points(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        indices = np.asarray(self.session.calibration_indices, dtype=np.int64).copy()
        if indices.size == 0:
            raise ValueError("Add calibration points from the map first.")
        phase_id = int(self.phase_id_var.get())
        if not getattr(self.session, "phase_masters", {}) and indices.size > 1 and np.any(self.session.current_phases[indices] != phase_id):
            raise ValueError(f"Select calibration points from phase {phase_id} only before optimizing.")
        trust_euler = float(self.calibration_trust_euler_var.get())
        trust_pc = float(self.trust_pc_var.get())
        maxfev = int(self.calibration_maxfev_var.get())
        if not np.isfinite(trust_euler) or trust_euler <= 0 or not np.isfinite(trust_pc) or trust_pc <= 0 or maxfev < 1:
            raise ValueError("Calibration trust regions and maximum evaluations must be positive.")
        self._completed_calibration_report = None
        self._update_calibration_summary()
        self._mark_calibration_unapplied()
        self._run_threaded(lambda: self.session.refine_indices(
            indices=indices, phase_id=phase_id, trust_euler_deg=trust_euler,
            trust_pc=trust_pc, maxfev=maxfev, progress_callback=self._calibration_progress,
            parallel_cores=indexing_cores,
        ), on_success=lambda msg: self._finish_calibration_optimization(msg, indices))

    @_guarded_action
    def _apply_average_calibration_pc(self) -> None:
        settings = self._calibration_application_settings()
        enabled, pitch = settings
        self._mark_calibration_unapplied()
        self._run_threaded(lambda: self.session.apply_average_calibration_pc(
            use_scan_geometry=enabled, effective_detector_px_size_um=pitch,
        ), on_success=lambda msg: self._finish_calibration_application(msg, settings))

    @_guarded_action
    def _refine_roi(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        indices = self.session.roi_indices(*self._roi_bounds())
        if indices.size == 0:
            raise ValueError("Add calibration points from the map first.")
        phase_id = int(self.phase_id_var.get())
        trust_euler = float(self.calibration_trust_euler_var.get())
        trust_pc = float(self.trust_pc_var.get())
        maxfev = int(self.calibration_maxfev_var.get())
        if not np.isfinite(trust_euler) or trust_euler <= 0 or not np.isfinite(trust_pc) or trust_pc <= 0 or maxfev < 1:
            raise ValueError("Calibration trust regions and maximum evaluations must be positive.")
        self._mark_calibration_unapplied()
        self._run_threaded(lambda: self.session.refine_indices(
            indices=indices, phase_id=phase_id, trust_euler_deg=trust_euler,
            trust_pc=trust_pc, maxfev=maxfev, progress_callback=self._calibration_progress,
            parallel_cores=indexing_cores,
        ))

    @_guarded_action
    def _apply_selected_pc_to_full_map(self) -> None:
        idx = int(self.index_var.get())
        self._mark_calibration_unapplied()
        self._run_threaded(lambda: self.session.apply_point_pc_to_full_map(idx))

    # -------------------------- Step 2 -------------------------- #

    def _index_refinement_settings(self) -> tuple[float, int, bool]:
        cache = self.session.dictionary_cache
        if bool(self.follow_dictionary_trust_var.get()):
            trust = float(cache.resolution_deg if cache is not None else self.di_res_deg_var.get())
        else:
            trust = float(self.trust_euler_var.get())
        maxfev = int(self.maxfev_var.get())
        if not np.isfinite(trust) or trust <= 0 or maxfev < 1:
            raise ValueError("Orientation trust region and maximum evaluations must be positive.")
        return trust, maxfev, bool(self.refine_full_resolution_var.get())

    def _index_primary_indices(self, indices: np.ndarray, *, label: str) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        phase_id = int(self.phase_id_var.get())
        resolution_deg = float(self.di_res_deg_var.get())
        keep_n = int(self.dictionary_keep_n_var.get())
        auto_refine = bool(self.auto_refine_var.get())
        refinement = self._index_refinement_settings() if auto_refine else None
        if keep_n < 1 or not np.isfinite(resolution_deg) or resolution_deg <= 0:
            raise ValueError("Keep N and dictionary spacing must be positive.")
        self._set_reindex_progress(0.0, f"Starting {label} indexing...")
        def progress(value, message):
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_reindex_progress(v, m))
        def refinement_progress(value, message):
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_refinement_progress(v, m))
        def action():
            message = self.session.dictionary_index_indices(
                indices=indices, phase_id=phase_id, keep_n=keep_n,
                resolution_deg=resolution_deg, progress_callback=progress,
                parallel_cores=indexing_cores,
            )
            MultiStepOverlapGUI._autosave_stage(self, "primary indexing")
            if refinement is not None:
                self._check_job_cancelled()
                trust, maxfev, full_resolution = refinement
                refined = self.session.refine_orientations_indices(
                    indices, phase_id=phase_id, trust_euler_deg=trust, maxfev=maxfev,
                    use_full_resolution=full_resolution, progress_callback=refinement_progress,
                    parallel_cores=indexing_cores,
                )
                message = f"{message} {refined}"
            return message
        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _generate_dictionary(self) -> None:
        self._set_dictionary_progress(0.0, "Starting dictionary generation...")
        phase_id = int(self.phase_id_var.get())
        resolution_deg = float(self.di_res_deg_var.get())
        software_binning = int(self.di_binning_var.get())
        self._refresh_default_dictionary_path()

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_dictionary_progress(v, m))

        def action() -> str:
            return self.session.generate_dictionary(
                phase_id=phase_id,
                resolution_deg=resolution_deg,
                software_binning=software_binning,
                progress_callback=progress,
            )

        self._run_threaded(action)

    @_guarded_action
    def _save_dictionary(self) -> None:
        path = self._choose_dictionary_save_path()
        if not path:
            return
        self._run_threaded(lambda path=path: self.session.save_dictionary(path))

    @_guarded_action
    def _load_dictionary(self) -> None:
        selected = filedialog.askopenfilename(**self._dictionary_dialog_options())
        if not selected:
            return
        path = str(Path(selected).resolve())
        self.dictionary_path_var.set(path)
        self._auto_dictionary_path = None

        def action() -> str:
            msg = self.session.load_dictionary(path)
            cache = self.session.dictionary_cache
            if cache is not None:
                self._post_ui(lambda: (
                        self.phase_id_var.set(cache.phase_id),
                        self.di_res_deg_var.set(cache.resolution_deg),
                        self.di_binning_var.set(cache.software_binning),
                        self._sync_master_energy_mode_from_session(),
                    ),
                )
            return msg

        self._run_threaded(action)

    @_guarded_action
    def _index_selected_point(self) -> None:
        self._index_primary_indices(np.asarray([int(self.index_var.get())], dtype=np.int64), label="Selected-point")

    @_guarded_action
    def _index_roi(self) -> None:
        if self.session.data is None:
            raise ValueError("Load input data first.")
        self._index_primary_indices(self.session.roi_indices(*self._roi_bounds()), label="ROI")

    @_guarded_action
    def _refine_last_indexed(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        phase_id = int(self.phase_id_var.get())
        trust_euler, maxfev, use_full_resolution = self._index_refinement_settings()
        indices = self.session.last_indexed_indices
        if indices is None or indices.size == 0:
            raise ValueError("Run dictionary indexing on a point or ROI first.")
        indices = indices.copy()
        self._set_refinement_progress(0.0, "Starting orientation refinement...")
        def progress(value, message):
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_refinement_progress(v, m))
        self._run_threaded(lambda: self.session.refine_orientations_indices(
            indices, phase_id=phase_id, trust_euler_deg=trust_euler, maxfev=maxfev,
            use_full_resolution=use_full_resolution, progress_callback=progress,
            parallel_cores=indexing_cores,
        ), autosave=True)

    @_guarded_action
    def _run_complete_roi_analysis(self, *, include_step4: bool = True) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        workflow_label = "Steps 2–4" if include_step4 else "Steps 2–3"
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        if self.session.dictionary_cache is None:
            messagebox.showerror("Error", "Generate or load a dictionary before running the automated analysis.")
            return

        bounds = self._roi_bounds()
        roi_indices = self.session.roi_indices(*bounds)
        selected_index = int(self.index_var.get())
        try:
            phase_id = int(self.phase_id_var.get())
            resolution_deg = float(self.di_res_deg_var.get())
            keep_n = max(1, int(self.dictionary_keep_n_var.get()))
            auto_refine = bool(self.auto_refine_var.get())
            if auto_refine:
                primary_trust_euler, primary_maxfev, primary_full_resolution = self._index_refinement_settings()
            fit_blur_gain = bool(self.fit_blur_gain_var.get())
            blur_sigma = float(self.blur_sigma_var.get())
            if not np.isfinite(blur_sigma) or blur_sigma < 0:
                raise ValueError("Manual blur sigma must be finite and non-negative.")
            fit_method = _selected_fit_method(self)
            fit_maxiter = int(self.gain_fit_maxiter_var.get())
            fit_popsize = int(self.gain_fit_popsize_var.get())
            fit_bounds = self._primary_fit_bounds() if fit_blur_gain or include_step4 else None
            if auto_refine:
                residual_trust_euler, residual_maxfev, residual_full_resolution = (
                    primary_trust_euler, primary_maxfev, primary_full_resolution
                )
            step3_parallel_cores = int(self.step3_parallel_cores_var.get())
            step4_parallel_cores = int(self.step4_parallel_cores_var.get()) if include_step4 else 0
            primary_ncc_threshold = float(self._residual_ncc_threshold())
            residual_ncc_threshold = (
                float(self._overlap_mixture_residual_ncc_threshold()) if include_step4 else 0.0
            )
            write_patterns = bool(self.write_residual_patterns_var.get())
            residual_output_path = self.residual_pattern_path_var.get().strip()
        except Exception as exc:
            messagebox.showerror(f"Invalid {workflow_label} analysis settings", str(exc))
            return
        if write_patterns and not residual_output_path:
            messagebox.showerror("Missing residual output", "Choose a residual-pattern file name first.")
            return

        self._sync_residual_keep_n_to_dictionary(keep_n)
        roi_description = (
            f"{bounds[2]} × {bounds[3]} from row {bounds[0]}, col {bounds[1]} "
            f"({roi_indices.size} point(s))"
        )
        self._set_complete_analysis_progress(0.0, f"Preparing {workflow_label}: ROI {roi_description}...")
        self._log(f"Starting {workflow_label}: ROI {roi_description}.")

        stage_count = 6 if include_step4 else 5

        def stage_progress(stage_index: int, stage_name: str, setter):
            def callback(value: float, message: str) -> None:
                self._check_job_cancelled()
                stage_value = float(np.clip(value, 0.0, 100.0))
                overall = 100.0 * (stage_index + stage_value / 100.0) / stage_count

                def update() -> None:
                    self._set_complete_analysis_progress(
                        overall,
                        f"{stage_index + 1}/{stage_count} {stage_name}: {message}",
                    )
                    setter(stage_value, message)

                self._post_ui(update)

            return callback

        def finish_stage(
            stage_index: int,
            stage_name: str,
            message: str,
            *,
            map_view_index: int,
            skipped: bool = False,
        ) -> None:
            if not skipped:
                MultiStepOverlapGUI._autosave_stage(self, stage_name)
            self._check_job_cancelled()
            overall = 100.0 * (stage_index + 1) / stage_count

            def update() -> None:
                self._set_complete_analysis_progress(
                    overall,
                    f"{stage_index + 1}/{stage_count} {stage_name} "
                    + ("skipped (auto-refine off)." if skipped else "complete."),
                )
                self._log(f"ROI analysis {workflow_label} — {stage_name}: {message}")
                self._job_result_views.add(map_view_index)

            # Stage transitions use the same cost-aware preview gate as all
            # other progress boundaries, avoiding duplicate forced redraws.
            self._post_ui(update)

        def execute() -> str:
            primary_index_msg = self.session.dictionary_index_indices(
                indices=roi_indices,
                phase_id=phase_id,
                keep_n=keep_n,
                resolution_deg=resolution_deg,
                progress_callback=stage_progress(0, "Primary indexing", self._set_reindex_progress),
                parallel_cores=indexing_cores,
            )
            self.last_overlap = None
            self.last_overlap_mixture = None
            finish_stage(0, "Primary indexing", primary_index_msg, map_view_index=1)

            if auto_refine:
                primary_refine_msg = self.session.refine_orientations_indices(
                    roi_indices,
                    phase_id=phase_id,
                    trust_euler_deg=primary_trust_euler,
                    maxfev=primary_maxfev,
                    use_full_resolution=primary_full_resolution,
                    progress_callback=stage_progress(1, "Primary refinement", self._set_refinement_progress),
                    parallel_cores=indexing_cores,
                )
            else:
                primary_refine_msg = "Skipped because automatic orientation refinement is off."
            finish_stage(1, "Primary refinement", primary_refine_msg, map_view_index=1, skipped=not auto_refine)

            eligible_primary: list[int] = []
            for idx in roi_indices.tolist():
                score = self.session.get_primary_index_ncc(int(idx))
                if score is not None and (primary_ncc_threshold <= 0.0 or score >= primary_ncc_threshold):
                    eligible_primary.append(int(idx))
            residual_indices = np.asarray(eligible_primary, dtype=np.int64)
            skipped_primary = int(roi_indices.size - residual_indices.size)
            if residual_indices.size == 0:
                raise RuntimeError(
                    "Primary indexing finished, but no ROI points meet the minimum primary NCC "
                    f"of {primary_ncc_threshold:.3f}."
                )

            residual_generation_msg = self.session.compute_overlap_residual_indices(
                residual_indices,
                fit_blur_gain=fit_blur_gain,
                blur_sigma=blur_sigma,
                fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds if fit_blur_gain else None,
                write_patterns=write_patterns,
                residual_output_path=residual_output_path if write_patterns else None,
                parallel_cores=step3_parallel_cores,
                selected_index=(
                    selected_index if np.any(residual_indices == selected_index) else None
                ),
                progress_callback=stage_progress(2, "Residual generation", self._set_overlap_progress),
            )
            if np.any(residual_indices == selected_index):
                self.last_overlap = self.session.get_residual_point_result(selected_index)
            finish_stage(2, "Residual generation", residual_generation_msg, map_view_index=2)

            residual_index_msg = self.session.index_overlap_residual_indices(
                residual_indices,
                keep_n=keep_n,
                write_patterns=write_patterns,
                selected_index=(
                    selected_index if np.any(residual_indices == selected_index) else None
                ),
                progress_callback=stage_progress(3, "Residual indexing", self._set_overlap_progress),
                parallel_cores=indexing_cores,
            )
            if np.any(residual_indices == selected_index):
                self.last_overlap = self.session.get_residual_point_result(selected_index)
            finish_stage(3, "Residual indexing", residual_index_msg, map_view_index=2)

            if auto_refine:
                residual_refine_msg = self.session.refine_overlap_residual_indices(
                    residual_indices,
                    trust_euler_deg=residual_trust_euler,
                    maxfev=residual_maxfev,
                    use_full_resolution=residual_full_resolution,
                    write_patterns=write_patterns,
                    selected_index=(
                        selected_index if np.any(residual_indices == selected_index) else None
                    ),
                    progress_callback=stage_progress(4, "Residual refinement", self._set_overlap_progress),
                    parallel_cores=indexing_cores,
                )
                if np.any(residual_indices == selected_index):
                    self.last_overlap = self.session.get_residual_point_result(selected_index)
            else:
                residual_refine_msg = "Skipped because automatic orientation refinement is off."
            finish_stage(4, "Residual refinement", residual_refine_msg, map_view_index=2, skipped=not auto_refine)

            if not include_step4:
                self.last_overlap_mixture = None
                self._post_ui(lambda: self._set_complete_analysis_progress(
                        100.0,
                        "Steps 2–3 ROI analysis finished successfully.",
                    ),
                )
                return (
                    f"Steps 2–3 ROI analysis finished for {roi_indices.size} point(s): re-indexed "
                    f"the primary orientations, then generated and indexed {residual_indices.size} residual(s). "
                    + ("Primary and residual orientations refined. " if auto_refine else "Orientation refinement skipped (auto-refine off). ")
                    + f"Skipped {skipped_primary} point(s) at the primary NCC threshold. "
                    "Step 4 mixture fitting was not run."
                )

            mixture_indices: list[int] = []
            for idx in residual_indices.tolist():
                score = self._overlap_mixture_residual_ncc_for_index(int(idx))
                if residual_ncc_threshold <= 0.0 or (
                    score is not None and score >= residual_ncc_threshold
                ):
                    mixture_indices.append(int(idx))
            overlap_indices = np.asarray(mixture_indices, dtype=np.int64)
            skipped_residual = int(residual_indices.size - overlap_indices.size)
            if overlap_indices.size == 0:
                raise RuntimeError(
                    "Residual processing finished, but no ROI points meet the minimum residual NCC "
                    f"of {residual_ncc_threshold:.3f}."
                )

            mixture_msg = self.session.compute_overlap_mixture_indices(
                overlap_indices,
                fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
                parallel_cores=step4_parallel_cores,
                selected_index=(
                    selected_index if np.any(overlap_indices == selected_index) else None
                ),
                progress_callback=stage_progress(5, "Overlap optimization", self._set_overlap_optimization_progress),
            )
            if np.any(overlap_indices == selected_index):
                self.last_overlap_mixture = self.session.get_overlap_mixture_result(selected_index)
            finish_stage(5, "Overlap optimization", mixture_msg, map_view_index=3)

            if np.any(residual_indices == selected_index):
                selected_residual = self.session.get_residual_point_result(selected_index)
                self.last_overlap = selected_residual
            else:
                self.last_overlap = None
            if np.any(overlap_indices == selected_index):
                self.last_overlap_mixture = self.session.get_overlap_mixture_result(selected_index)
            else:
                self.last_overlap_mixture = None

            self._post_ui(lambda: self._set_complete_analysis_progress(
                    100.0,
                    "Complete ROI analysis finished successfully.",
                ),
            )
            return (
                f"Complete ROI analysis finished for {roi_indices.size} point(s): re-indexed "
                f"the primary orientations, processed {residual_indices.size} residual(s), and optimized "
                f"{overlap_indices.size} overlap mixture(s). "
                + ("Primary and residual orientations refined. " if auto_refine else "Orientation refinement skipped (auto-refine off). ")
                + f"Skipped {skipped_primary} point(s) at the primary NCC threshold and "
                f"{skipped_residual} at the residual NCC threshold."
            )

        def action() -> str:
            try:
                return execute()
            except Exception as exc:
                error_message = str(exc)
                self._post_ui(lambda error_message=error_message: self.complete_analysis_status_var.set(
                        f"ROI analysis {workflow_label} stopped: {error_message}"
                    ),
                )
                raise

        self._run_threaded(action, autosave=True)

    # -------------------------- Step 3 -------------------------- #

    @_guarded_action
    def _run_residual_roi_analysis(self, *, include_step4: bool = False) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        if self.session.data is None or self.session.dictionary_cache is None:
            raise ValueError("Load input and a dictionary before analyzing residuals.")
        indices, skipped = self._filter_roi_indices_by_threshold(self.session.roi_indices(*self._roi_bounds()))
        if indices.size == 0:
            raise ValueError("No indexed ROI points meet the minimum primary NCC.")
        selected_index = int(self.index_var.get())
        selected_index = selected_index if np.any(indices == selected_index) else None
        keep_n = int(self.dictionary_keep_n_var.get())
        auto_refine = bool(self.auto_refine_var.get())
        refinement = self._index_refinement_settings() if auto_refine else None
        fit_blur_gain = bool(self.fit_blur_gain_var.get())
        blur_sigma = float(self.blur_sigma_var.get())
        if not np.isfinite(blur_sigma) or blur_sigma < 0:
            raise ValueError("Manual blur sigma must be finite and non-negative.")
        fit_method = _selected_fit_method(self)
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        fit_bounds = self._primary_fit_bounds() if fit_blur_gain or include_step4 else None
        parallel_cores = int(self.parallel_cores_var.get())
        write_patterns = bool(self.write_residual_patterns_var.get())
        output = self.residual_pattern_path_var.get().strip()
        if keep_n < 1 or fit_maxiter < 1 or fit_popsize < 1 or parallel_cores < 1:
            raise ValueError("Keep N, optimizer limits and parallel cores must be positive.")
        if write_patterns and not output:
            raise ValueError("Choose a residual-pattern output path first.")
        mixture_threshold = float(self._overlap_mixture_residual_ncc_threshold()) if include_step4 else 0.0
        mixture_cores = int(self.step4_parallel_cores_var.get()) if include_step4 else parallel_cores
        if mixture_cores < 1:
            raise ValueError("Mixture worker cores must be positive.")
        stages = (3 if auto_refine else 2) + int(include_step4)
        def progress(stage):
            def update(value, message):
                self._check_job_cancelled()
                overall = (stage * 100.0 + float(value)) / stages
                self._post_ui(lambda v=overall, m=message: self._set_overlap_progress(v, m))
                if include_step4 and stage == stages - 1:
                    self._post_ui(lambda v=float(value), m=message: self._set_overlap_optimization_progress(v, m))
            return update
        def action():
            messages = [self.session.compute_overlap_residual_indices(
                indices, fit_blur_gain=fit_blur_gain, blur_sigma=blur_sigma, fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize, fit_bounds=fit_bounds if fit_blur_gain else None, parallel_cores=parallel_cores,
                write_patterns=write_patterns, residual_output_path=output if write_patterns else None,
                selected_index=selected_index, progress_callback=progress(0),
            )]
            self._check_job_cancelled()
            messages.append(self.session.index_overlap_residual_indices(
                indices, keep_n=keep_n, write_patterns=write_patterns,
                selected_index=selected_index, progress_callback=progress(1),
                parallel_cores=indexing_cores,
            ))
            MultiStepOverlapGUI._autosave_stage(self, "residual indexing")
            if refinement is not None:
                self._check_job_cancelled()
                trust, maxfev, full_resolution = refinement
                messages.append(self.session.refine_overlap_residual_indices(
                    indices, trust_euler_deg=trust, maxfev=maxfev,
                    use_full_resolution=full_resolution, write_patterns=write_patterns,
                    selected_index=selected_index, progress_callback=progress(2),
                    parallel_cores=indexing_cores,
                ))
            if include_step4:
                self._check_job_cancelled()
                MultiStepOverlapGUI._autosave_stage(self, "residual analysis")
                mixture_indices = np.asarray([
                    int(idx) for idx in indices
                    if mixture_threshold <= 0.0 or (
                        (score := self._overlap_mixture_residual_ncc_for_index(int(idx))) is not None
                        and score >= mixture_threshold
                    )
                ], dtype=np.int64)
                if mixture_indices.size == 0:
                    raise RuntimeError("Residual analysis finished, but no ROI points meet the tab 4 residual NCC threshold.")
                messages.append(self.session.compute_overlap_mixture_indices(
                    mixture_indices, fit_maxiter=fit_maxiter, fit_popsize=fit_popsize,
                    fit_bounds=fit_bounds, fit_method=fit_method, parallel_cores=mixture_cores,
                    selected_index=selected_index if selected_index in mixture_indices else None,
                    progress_callback=progress(stages - 1),
                ))
                messages.append(f"Skipped {indices.size - mixture_indices.size} point(s) at the residual NCC filter.")
            return " ".join(messages) + f" Skipped {skipped} point(s) at the primary NCC filter."
        self._set_overlap_progress(0.0, "Starting steps 3–4 ROI analysis..." if include_step4 else "Starting residual ROI analysis...")
        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _analyze_overlap(self) -> None:
        index = int(self.index_var.get())
        blur_sigma = float(self.blur_sigma_var.get())
        fit_blur_gain = bool(self.fit_blur_gain_var.get())
        fit_method = _selected_fit_method(self)
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        try:
            fit_bounds = self._primary_fit_bounds() if fit_blur_gain else None
        except Exception as exc:
            messagebox.showerror("Invalid fit bounds", str(exc))
            return
        threshold = self._residual_ncc_threshold()
        primary_ncc = self._selected_primary_ncc(index)
        self._set_overlap_progress(0.0, "Fitting the selected point residual...")

        def action() -> str:
            result = self.session.analyze_overlap_point(
                index,
                blur_sigma=blur_sigma,
                fit_blur_gain=fit_blur_gain,
                fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
            )
            self.last_overlap = result
            self.last_overlap_mixture = None
            resid_rms = float(np.sqrt(np.mean(np.square(result.residual))))
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Selected-point residual fit complete."))
            threshold_note = ""
            note_ncc = primary_ncc if primary_ncc is not None else float(result.ncc_es)
            if np.isfinite(note_ncc) and threshold > 0.0 and note_ncc < threshold:
                threshold_note = (
                    f" Primary NCC {note_ncc:.4f} is below the residual-work threshold {threshold:.4f}; "
                    "reindexing/refinement will be skipped."
                )
            return (
                f"idx={result.index} (row={result.row}, col={result.col}) "
                f"NCC {result.ncc_unfitted:.4f} → {result.ncc_es:.4f}, fitted σ={result.fitted_sigma:.4f}; "
                f"residual=E-NCC·S′, RMS={resid_rms:.4f}.{threshold_note} {result.fit_message}"
            )

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _index_overlap_residual(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        index = int(self.index_var.get())
        blur_sigma = float(self.blur_sigma_var.get())
        keep_n = max(1, int(self.residual_keep_n_var.get()))
        residual_result = self.session.get_residual_point_result(index)
        threshold = self._residual_ncc_threshold()
        primary_ncc = self._selected_primary_ncc(index)
        if residual_result is None or residual_result.index != index:
            messagebox.showerror(
                "Residual unavailable",
                "Fit the primary pattern and build the residual for this point before indexing it.",
            )
            return
        if primary_ncc is None:
            messagebox.showinfo(
                "Primary point not indexed",
                "Dictionary index the selected primary point in step 2 before indexing its residual.",
            )
            return
        if primary_ncc is not None and threshold > 0.0 and primary_ncc < threshold:
            messagebox.showinfo(
                "Below NCC threshold",
                f"Primary NCC {primary_ncc:.4f} is below the minimum {threshold:.4f}; residual indexing is skipped.",
            )
            return
        self._set_overlap_progress(0.0, "Indexing the selected-point residual...")

        def action() -> str:
            current_result = self.session.get_residual_point_result(index)
            if current_result is None:
                raise ValueError("The selected result is no longer valid. Recompute it using the current settings first.")
            result = self.session.index_overlap_residual(
                index,
                blur_sigma=blur_sigma,
                keep_n=keep_n,
                residual_result=current_result,
                parallel_cores=indexing_cores,
            )
            self.last_overlap = result
            self.last_overlap_mixture = None
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Selected-point residual indexed."))
            return (
                f"Indexed residual at idx={result.index} with the step 2 dictionary: "
                f"KP NCC={result.secondary_ncc_kp:.4f}, full-pattern NCC={result.secondary_ncc_full:.4f}, "
                f"keep_n={keep_n}."
            )

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _refine_overlap_residual(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        index = int(self.index_var.get())
        result = self.session.get_residual_point_result(index)
        if result is None or result.index != index or result.secondary_euler_rad is None:
            messagebox.showerror(
                "Residual match unavailable",
                "Build and index the residual for the selected point before refinement.",
            )
            return
        trust_euler, maxfev, use_full_resolution = self._index_refinement_settings()
        threshold = self._residual_ncc_threshold()
        primary_ncc = self._selected_primary_ncc(index)
        if primary_ncc is None:
            messagebox.showinfo(
                "Primary point not indexed",
                "Dictionary index the selected primary point in step 2 before refining its residual.",
            )
            return
        if primary_ncc is not None and threshold > 0.0 and primary_ncc < threshold:
            messagebox.showinfo(
                "Below NCC threshold",
                f"Primary NCC {primary_ncc:.4f} is below the minimum {threshold:.4f}; residual refinement is skipped.",
            )
            return
        self._set_overlap_progress(0.0, "Refining the selected-point residual match...")

        def action() -> str:
            current_result = self.session.get_residual_point_result(index)
            if current_result is None:
                raise ValueError("The selected result is no longer valid. Recompute it using the current settings first.")
            refined = self.session.refine_overlap_residual(
                current_result,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
                use_full_resolution=use_full_resolution,
                progress_callback=lambda value, message: self._post_ui(lambda v=value, m=message: self._set_overlap_progress(v, m),
                ),
                parallel_cores=indexing_cores,
            )
            self.last_overlap = refined
            self.last_overlap_mixture = None
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Selected-point residual refined."))
            return (
                f"Refined residual orientation at idx={refined.index}: "
                f"KP NCC={refined.secondary_ncc_kp:.4f}, full-pattern NCC={refined.secondary_ncc_full:.4f}. "
                f"{refined.secondary_refinement_note}"
            )

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _compute_overlap_residual_roi(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        indices = self.session.roi_indices(*bounds)
        threshold = self._residual_ncc_threshold()
        indices, skipped = self._filter_roi_indices_by_threshold(indices)
        if indices.size == 0:
            messagebox.showinfo(
                "No eligible primary points",
                "No ROI points are dictionary indexed and above the requested primary NCC threshold.",
            )
            return
        selected_index = int(self.index_var.get())
        fit_blur_gain = bool(self.fit_blur_gain_var.get())
        blur_sigma = float(self.blur_sigma_var.get())
        if not np.isfinite(blur_sigma) or blur_sigma < 0:
            raise ValueError("Manual blur sigma must be finite and non-negative.")
        fit_method = _selected_fit_method(self)
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        parallel_cores = int(self.step3_parallel_cores_var.get())
        write_patterns = bool(self.write_residual_patterns_var.get())
        residual_output_path = self.residual_pattern_path_var.get().strip()
        if write_patterns and not residual_output_path:
            messagebox.showerror("Missing residual output", "Choose a file name for the residual patterns first.")
            return
        try:
            fit_bounds = self._primary_fit_bounds() if fit_blur_gain else None
        except Exception as exc:
            messagebox.showerror("Invalid fit bounds", str(exc))
            return
        self._set_overlap_progress(0.0, f"Computing residuals for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.compute_overlap_residual_indices(
                indices,
                fit_blur_gain=fit_blur_gain,
                blur_sigma=blur_sigma,
                fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
                write_patterns=write_patterns,
                residual_output_path=residual_output_path if write_patterns else None,
                parallel_cores=parallel_cores,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
            )
            if np.any(indices == selected_index):
                selected_result = self.session.get_residual_point_result(selected_index)
                if selected_result is not None:
                    self.last_overlap = selected_result
            self.last_overlap_mixture = None
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Residuals computed for the ROI."))
            skipped_note = (
                f" Skipped {skipped} point(s) that were not dictionary indexed or were below NCC {threshold:.3f}."
                if skipped > 0 else ""
            )
            return f"{msg}{skipped_note} ROI bounds r0={bounds[0]}, c0={bounds[1]}, nrows={bounds[2]}, ncols={bounds[3]}."

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _index_overlap_residual_roi(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        if self.session.dictionary_cache is None:
            messagebox.showerror("Error", "Generate or load a dictionary in tab 1 first.")
            return
        bounds = self._roi_bounds()
        indices = self.session.roi_indices(*bounds)
        threshold = self._residual_ncc_threshold()
        indices, skipped = self._filter_roi_indices_by_threshold(indices)
        if indices.size == 0:
            messagebox.showinfo(
                "No eligible primary points",
                "No ROI points are dictionary indexed and above the requested primary NCC threshold.",
            )
            return
        selected_index = int(self.index_var.get())
        write_patterns = bool(self.write_residual_patterns_var.get())
        keep_n = max(1, int(self.residual_keep_n_var.get()))
        self._set_overlap_progress(0.0, f"Indexing residuals for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.index_overlap_residual_indices(
                indices,
                keep_n=keep_n,
                write_patterns=write_patterns,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
                parallel_cores=indexing_cores,
            )
            if np.any(indices == selected_index):
                selected_result = self.session.get_residual_point_result(selected_index)
                if selected_result is not None:
                    self.last_overlap = selected_result
            self.last_overlap_mixture = None
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Residual ROI indexing complete."))
            skipped_note = (
                f" Skipped {skipped} point(s) that were not dictionary indexed or were below NCC {threshold:.3f}."
                if skipped > 0 else ""
            )
            return f"{msg}{skipped_note}"

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _refine_overlap_residual_roi(self) -> None:
        indexing_cores = int(self.parallel_cores_var.get())
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        if self.session.dictionary_cache is None:
            messagebox.showerror("Error", "Generate or load a dictionary in tab 1 first.")
            return
        bounds = self._roi_bounds()
        indices = self.session.roi_indices(*bounds)
        threshold = self._residual_ncc_threshold()
        indices, skipped = self._filter_roi_indices_by_threshold(indices)
        if indices.size == 0:
            messagebox.showinfo(
                "No eligible primary points",
                "No ROI points are dictionary indexed and above the requested primary NCC threshold.",
            )
            return
        selected_index = int(self.index_var.get())
        trust_euler, maxfev, use_full_resolution = self._index_refinement_settings()
        write_patterns = bool(self.write_residual_patterns_var.get())
        self._set_overlap_progress(0.0, f"Refining residuals for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.refine_overlap_residual_indices(
                indices,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
                use_full_resolution=use_full_resolution,
                write_patterns=write_patterns,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
                parallel_cores=indexing_cores,
            )
            if np.any(indices == selected_index):
                selected_result = self.session.get_residual_point_result(selected_index)
                if selected_result is not None:
                    self.last_overlap = selected_result
            self.last_overlap_mixture = None
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Residual ROI refinement complete."))
            skipped_note = (
                f" Skipped {skipped} point(s) that were not dictionary indexed or were below NCC {threshold:.3f}."
                if skipped > 0 else ""
            )
            return f"{msg} trust Euler={trust_euler:g}°, maxfev={maxfev}.{skipped_note}"

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _fit_overlap_mixture(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        index = int(self.index_var.get())
        fit_method = _selected_fit_method(self)
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        residual_result = self.session.get_residual_point_result(index)
        threshold = self._overlap_mixture_residual_ncc_threshold()
        residual_ncc = self._overlap_mixture_residual_ncc_for_index(index)
        if threshold > 0.0 and (residual_ncc is None or residual_ncc < threshold):
            shown = "n/a" if residual_ncc is None else f"{residual_ncc:.4f}"
            messagebox.showinfo(
                "Below residual NCC threshold",
                f"Residual NCC {shown} is below the minimum {threshold:.4f}; mixture fitting is skipped.",
            )
            self._refresh_plot()
            return
        try:
            fit_bounds = self._primary_fit_bounds()
        except Exception as exc:
            messagebox.showerror("Invalid fit bounds", str(exc))
            return
        self._set_overlap_optimization_progress(0.0, "Fitting selected-point overlap mixture...")

        def action() -> str:
            current_result = self.session.get_residual_point_result(index)
            if current_result is None:
                raise ValueError("The selected result is no longer valid. Recompute it using the current settings first.")
            result = self.session.fit_overlap_mixture_point(
                index,
                residual_result=current_result,
                fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
            )
            self.last_overlap_mixture = result
            self._post_ui(lambda: self._set_overlap_optimization_progress(100.0, "Selected-point mixture fit complete."))
            return (
                f"idx={result.index} overlap mixture: primary={result.primary_fraction:.3f}, "
                f"secondary={result.secondary_fraction:.3f}, NCC={result.ncc_mixture:.4f}, "
                f"RMS={result.residual_rms:.4f}."
            )

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _refine_overlap_mixture_orientations(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        index = int(self.index_var.get())
        result = self.session.get_overlap_mixture_result(index)
        if result is None:
            messagebox.showinfo(
                "Mixture unavailable",
                "Fit the selected point mixture before refining its orientations.",
            )
            return
        threshold = self._overlap_mixture_residual_ncc_threshold()
        residual_ncc = self._overlap_mixture_residual_ncc_for_index(index)
        if threshold > 0.0 and (residual_ncc is None or residual_ncc < threshold):
            shown = "n/a" if residual_ncc is None else f"{residual_ncc:.4f}"
            messagebox.showinfo(
                "Below residual NCC threshold",
                f"Residual NCC {shown} is below the minimum {threshold:.4f}; orientation refinement is skipped.",
            )
            self._refresh_plot()
            return

        trust_euler = float(self.overlap_mixture_trust_euler_var.get())
        maxfev = int(self.overlap_mixture_maxfev_var.get())
        self._set_overlap_optimization_progress(0.0, "Refining selected mixture orientations...")

        def action() -> str:
            current_result = self.session.get_overlap_mixture_result(index)
            if current_result is None:
                raise ValueError("The selected result is no longer valid. Recompute it using the current settings first.")
            refined = self.session.refine_overlap_mixture_orientations(
                current_result,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
            )
            self.last_overlap_mixture = refined
            if refined.orientation_refined:
                self.last_overlap = None
            self._post_ui(lambda: self._set_overlap_optimization_progress(100.0, "Selected mixture orientation refinement complete."))
            initial = refined.initial_mixture_ncc if refined.initial_mixture_ncc is not None else float("nan")
            return (
                f"Refined mixture orientations at idx={refined.index}: NCC {initial:.4f} -> "
                f"{refined.ncc_mixture:.4f}, primary={refined.primary_fraction:.3f}, "
                f"residual={refined.secondary_fraction:.3f}. {refined.orientation_refinement_note}"
            )

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _fit_overlap_mixture_roi(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        indices = self.session.roi_indices(*bounds)
        threshold = self._overlap_mixture_residual_ncc_threshold()
        indices, skipped_low_residual = self._filter_overlap_mixture_indices_by_residual_threshold(indices)
        if indices.size == 0:
            messagebox.showinfo(
                "Below residual NCC threshold",
                f"No ROI points meet the minimum residual NCC of {threshold:.3f} for overlap optimization.",
            )
            self._refresh_plot()
            return
        selected_index = int(self.index_var.get())
        fit_method = _selected_fit_method(self)
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        parallel_cores = int(self.step4_parallel_cores_var.get())
        try:
            fit_bounds = self._primary_fit_bounds()
        except Exception as exc:
            messagebox.showerror("Invalid fit bounds", str(exc))
            return
        self._set_overlap_optimization_progress(0.0, f"Fitting overlap mixtures for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_optimization_progress(v, m))

        def action() -> str:
            msg = self.session.compute_overlap_mixture_indices(
                indices,
                fit_maxiter=fit_maxiter,
                fit_method=fit_method,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
                parallel_cores=parallel_cores,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
            )
            selected_result = self.session.get_overlap_mixture_result(selected_index)
            if selected_result is not None:
                self.last_overlap_mixture = selected_result
            self._post_ui(lambda: self._set_overlap_optimization_progress(100.0, "Overlap mixture ROI fit complete."))
            threshold_note = (
                f" Skipped {skipped_low_residual} point(s) below residual NCC {threshold:.3f}."
                if threshold > 0.0 and skipped_low_residual > 0
                else ""
            )
            return f"{msg}{threshold_note} ROI bounds r0={bounds[0]}, c0={bounds[1]}, nrows={bounds[2]}, ncols={bounds[3]}."

        self._run_threaded(action, autosave=True)

    @_guarded_action
    def _export_overlap_optimization_results(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        try:
            settings = {
                "fit_method": _selected_fit_method(self),
                "fit_max_iterations": int(self.gain_fit_maxiter_var.get()),
                "fit_population_size": int(self.gain_fit_popsize_var.get()),
                "minimum_primary_ncc": float(self._residual_ncc_threshold()),
                "minimum_residual_ncc": float(self._overlap_mixture_residual_ncc_threshold()),
                "orientation_trust_deg": float(self.overlap_mixture_trust_euler_var.get()),
                "orientation_max_evaluations": int(self.overlap_mixture_maxfev_var.get()),
                "parallel_worker_cores": int(self.step4_parallel_cores_var.get()),
                "primary_fit_bounds": self._primary_fit_bounds(),
            }
        except Exception as exc:
            messagebox.showerror("Invalid export settings", str(exc))
            return
        output_path = self._browse_overlap_optimization_export()
        if output_path is None:
            return
        self._set_overlap_optimization_progress(0.0, "Exporting full-map Step 4 HDF5 results...")

        def action() -> str:
            msg = self.session.export_overlap_optimization_results(
                output_path,
                bounds,
                settings=settings,
            )
            self._post_ui(lambda: self._set_overlap_optimization_progress(100.0, "Step 4 HDF5 export complete."),
            )
            return msg

        self._run_threaded(action)

    @_guarded_action
    def _export_fitted_primary_patterns(self) -> None:
        self._export_fitted_component_patterns(secondary=False)

    @_guarded_action
    def _export_fitted_residual_patterns(self) -> None:
        self._export_fitted_component_patterns(secondary=True)

    def _export_fitted_component_patterns(self, *, secondary: bool) -> None:
        from .fitted_export import export_fitted_patterns
        if self.session.data is None or not self.session.overlap_mixture_results:
            messagebox.showinfo("No fitted patterns", "Complete mixture fitting in tab 4 before exporting.")
            return
        component = "residual" if secondary else "primary"
        source = Path(self.session.data.pattern_path)
        path = filedialog.asksaveasfilename(
            title=f"Export fitted {component} solutions and patterns — H5OINA (full map)",
            initialdir=str(source.parent),
            initialfile=f"{self._source_stem(source)}_fitted_{component}.h5oina",
            defaultextension=".h5oina", filetypes=[("H5OINA", "*.h5oina")],
        )
        if not path:
            return
        def progress(value, message):
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_optimization_progress(v, m))
        self._run_threaded(lambda: export_fitted_patterns(self.session, path, secondary=secondary,
            accepted_only=True, primary_subtract_residual=True, progress_callback=progress))

    @_guarded_action
    def _export_primary_roi_map(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        include_patterns = bool(self.include_primary_patterns_export_var.get())
        output_path = self._browse_primary_roi_export()
        if output_path is None:
            return
        self._set_overlap_progress(0.0, "Exporting primary ROI map...")

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.export_primary_roi_results(
                bounds,
                output_path,
                include_primary_patterns=include_patterns,
                progress_callback=progress,
            )
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Primary ROI export complete."))
            return msg

        self._run_threaded(action)

    @_guarded_action
    def _export_residual_roi_map(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        threshold = self._residual_ncc_threshold()
        include_patterns = bool(self.include_residual_patterns_export_var.get())
        output_path = self._browse_residual_roi_export()
        if output_path is None:
            return
        self._set_overlap_progress(0.0, "Exporting residual ROI map...")

        def progress(value: float, message: str) -> None:
            self._check_job_cancelled()
            self._post_ui(lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.export_residual_roi_results(
                bounds,
                output_path,
                primary_ncc_threshold=threshold,
                include_residual_patterns=include_patterns,
                progress_callback=progress,
            )
            self._post_ui(lambda: self._set_overlap_progress(100.0, "Residual ROI export complete."))
            return msg

        self._run_threaded(action)

    @_guarded_action
    def _show_current_overlap_inspection(self) -> None:
        index = int(self.index_var.get())
        result = self.session.get_residual_point_result(index)
        if result is None:
            messagebox.showinfo("No inspection available", "Fit and index the selected residual point first.")
            return
        self._show_overlap_inspection(result)

    def _show_overlap_inspection(self, result: OverlapPointResult) -> None:
        if self.overlap_inspection_window is None or not self.overlap_inspection_window.winfo_exists():
            window = tk.Toplevel(self)
            window.title("Residual Pattern Inspection")
            window.geometry("1200x850")
            figure = Figure(figsize=(11, 7.5), dpi=100)
            axes = figure.subplots(2, 2)
            canvas = FigureCanvasTkAgg(figure, master=window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill=tk.X)
            self.overlap_inspection_window = window
            self.overlap_inspection_figure = figure
            self.overlap_inspection_axes = axes
            self.overlap_inspection_canvas = canvas
            window.protocol("WM_DELETE_WINDOW", self._close_overlap_inspection)
        else:
            self.overlap_inspection_window.deiconify()
            self.overlap_inspection_window.lift()

        axes = self.overlap_inspection_axes
        figure = self.overlap_inspection_figure
        canvas = self.overlap_inspection_canvas
        if axes is None or figure is None or canvas is None:
            return
        for axis in axes.flat:
            axis.clear()
            axis.set_axis_off()
        axes[0, 0].imshow(result.experimental, cmap="gray")
        self._overlay_pattern_mask(axes[0, 0])
        axes[0, 0].set_title(f"Experimental pattern — idx={result.index} ({result.row}, {result.col})")
        axes[0, 1].imshow(result.simulated, cmap="gray")
        axes[0, 1].set_title(self._simulation_title(f"Primary simulated pattern — NCC={result.ncc_es:.4f}", result.index, result=result))
        rabs = max(float(np.max(np.abs(result.residual))), 1e-8)
        axes[1, 0].imshow(result.residual, cmap=RESIDUAL_PATTERN_CMAP, vmin=-rabs, vmax=rabs)
        axes[1, 0].set_title(f"Residual: Zexp − {result.scale:.4f}·Zsim")
        if result.secondary_simulated is not None:
            axes[1, 1].imshow(result.secondary_simulated, cmap="gray")
            if result.secondary_refined:
                seed = result.secondary_dictionary_ncc_kp
                seed_note = f"; binned dictionary seed={seed:.4f}" if seed is not None else ""
                axes[1, 1].set_title(
                    self._simulation_title(f"Full-resolution refined residual match — KP NCC={result.secondary_ncc_kp:.4f}{seed_note}", result.index, result=result, secondary=True)
                )
            else:
                axes[1, 1].set_title(self._simulation_title(f"Binned dictionary residual match — KP NCC={result.secondary_ncc_kp:.4f}", result.index, result=result, secondary=True))
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Index the residual with the linked phase dictionaries",
                ha="center",
                va="center",
            )
            axes[1, 1].set_title("Residual simulated-pattern match")
        figure.suptitle(result.secondary_refinement_note if result.secondary_refined else "")
        self._apply_plot_font_sizes(figure, axes)
        self._safe_tight_layout(figure)
        canvas.draw_idle()

    def _close_overlap_inspection(self) -> None:
        if self.overlap_inspection_window is not None:
            self.overlap_inspection_window.destroy()
        self.overlap_inspection_window = None
        self.overlap_inspection_figure = None
        self.overlap_inspection_axes = None
        self.overlap_inspection_canvas = None

    # -------------------------- Plotting ------------------------- #

    def _edited_point_overrides(self) -> tuple[np.ndarray, np.ndarray]:
        euler_deg = np.array(
            [
                float(self.euler1_deg_var.get()),
                float(self.euler2_deg_var.get()),
                float(self.euler3_deg_var.get()),
            ],
            dtype=np.float64,
        )
        pc_custom = np.array(
            [
                float(self.pcx_var.get()),
                float(self.pcy_var.get()),
                float(self.pcz_var.get()),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(euler_deg)) or not np.all(np.isfinite(pc_custom)):
            raise ValueError("Non-finite Euler/PC values.")
        return np.deg2rad(euler_deg), pc_custom

    def _apply_plot_font_sizes(self, figure: Figure, axes, *, colorbar=None) -> None:
        axes_arr = np.asarray(axes, dtype=object).ravel()
        for axis in axes_arr:
            if axis is None:
                continue
            axis.title.set_fontsize(PLOT_TITLE_FONTSIZE)
            axis.xaxis.label.set_fontsize(PLOT_TEXT_FONTSIZE)
            axis.yaxis.label.set_fontsize(PLOT_TEXT_FONTSIZE)
            axis.tick_params(axis="both", which="both", labelsize=PLOT_TICK_FONTSIZE)
            for text in axis.texts:
                text.set_fontsize(PLOT_TEXT_FONTSIZE)
        for text in figure.texts:
            text.set_fontsize(PLOT_TEXT_FONTSIZE)
        suptitle = getattr(figure, "_suptitle", None)
        if suptitle is not None:
            suptitle.set_fontsize(PLOT_TITLE_FONTSIZE)
        if colorbar is not None:
            colorbar.ax.tick_params(axis="both", which="both", labelsize=PLOT_TICK_FONTSIZE)
            if colorbar.ax.yaxis.label is not None:
                colorbar.ax.yaxis.label.set_fontsize(PLOT_TEXT_FONTSIZE)

    def _safe_tight_layout(self, figure: Figure) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_TIGHT_LAYOUT_WARNING, category=UserWarning)
            figure.tight_layout()

    def _draw_instruction(self, text: str) -> None:
        for ax in self.axes.flat:
            ax.clear()
            ax.set_axis_off()
        self.figure.texts.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=PLOT_INSTRUCTION_FONTSIZE)
        self._apply_plot_font_sizes(self.figure, self.axes)
        self._safe_tight_layout(self.figure)
        self.canvas.draw_idle()

    def _refresh_complete_analysis_maps(self, stage_view_index: int) -> None:
        active_view = 0
        if self.workflow_notebook is not None:
            active_view = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        # Always redraw the visible map and the tab whose state changed. This
        # keeps intermediate results ready even when that tab is not selected.
        for view_index in dict.fromkeys((active_view, int(stage_view_index))):
            view = self._plot_views[view_index]
            old_axes = self._plot_views[view_index].get("axes", [])
            previous_roi = view.get("roi_bounds")
            limits = {
                i: (ax.get_xlim(), ax.get_ylim())
                for i, ax in enumerate(np.asarray(old_axes, dtype=object).flat)
                if getattr(ax, "_overlap_ebsd_scan_map", False) and ax.images
            }
            self._refresh_plot(view_index=view_index)
            if previous_roi != view.get("roi_bounds"):
                # A hidden tab can still have the previous run's ROI zoom.
                # Preserve manual zoom only while the shared ROI is unchanged.
                limits = {}
            for i, (xlim, ylim) in limits.items():
                axis = np.asarray(self._plot_views[view_index]["axes"], dtype=object).flat[i]
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
            canvas = self._plot_views[view_index].get("canvas")
            if canvas is not None:
                canvas.draw()
        self._activate_plot_view(active_view)

    def _refresh_plot(self, view_index: int | None = None) -> None:
        if view_index is None:
            view_index = 0
            if self.workflow_notebook is not None:
                view_index = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        view_index = int(view_index)
        self._activate_plot_view(view_index)
        if self._roi_drag_patch is not None:
            self._cancel_roi_drag()
        if not self.busy:
            self._sync_pattern_conditioning_for_refresh()
        view = self._plot_views[view_index]
        self._refresh_phase_legend(view)
        if self.session.data is None:
            self._draw_instruction("Load data and master pattern to start.")
            self._set_info_lines(["Load data and master pattern to start."])
            return
        view["roi_bounds"] = self._roi_bounds()
        colorbar = view.get("colorbar")
        if colorbar is not None:
            try:
                colorbar.remove()
            except Exception:
                pass
            view["colorbar"] = None
        self.figure.texts.clear()
        for ax in self.axes.flat:
            ax.clear()
            ax.set_axis_off()
            setattr(ax, "_overlap_ebsd_scan_map", False)

        data = self.session.data
        if data is not None and not getattr(data, "patterns_available", True):
            image = self.session.get_ipf_color_map()
            self.axes.flat[0].imshow(image)
            self.axes.flat[0].set_title("Imported IPF map · no pattern payload")
            self.axes.flat[1].text(.5, .5, "Maps remain viewable.\nLoad matching pattern data to run indexing or fitting.", ha="center", va="center", wrap=True)
            self.canvas.draw_idle()
            return

        row = max(0, min(int(self.row_var.get()), data.rows - 1))
        col = max(0, min(int(self.col_var.get()), data.cols - 1))
        idx = self.session.index_from_row_col(row, col)
        self.last_overlap = self.session.get_residual_point_result(idx)
        self.last_overlap_mixture = self.session.get_overlap_mixture_result(idx)
        self.index_var.set(idx)
        self.row_var.set(row)
        self.col_var.set(col)

        if view_index == 3:
            self._refresh_overlap_optimization_view(row=row, col=col, index=idx)
            return
        if view_index == 2:
            self._refresh_overlap_map_view(row=row, col=col, index=idx)
            return
        if view_index == 1:
            self._refresh_index_selection_map_view(row=row, col=col, index=idx)
            return

        available = self.session.available_layers()
        if available:
            calibration_view = self._plot_views.get(0)
            if calibration_view is not None and calibration_view.get("combo") is not None:
                calibration_view["combo"]["values"] = available
        layer = self.map_layer_var.get()
        if available and layer not in available:
            layer = available[0]
            self.map_layer_var.set(layer)
        m = self.session.get_layer_map(layer)
        ax_map = self.axes[0, 0]
        if m.ndim == 3 and m.shape[-1] == 3:
            ax_map.imshow(np.clip(m, 0.0, 1.0), origin="upper")
        else:
            cmap = "tab20" if layer.lower() == "phase" else "gray"
            vals = m[np.isfinite(m)]
            if vals.size > 0:
                vmin = float(np.percentile(vals, 2))
                vmax = float(np.percentile(vals, 98))
                if vmax <= vmin:
                    vmin = float(vals.min())
                    vmax = float(vals.max()) + 1e-8
            else:
                vmin, vmax = 0.0, 1.0
            ax_map.imshow(m, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        self._draw_inspection_marker(
            ax_map,
            row=row,
            col=col,
            ipf=str(layer).strip().upper().startswith("IPF"),
        )
        if self.session.calibration_indices:
            calibration_rc = np.asarray(
                [self.session.row_col_from_index(i) for i in self.session.calibration_indices],
                dtype=np.float64,
            )
            ax_map.scatter(
                calibration_rc[:, 1],
                calibration_rc[:, 0],
                marker="o",
                facecolors="none",
                edgecolors="yellow",
                s=80,
                linewidths=1.5,
            )
        r0, c0, nrows, ncols = self._roi_bounds()
        ax_map.add_patch(
            Rectangle(
                (c0 - 0.5, r0 - 0.5),
                ncols,
                nrows,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )
        if view_index == 1:
            ax_map.set_title(f"Map: {layer}")
        else:
            ax_map.set_title(f"Map: {layer} (row={row}, col={col})")
        ax_map.set_axis_off()

        exp = self.session._processed_pattern_at(idx)  # Uses lazy slicing unless dynamic BG subtraction is enabled.
        self.axes[0, 1].imshow(normalize_for_view(exp), cmap="gray")
        self._overlay_pattern_mask(self.axes[0, 1])
        self.axes[0, 1].set_title("Experimental")
        self.axes[0, 1].set_axis_off()

        sim_shown = False
        preview_error = "Load a master pattern"
        live_residual = None
        live_ncc: float | None = None
        if self.session.master is not None:
            try:
                overlap_tab_active = (
                    self.workflow_notebook is not None
                    and self.workflow_notebook.index(self.workflow_notebook.select()) == 2
                )
                if overlap_tab_active and self.last_overlap is not None and self.last_overlap.index == idx:
                    sim = self.last_overlap.simulated
                    ncc_es = self.last_overlap.ncc_es
                    live_residual = self.last_overlap.residual
                else:
                    e_override, pc_override = self._edited_point_overrides()
                    sim, ncc_es, live_residual, _scale, _ncc_resid = self.session.preview_simulated_pattern_with_ncc(
                        idx,
                        euler_rad_override=e_override,
                        pc_custom_override=pc_override,
                    )
                self.axes[0, 2].imshow(sim, cmap="gray")
                self.axes[0, 2].set_title(self._simulation_title(f"Primary Simulation | NCC={ncc_es:.4f}", idx))
                self.axes[0, 2].set_axis_off()
                sim_shown = True
                live_ncc = float(ncc_es)
            except Exception as exc:
                sim_shown = False
                preview_error = f"Simulation unavailable: {exc}"

        if not sim_shown:
            self.axes[0, 2].text(0.5, 0.5, preview_error, ha="center", va="center", wrap=True)
            self.axes[0, 2].set_title("Primary Simulation")

        if sim_shown and live_residual is not None:
            rmin = float(np.nanmin(live_residual))
            rmax = float(np.nanmax(live_residual))
            rabs = max(abs(rmin), abs(rmax), 1e-8)
            rrms = float(np.sqrt(np.mean(np.square(live_residual))))
            im_resid = self.axes[1, 0].imshow(
                live_residual,
                cmap=RESIDUAL_PATTERN_CMAP,
                vmin=-rabs,
                vmax=rabs,
            )
            self.axes[1, 0].set_title(f"Residual (Zexp - NCC·Zsim) | RMS={rrms:.4f}")
            self.axes[1, 0].set_axis_off()
            view["colorbar"] = self.figure.colorbar(
                im_resid,
                ax=self.axes[1, 0],
                fraction=0.046,
                pad=0.04,
                ticks=[-rabs, 0.0, rabs],
            )
        elif self.last_overlap is not None and self.last_overlap.index == idx:
            rmin = float(np.nanmin(self.last_overlap.residual))
            rmax = float(np.nanmax(self.last_overlap.residual))
            rabs = max(abs(rmin), abs(rmax), 1e-8)
            im_resid = self.axes[1, 0].imshow(
                self.last_overlap.residual,
                cmap=RESIDUAL_PATTERN_CMAP,
                vmin=-rabs,
                vmax=rabs,
            )
            self.axes[1, 0].set_title("Residual (Zexp - NCC·Zsim)")
            self.axes[1, 0].set_axis_off()
            view["colorbar"] = self.figure.colorbar(
                im_resid,
                ax=self.axes[1, 0],
                fraction=0.046,
                pad=0.04,
                ticks=[-rabs, 0.0, rabs],
            )
        else:
            self.axes[1, 0].text(0.5, 0.5, "Build a residual in step 3", ha="center", va="center")
            self.axes[1, 0].set_title("Primary Residual")

        if (
            self.last_overlap is not None
            and self.last_overlap.index == idx
            and self.last_overlap.secondary_simulated is not None
        ):
            self.axes[1, 1].imshow(self.last_overlap.secondary_simulated, cmap="gray")
            self.axes[1, 1].set_title(self._simulation_title(f"Residual-indexed Simulation | KP NCC={self.last_overlap.secondary_ncc_kp:.4f}", idx, result=self.last_overlap, secondary=True))
            self.axes[1, 1].set_axis_off()
        else:
            self.axes[1, 1].text(0.5, 0.5, "Index residual with step 2 dictionary", ha="center", va="center")
            self.axes[1, 1].set_title("Secondary Simulation")

        if self.session.last_scores_map is not None:
            sm = self.session.last_scores_map
            self.axes[1, 2].imshow(sm, cmap="viridis", origin="upper")
            self._draw_inspection_marker(self.axes[1, 2], row=row, col=col, ipf=False)
            self.axes[1, 2].set_title("Latest KP Score Map")
            self.axes[1, 2].set_axis_off()

        point = self.session.get_point_state(idx)
        e_deg = np.asarray(point["euler_deg"], dtype=np.float64).reshape(3)
        pc = np.asarray(point["pc_custom"], dtype=np.float64).reshape(3)
        phase = int(point["phase"])
        conv = str(point["pc_convention"])
        resid_rms = None
        if live_residual is not None:
            resid_rms = float(np.sqrt(np.mean(np.square(live_residual))))
        info_lines = [
            f"Status: {self.status_var.get()}",
            f"Selected point: idx={idx}, row={row}, col={col}, phase={phase}",
            f"Euler (deg): phi1={e_deg[0]:.2f}, Phi={e_deg[1]:.2f}, phi2={e_deg[2]:.2f}",
            f"PC ({conv}): x={pc[0]:.3f}, y={pc[1]:.3f}, z={pc[2]:.3f}",
            f"NCC(E,S): {live_ncc:.4f}" if live_ncc is not None else "NCC(E,S): n/a",
            f"Residual RMS (Zexp - NCC*Zsim): {resid_rms:.4f}" if resid_rms is not None else "Residual RMS: n/a",
            f"Calibration points: {len(self.session.calibration_indices)}",
        ]
        if self.last_overlap is not None and self.last_overlap.index == idx and self.last_overlap.secondary_ncc_kp is not None:
            info_lines.append(f"Residual dictionary NCC (KP): {self.last_overlap.secondary_ncc_kp:.4f}")
        self._set_info_lines(info_lines)

        self._apply_plot_font_sizes(self.figure, self.axes, colorbar=view.get("colorbar"))
        self._safe_tight_layout(self.figure)
        self.canvas.draw_idle()

    def _refresh_index_selection_map_view(self, *, row: int, col: int, index: int) -> None:
        if self.session.data is None:
            return
        data = self.session.data
        axes = np.asarray(self.axes, dtype=object)
        if axes.shape != (2, 3):
            return

        for ax in axes.flat:
            ax.clear()
            ax.set_axis_off()

        r0, c0, nrows, ncols = self._roi_bounds()
        r1 = min(data.rows - 1, r0 + nrows - 1)
        c1 = min(data.cols - 1, c0 + ncols - 1)
        roi_nrows = max(1, r1 - r0 + 1)
        roi_ncols = max(1, c1 - c0 + 1)
        xlim_full = (-0.5, data.cols - 0.5)
        ylim_full = (data.rows - 0.5, -0.5)
        xlim_roi = (c0 - 0.5, c1 + 0.5)
        ylim_roi = (r1 + 0.5, r0 - 0.5)

        quality_layer = str(self.index_quality_layer_var.get()).strip()
        choices = self._index_quality_layer_choices()
        if quality_layer not in choices:
            quality_layer = self._default_index_quality_layer()
            self.index_quality_layer_var.set(quality_layer)

        quality_full = np.asarray(self.session.get_layer_map(quality_layer), dtype=np.float32).reshape(data.rows, data.cols)
        phase_legend = self.session.phase_map_legend(quality_layer)
        if phase_legend:
            from matplotlib.colors import ListedColormap
            categorical = np.full_like(quality_full, np.nan)
            for n, (pid, _label, _color) in enumerate(phase_legend):
                categorical[quality_full == pid] = n
            quality_full = categorical
        quality_cmap = "tab20" if quality_layer.lower() == "phase" else "viridis"
        quality_vals = quality_full[np.isfinite(quality_full)]
        if quality_vals.size > 0:
            qvmin = float(np.nanpercentile(quality_vals, 2))
            qvmax = float(np.nanpercentile(quality_vals, 98))
            if qvmax <= qvmin:
                qvmax = qvmin + 1e-8
        else:
            qvmin, qvmax = 0.0, 1.0

        if phase_legend:
            quality_cmap = ListedColormap([color for _, _, color in phase_legend])
            qvmin, qvmax = -.5, len(phase_legend) - .5

        ipf_direction, ipf_label = self._selected_solution_map()
        preliminary_ipf = self._solution_map_image(ipf_direction, preliminary=True)
        indexed_mask = self.session.indexed_mask
        indexed = indexed_mask is not None and bool(np.any(indexed_mask))
        updated_ipf = None
        if indexed and self.session.current_eulers_rad is not None:
            current_ipf = self._solution_map_image(ipf_direction)
            updated_ipf = np.ones_like(current_ipf, dtype=np.float32)
            indexed_indices = np.flatnonzero(np.asarray(indexed_mask, dtype=bool).reshape(-1))
            updated_ipf.reshape(-1, 3)[indexed_indices] = current_ipf.reshape(-1, 3)[indexed_indices]
        ncc_map = None
        if indexed and self.session.last_scores_map is not None:
            ncc_map = np.asarray(self.session.last_scores_map, dtype=np.float32).copy()
            ncc_map.reshape(-1)[~np.asarray(indexed_mask, dtype=bool).reshape(-1)] = np.nan

        def _draw_rgb(ax, image: np.ndarray | None, title: str, *, zoom: bool) -> None:
            if image is None:
                ax.text(0.5, 0.5, f"No {ipf_label} available", ha="center", va="center")
            else:
                ax.imshow(np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0), origin="upper")
            self._draw_inspection_marker(ax, row=row, col=col, ipf=True)
            if zoom:
                ax.set_xlim(*xlim_roi)
                ax.set_ylim(*ylim_roi)
            else:
                ax.set_xlim(*xlim_full)
                ax.set_ylim(*ylim_full)
                ax.add_patch(
                    Rectangle(
                        (c0 - 0.5, r0 - 0.5),
                        roi_ncols,
                        roi_nrows,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.3,
                    )
                )
            ax.set_title(title)
            ax.set_axis_off()

        def _draw_numeric(
            ax,
            image: np.ndarray | None,
            title: str,
            *,
            zoom: bool,
            cmap: str,
            vmin: float,
            vmax: float,
        ) -> None:
            if image is None:
                ax.text(0.5, 0.5, "Run indexing to populate", ha="center", va="center")
            else:
                ax.imshow(np.asarray(image, dtype=np.float32), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            self._draw_inspection_marker(ax, row=row, col=col, ipf=False)
            if zoom:
                ax.set_xlim(*xlim_roi)
                ax.set_ylim(*ylim_roi)
            else:
                ax.set_xlim(*xlim_full)
                ax.set_ylim(*ylim_full)
                ax.add_patch(
                    Rectangle(
                        (c0 - 0.5, r0 - 0.5),
                        roi_ncols,
                        roi_nrows,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.3,
                    )
                )
            ax.set_title(title)
            ax.set_axis_off()

        _draw_rgb(axes[0, 0], preliminary_ipf, f"Preliminary {ipf_label} (full map)", zoom=False)
        _draw_numeric(
            axes[0, 1],
            quality_full,
            f"{quality_layer} map (full map)",
            zoom=False,
            cmap=quality_cmap,
            vmin=qvmin,
            vmax=qvmax,
        )
        if phase_legend:
            from matplotlib.patches import Patch
            axes[0, 1].legend(handles=[Patch(color=color, label=label) for _, label, color in phase_legend],
                              loc="upper left", fontsize=6, framealpha=.85)
        _draw_rgb(axes[0, 2], preliminary_ipf, f"Preliminary {ipf_label} (ROI zoom)", zoom=True)
        _draw_numeric(
            axes[1, 0],
            quality_full,
            f"Initial {quality_layer} map (ROI zoom)",
            zoom=True,
            cmap=quality_cmap,
            vmin=qvmin,
            vmax=qvmax,
        )
        _draw_rgb(axes[1, 1], updated_ipf, f"Re-indexed ROI {ipf_label}", zoom=True)
        if ncc_map is not None:
            finite = np.asarray(ncc_map, dtype=np.float32)[np.isfinite(np.asarray(ncc_map, dtype=np.float32))]
            if finite.size > 0:
                nvmin = float(np.nanpercentile(finite, 2))
                nvmax = float(np.nanpercentile(finite, 98))
                if nvmax <= nvmin:
                    nvmax = nvmin + 1e-8
            else:
                nvmin, nvmax = 0.0, 1.0
        else:
            nvmin, nvmax = 0.0, 1.0
        _draw_numeric(
            axes[1, 2],
            ncc_map,
            "ROI NCC map after indexing",
            zoom=True,
            cmap="viridis",
            vmin=nvmin,
            vmax=nvmax,
        )

        point = self.session.get_point_state(index)
        e_deg = np.asarray(point["euler_deg"], dtype=np.float64).reshape(3)
        phase = int(point["phase"])
        info_lines = [
            "Step 2: region selection and dictionary re-indexing",
            f"Selected point: idx={index}, row={row}, col={col}, phase={phase}",
            f"Euler (deg): phi1={e_deg[0]:.2f}, Phi={e_deg[1]:.2f}, phi2={e_deg[2]:.2f}",
            f"ROI: r0={r0}, c0={c0}, nrows={roi_nrows}, ncols={roi_ncols}",
            f"Quality layer: {quality_layer}",
            f"IPF direction: {ipf_direction.upper()}",
            "Click any map to move the selected point.",
        ]
        if indexed_mask is not None and np.any(indexed_mask):
            info_lines.append(f"Dictionary-indexed points in workflow: {int(np.count_nonzero(indexed_mask))}")
        self._set_info_lines(info_lines)
        self._apply_plot_font_sizes(self.figure, axes)
        self._safe_tight_layout(self.figure)
        self.canvas.draw_idle()

    def _refresh_overlap_map_view(self, *, row: int, col: int, index: int) -> None:
        if self.session.data is None:
            return
        data = self.session.data
        axes = np.asarray(self.axes, dtype=object)
        primary_ax = axes[0, 0]
        residual_ax = axes[0, 1]
        exp_ax = axes[0, 2]
        sim_ax = axes[0, 3]
        primary_residual_ax = axes[1, 0]
        gain_ax = axes[1, 1]
        residual_sim_ax = axes[1, 2]
        residual_score_ax = axes[1, 3]
        residual_indexed = (
            (self.session.last_residual_indexed_indices is not None and self.session.last_residual_indexed_indices.size > 0)
            or any(res.secondary_euler_rad is not None for res in self.session.residual_point_results.values())
        )

        for ax in axes.flat:
            ax.clear()
            ax.set_axis_off()

        r0, c0, nrows, ncols = self._roi_bounds()
        r1 = min(data.rows - 1, r0 + nrows - 1)
        c1 = min(data.cols - 1, c0 + ncols - 1)
        zoom_note = "full map"
        if r0 > 0 or c0 > 0 or nrows < data.rows or ncols < data.cols:
            zoom_note = f"ROI rows {r0}:{r1}, cols {c0}:{c1}"
        xlim = (c0 - 0.5, c1 + 0.5) if zoom_note != "full map" else (-0.5, data.cols - 0.5)
        ylim = (r1 + 0.5, r0 - 0.5) if zoom_note != "full map" else (data.rows - 0.5, -0.5)

        def _decorate_map_axis(ax, image: np.ndarray | None, title: str, placeholder: str) -> None:
            if image is not None:
                ax.imshow(np.clip(image, 0.0, 1.0), origin="upper")
            else:
                ax.text(0.5, 0.5, placeholder, ha="center", va="center")
            self._draw_inspection_marker(ax, row=row, col=col, ipf=True)
            ax.add_patch(
                Rectangle(
                    (c0 - 0.5, r0 - 0.5),
                    ncols,
                    nrows,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.3,
                )
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.set_axis_off()

        def _decorate_score_axis(ax) -> None:
            self._draw_inspection_marker(ax, row=row, col=col, ipf=False)
            ax.add_patch(
                Rectangle(
                    (c0 - 0.5, r0 - 0.5),
                    ncols,
                    nrows,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.3,
                )
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_axis_off()

        ipf_direction, ipf_label = self._selected_solution_map()
        primary_ipf = self._solution_map_image(ipf_direction)
        primary_threshold_mask = self._primary_threshold_mask()
        primary_ipf = self._apply_white_mask(primary_ipf, primary_threshold_mask)
        residual_ipf = None
        residual_map_note = f"Residual {ipf_label} will appear after residual workflow runs"
        threshold_mask = self._residual_threshold_mask()
        if residual_indexed:
            try:
                residual_ipf = self._solution_map_image(ipf_direction, secondary=True)
                residual_map_note = f"Residual {ipf_label} (reindexed only)"
            except Exception:
                residual_ipf = None

        if primary_threshold_mask is not None:
            threshold_mask = (
                primary_threshold_mask
                if threshold_mask is None
                else np.logical_or(threshold_mask, primary_threshold_mask)
            )
        residual_ipf = self._apply_white_mask(residual_ipf, threshold_mask)

        _decorate_map_axis(
            primary_ax,
            primary_ipf,
            f"Dictionary-indexed primary {ipf_label}",
            f"Dictionary-indexed primary {ipf_label}",
        )
        _decorate_map_axis(residual_ax, residual_ipf, residual_map_note, residual_map_note)

        exp_raw = self.session._processed_pattern_at(index)
        preview_sim = None
        preview_ncc = None
        preview_residual = None
        preview_scale = None
        preview_residual_ncc = None
        if self.session.master is not None and self.session.current_eulers_rad is not None:
            try:
                preview_sim, preview_ncc, preview_residual, preview_scale, preview_residual_ncc = self.session.preview_simulated_pattern_with_ncc(index)
            except Exception:
                preview_sim = None
                preview_ncc = None
                preview_residual = None
                preview_scale = None
                preview_residual_ncc = None

        result = self.session.get_residual_point_result(index)
        if result is not None and result.index == index:
            exp_ax.imshow(normalize_for_view(result.experimental), cmap="gray")
            self._overlay_pattern_mask(exp_ax)
            exp_ax.set_title("Experimental pattern")
            sim_ax.imshow(normalize_for_view(result.simulated), cmap="gray")
            pre = result.ncc_unfitted if result.ncc_unfitted is not None else float("nan")
            sim_ax.set_title(self._simulation_title(f"Primary simulation — NCC {pre:.4f} → {result.ncc_es:.4f}", index, result=result))
            if result.gain_map is not None:
                gain_ax.imshow(result.gain_map, cmap="viridis")
                gain_ax.set_title("Fitted gain mask")
            else:
                gain_ax.text(0.5, 0.5, "Gain fitting disabled", ha="center", va="center")
                gain_ax.set_title("Gain mask")
            rabs = max(float(np.max(np.abs(result.residual))), 1e-8)
            primary_residual_ax.imshow(
                result.residual,
                cmap=RESIDUAL_PATTERN_CMAP,
                vmin=-rabs,
                vmax=rabs,
            )
            primary_residual_ax.set_title(f"Residual E − {result.scale:.4f}·S′")
            if result.secondary_simulated is not None:
                residual_sim_ax.imshow(normalize_for_view(result.secondary_simulated), cmap="gray")
                match_label = "Refined residual simulation" if result.secondary_refined else "Residual simulation"
                residual_sim_ax.set_title(self._simulation_title(f"{match_label} — KP NCC={result.secondary_ncc_kp:.4f}", index, result=result, secondary=True))
            else:
                residual_sim_ax.text(0.5, 0.5, "Index the residual with tab 2 first", ha="center", va="center")
                residual_sim_ax.set_title("Residual simulated pattern")
            if residual_indexed and self.session.last_residual_scores_map is not None:
                residual_scores = self.session.last_residual_scores_map
                finite_scores = residual_scores[np.isfinite(residual_scores)]
                if finite_scores.size > 0:
                    vmin = float(np.nanpercentile(finite_scores, 2))
                    vmax = float(np.nanpercentile(finite_scores, 98))
                    if vmax <= vmin:
                        vmax = vmin + 1e-8
                    residual_score_ax.imshow(residual_scores, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
                    residual_score_ax.set_title("Residual KP score map")
                else:
                    residual_score_ax.text(0.5, 0.5, "Residual score map is empty", ha="center", va="center")
                    residual_score_ax.set_title("Residual score map")
            elif self.session.last_scores_map is not None:
                residual_score_ax.imshow(self.session.last_scores_map, cmap="viridis", origin="upper")
                residual_score_ax.set_title("Latest KP score map")
            else:
                residual_score_ax.text(0.5, 0.5, "Residual scores appear after ROI indexing", ha="center", va="center")
                residual_score_ax.set_title("Residual score map")
        else:
            exp_ax.imshow(normalize_for_view(exp_raw), cmap="gray")
            self._overlay_pattern_mask(exp_ax)
            exp_ax.set_title("Experimental pattern")
            if preview_sim is not None and preview_ncc is not None:
                sim_ax.imshow(normalize_for_view(preview_sim), cmap="gray")
                sim_ax.set_title(self._simulation_title(f"Indexed solution simulation — NCC={preview_ncc:.4f}", index))
            else:
                sim_ax.text(0.5, 0.5, "Run step 2 indexing first", ha="center", va="center")
                sim_ax.set_title("Indexed solution simulation")
            if preview_residual is not None and preview_scale is not None:
                rabs = max(float(np.max(np.abs(preview_residual))), 1e-8)
                primary_residual_ax.imshow(
                    preview_residual,
                    cmap=RESIDUAL_PATTERN_CMAP,
                    vmin=-rabs,
                    vmax=rabs,
                )
                primary_residual_ax.set_title(f"Preview residual E − {preview_scale:.4f}·S")
            else:
                primary_residual_ax.text(0.5, 0.5, "Preview residual appears after step 2 indexing", ha="center", va="center")
                primary_residual_ax.set_title("Primary residual preview")
            gain_ax.text(0.5, 0.5, "Fit the primary pattern to see the gain mask", ha="center", va="center")
            gain_ax.set_title("Gain mask")
            if preview_residual_ncc is not None:
                residual_sim_ax.text(
                    0.5,
                    0.5,
                    f"Residual NCC preview: {preview_residual_ncc:.4f}",
                    ha="center",
                    va="center",
                )
                residual_sim_ax.set_title("Residual simulated pattern")
            else:
                residual_sim_ax.text(0.5, 0.5, "Fit and index the residual to show the match", ha="center", va="center")
                residual_sim_ax.set_title("Residual simulated pattern")
            if residual_indexed and self.session.last_residual_scores_map is not None:
                residual_scores = self.session.last_residual_scores_map
                finite_scores = residual_scores[np.isfinite(residual_scores)]
                if finite_scores.size > 0:
                    vmin = float(np.nanpercentile(finite_scores, 2))
                    vmax = float(np.nanpercentile(finite_scores, 98))
                    if vmax <= vmin:
                        vmax = vmin + 1e-8
                    residual_score_ax.imshow(residual_scores, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
                    residual_score_ax.set_title("Residual KP score map")
                else:
                    residual_score_ax.text(0.5, 0.5, "Residual score map is empty", ha="center", va="center")
                    residual_score_ax.set_title("Residual KP score map")
            elif self.session.last_scores_map is not None:
                score_map = self.session.last_scores_map
                finite_scores = score_map[np.isfinite(score_map)]
                if finite_scores.size > 0:
                    vmin = float(np.nanpercentile(finite_scores, 2))
                    vmax = float(np.nanpercentile(finite_scores, 98))
                    if vmax <= vmin:
                        vmax = vmin + 1e-8
                    residual_score_ax.imshow(score_map, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
                    residual_score_ax.set_title("Latest KP score map")
                else:
                    residual_score_ax.text(0.5, 0.5, "Score map is empty", ha="center", va="center")
                    residual_score_ax.set_title("Latest KP score map")
            else:
                residual_score_ax.text(0.5, 0.5, "Residual score map appears after ROI indexing", ha="center", va="center")
                residual_score_ax.set_title("Residual score map")

        _decorate_score_axis(residual_score_ax)

        info_lines = [
            f"Re-indexed {ipf_label} / residual view",
            f"Selected inspection point: idx={index}, row={row}, col={col}",
            "The same selected point is marked on both IPF maps.",
        ]
        if result is not None:
            info_lines.extend(
                [
                    f"Primary NCC: {result.ncc_unfitted:.4f} before fit → {result.ncc_es:.4f} after fit",
                    f"Fitted σ={result.fitted_sigma:.4f}; gain (gmin, gmax, p)={result.gain_params}",
                    f"Ellipse (a, b, y offset, x offset)={result.ellipse_params}",
                    f"Fit status: {result.fit_message}",
                ]
            )
            if result.secondary_ncc_kp is not None:
                label = "Refined residual NCC" if result.secondary_refined else "Residual dictionary NCC"
                info_lines.append(f"{label}: {result.secondary_ncc_kp:.4f}")
            if result.secondary_refinement_note:
                info_lines.append(result.secondary_refinement_note)
        elif preview_ncc is not None:
            info_lines.extend(
                [
                    f"Indexed-solution NCC: {preview_ncc:.4f}",
                    f"Preview residual scale: {preview_scale:.4f}" if preview_scale is not None else "Preview residual scale: n/a",
                    f"Preview residual NCC: {preview_residual_ncc:.4f}" if preview_residual_ncc is not None else "Preview residual NCC: n/a",
                ]
            )
        self._set_info_lines(info_lines)
        self._apply_plot_font_sizes(self.figure, axes)
        self._safe_tight_layout(self.figure)
        self.canvas.draw_idle()

    def _refresh_overlap_optimization_view(self, *, row: int, col: int, index: int) -> None:
        if self.session.data is None:
            return
        data = self.session.data
        axes = np.asarray(self.axes, dtype=object)
        if axes.shape != (2, 4):
            return
        primary_ax = axes[0, 0]
        residual_ax = axes[0, 1]
        exp_ax = axes[0, 2]
        primary_sim_ax = axes[0, 3]
        secondary_sim_ax = axes[1, 0]
        final_residual_ax = axes[1, 1]
        primary_fraction_ax = axes[1, 2]
        secondary_fraction_ax = axes[1, 3]

        for ax in axes.flat:
            ax.clear()
            ax.set_axis_off()

        r0, c0, nrows, ncols = self._roi_bounds()
        r1 = min(data.rows - 1, r0 + nrows - 1)
        c1 = min(data.cols - 1, c0 + ncols - 1)
        zoom_note = "full map"
        if r0 > 0 or c0 > 0 or nrows < data.rows or ncols < data.cols:
            zoom_note = f"ROI rows {r0}:{r1}, cols {c0}:{c1}"
        xlim = (c0 - 0.5, c1 + 0.5) if zoom_note != "full map" else (-0.5, data.cols - 0.5)
        ylim = (r1 + 0.5, r0 - 0.5) if zoom_note != "full map" else (data.rows - 0.5, -0.5)

        def _fmt(value: float | None, digits: int = 4) -> str:
            if value is None:
                return "n/a"
            try:
                fval = float(value)
            except Exception:
                return "n/a"
            return f"{fval:.{digits}f}" if np.isfinite(fval) else "n/a"

        def _max_abs_delta_deg(values: tuple[float, ...] | list[float] | None) -> float:
            if not values:
                return 0.0
            arr = np.asarray(values, dtype=np.float64).ravel()
            if arr.size == 0 or not np.any(np.isfinite(arr)):
                return 0.0
            return float(np.nanmax(np.abs(arr)))

        def _decorate_map_axis(ax, image: np.ndarray | None, title: str, placeholder: str) -> None:
            if image is not None:
                ax.imshow(np.clip(image, 0.0, 1.0), origin="upper")
            else:
                ax.text(0.5, 0.5, placeholder, ha="center", va="center")
            self._draw_inspection_marker(ax, row=row, col=col, ipf=True)
            ax.add_patch(
                Rectangle(
                    (c0 - 0.5, r0 - 0.5),
                    ncols,
                    nrows,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.3,
                )
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.set_axis_off()

        def _decorate_fraction_axis(ax, fraction_map: np.ndarray | None, title: str) -> None:
            if fraction_map is not None and np.any(np.isfinite(fraction_map)):
                masked = np.ma.masked_invalid(np.asarray(fraction_map, dtype=np.float32))
                ax.imshow(masked, cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
            else:
                ax.text(0.5, 0.5, "Fit ROI mixture", ha="center", va="center")
            self._draw_inspection_marker(ax, row=row, col=col, ipf=False)
            ax.add_patch(
                Rectangle(
                    (c0 - 0.5, r0 - 0.5),
                    ncols,
                    nrows,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.3,
                )
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.set_axis_off()

        ipf_direction, ipf_label = self._selected_solution_map()
        primary_ipf = self._solution_map_image(ipf_direction)
        primary_threshold_mask = self._primary_threshold_mask()
        primary_ipf = self._apply_white_mask(primary_ipf, primary_threshold_mask)
        residual_ipf = None
        residual_note = f"Residual {ipf_label}"
        try:
            residual_ipf = self._solution_map_image(ipf_direction, secondary=True)
        except Exception:
            residual_note = f"Residual {ipf_label} after step 3"
        threshold_mask = self._overlap_mixture_residual_threshold_mask()
        residual_ipf = self._apply_white_mask(residual_ipf, threshold_mask)

        _decorate_map_axis(
            primary_ax,
            primary_ipf,
            f"Dictionary-indexed primary {ipf_label}",
            f"Dictionary-indexed primary {ipf_label}",
        )
        _decorate_map_axis(residual_ax, residual_ipf, residual_note, residual_note)

        result = self.session.get_overlap_mixture_result(index)

        if result is not None and result.index == index:
            exp_ax.imshow(normalize_for_view(result.experimental), cmap="gray")
            self._overlay_pattern_mask(exp_ax)
            exp_ax.set_title("Experimental pattern")
            refined_tag = "refined " if result.orientation_refined else ""
            show_delta = result.initial_mixture_ncc is not None or result.orientation_refined
            primary_title = f"{refined_tag}Primary sim | f={_fmt(result.primary_fraction, 3)}"
            secondary_title = f"{refined_tag}Residual sim | f={_fmt(result.secondary_fraction, 3)}"
            if show_delta:
                primary_title += f"\ndEuler={_fmt(_max_abs_delta_deg(result.primary_euler_delta_deg), 3)} deg"
                secondary_title += f"\ndEuler={_fmt(_max_abs_delta_deg(result.secondary_euler_delta_deg), 3)} deg"
            primary_sim_ax.imshow(normalize_for_view(result.primary_simulated), cmap="gray")
            primary_sim_ax.set_title(self._simulation_title(primary_title, index, result=result))
            secondary_sim_ax.imshow(normalize_for_view(result.secondary_simulated), cmap="gray")
            secondary_sim_ax.set_title(self._simulation_title(secondary_title, index, result=result, secondary=True))
            rabs = max(float(np.max(np.abs(result.residual))), 1e-8)
            final_residual_ax.imshow(
                result.residual,
                cmap=RESIDUAL_PATTERN_CMAP,
                vmin=-rabs,
                vmax=rabs,
            )
            final_residual_ax.set_title(
                f"Final residual | NCC={_fmt(result.ncc_mixture)}\n"
                f"old primary={_fmt(result.old_primary_ncc)}, old residual={_fmt(result.old_secondary_ncc)}"
            )
        else:
            exp_raw = self.session._processed_pattern_at(index)
            exp_ax.imshow(normalize_for_view(exp_raw), cmap="gray")
            self._overlay_pattern_mask(exp_ax)
            exp_ax.set_title("Experimental pattern")
            primary_sim_ax.text(0.5, 0.5, "Fit selected mixture", ha="center", va="center")
            primary_sim_ax.set_title("Primary sim")
            secondary_sim_ax.text(0.5, 0.5, "Requires residual orientation", ha="center", va="center")
            secondary_sim_ax.set_title("Residual sim")
            final_residual_ax.text(0.5, 0.5, "Fit selected mixture", ha="center", va="center")
            final_residual_ax.set_title("Final residual")

        for ax in (exp_ax, primary_sim_ax, secondary_sim_ax, final_residual_ax):
            ax.set_axis_off()

        _decorate_fraction_axis(primary_fraction_ax, self.session.overlap_primary_fraction_map, "Primary fitted contribution")
        _decorate_fraction_axis(secondary_fraction_ax, self.session.overlap_secondary_fraction_map, "Residual fitted contribution")

        info_lines = [
            "Overlap optimization",
            f"Selected point: idx={index}, row={row}, col={col}",
            f"IPF direction: {ipf_direction.upper()}",
            f"Residual NCC threshold: {_fmt(self._overlap_mixture_residual_ncc_threshold(), 3)}",
        ]
        if result is not None and result.index == index:
            info_lines.extend(
                [
                    f"Pattern contributions: primary={_fmt(result.primary_fraction, 3)}, residual={_fmt(result.secondary_fraction, 3)}",
                    f"NCC: old primary={_fmt(result.old_primary_ncc)}, old residual={_fmt(result.old_secondary_ncc)}, combined={_fmt(result.ncc_mixture)}",
                    f"Fitted sigma={result.fitted_sigma:.4f}; gain (gmin, gmax, p)={result.gain_params}",
                    f"Ellipse (a, b, y offset, x offset)={result.ellipse_params}",
                    f"Coefficients: primary={result.primary_coefficient:.4f}, residual={result.secondary_coefficient:.4f}",
                ]
            )
            for title, key in (("Primary phase", result.primary_phase_key), ("Residual phase", result.secondary_phase_key)):
                if key:
                    entry = self.session.phase_registry.by_key(key)
                    info_lines.append(f"{title}: {entry.name} [{entry.output_id}]")
            if result.overlap_acceptance_note:
                info_lines.append(f"Overlap: {result.overlap_acceptance_note}")
            info_lines.append("Contributions describe pattern intensity, not phase volume fractions.")
            if result.initial_mixture_ncc is not None:
                info_lines.append(
                    f"Orientation refinement NCC: {result.initial_mixture_ncc:.4f} -> {result.ncc_mixture:.4f}"
                )
            if result.primary_euler_delta_deg or result.secondary_euler_delta_deg:
                p_delta = tuple(float(v) for v in result.primary_euler_delta_deg)
                s_delta = tuple(float(v) for v in result.secondary_euler_delta_deg)
                info_lines.append(f"Primary Euler delta (deg): {p_delta}")
                info_lines.append(f"Residual Euler delta (deg): {s_delta}")
            if result.orientation_refinement_note:
                info_lines.append(result.orientation_refinement_note)
            if abs(float(result.component_correlation)) > 0.90:
                info_lines.append(f"Component NCC={result.component_correlation:.4f}; fraction estimate is strongly coupled.")
            if result.fit_message:
                info_lines.append(f"Fit status: {result.fit_message}")
        else:
            info_lines.append("Fit a selected point or ROI after step 3 residual indexing/refinement.")
        self._set_info_lines(info_lines)
        self._apply_plot_font_sizes(self.figure, axes)
        self._safe_tight_layout(self.figure)
        self.canvas.draw_idle()

    @_guarded_action
    def _on_plot_click(self, event, view_index: int | None = None) -> None:
        if self.session.data is None:
            return
        active_view = int(view_index) if view_index is not None else None
        if view_index is not None:
            self._activate_plot_view(int(view_index))
        if event.inaxes is None:
            return
        if active_view == 1 and self._maybe_begin_roi_drag(event, active_view):
            return
        allowed_axes = [
            ax
            for ax in np.asarray(self.axes, dtype=object).flat
            if bool(getattr(ax, "_overlap_ebsd_scan_map", False))
        ]
        if event.inaxes not in allowed_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        row = int(np.clip(round(event.ydata), 0, self.session.data.rows - 1))
        col = int(np.clip(round(event.xdata), 0, self.session.data.cols - 1))
        self.row_var.set(row)
        self.col_var.set(col)
        self._sync_index_from_row_col()

    def _on_plot_motion(self, event, view_index: int | None = None) -> None:
        if self._roi_drag_view_index is None or self._roi_drag_view_index != int(view_index or -1):
            return
        self._update_roi_drag(event)

    def _on_plot_release(self, event, view_index: int | None = None) -> None:
        if self._roi_drag_view_index is None or self._roi_drag_view_index != int(view_index or -1):
            return
        self._finish_roi_drag(event)

    def _populate_point_vars(self) -> None:
        if self.session.data is None:
            return
        idx = int(self.index_var.get())
        state = self.session.get_point_state(idx)
        e_deg = np.asarray(state["euler_deg"], dtype=np.float64).reshape(3)
        pc = np.asarray(state["pc_custom"], dtype=np.float64).reshape(3)
        phase = state["phase"]
        conv = str(state["pc_convention"])
        self._suspend_point_trace = True
        try:
            for variable, value in zip(
                (self.euler1_deg_var, self.euler2_deg_var, self.euler3_deg_var,
                 self.pcx_var, self.pcy_var, self.pcz_var),
                (*e_deg, *pc),
            ):
                variable.set(float(value))
        finally:
            self._suspend_point_trace = False
        self.pc_conv_label_var.set(f"PC convention: {conv} | point phase: {phase}")


def normalize_for_view(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    lo = float(np.percentile(a, 1))
    hi = float(np.percentile(a, 99))
    if hi <= lo:
        lo = float(a.min())
        hi = float(a.max()) + 1e-8
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
