from __future__ import annotations

import threading
import traceback
import warnings
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .core import ORIENTATION_LAYER_LABEL, GeometryConfig, OverlapMixtureResult, OverlapPointResult, WorkflowSession

PLOT_TITLE_FONTSIZE = 9
PLOT_TEXT_FONTSIZE = 8
PLOT_TICK_FONTSIZE = 7
PLOT_INSTRUCTION_FONTSIZE = 10
_TIGHT_LAYOUT_WARNING = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect."


class MultiStepOverlapGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Overlap EBSD Multi-Step Workflow")
        self.geometry("1680x980")

        self.session = WorkflowSession()
        self.last_overlap: OverlapPointResult | None = None
        self.last_overlap_mixture: OverlapMixtureResult | None = None
        self.busy = False
        self._suspend_point_trace = False
        self._live_refresh_after_id: str | None = None
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
        self.btn_index_full: ttk.Button | None = None
        self.btn_refine_indexed: ttk.Button | None = None
        self.workflow_notebook: ttk.Notebook | None = None

        cwd = Path.cwd()
        self.pattern_path_var = tk.StringVar(value=str((cwd / "PJablonski 45 Site 1 Map Data 2.h5oina").resolve()))
        self.orientation_path_var = tk.StringVar(value=str((cwd / "insitu Specimen 1 0 Map Data 3_BW123.ang").resolve()))
        self.master_path_var = tk.StringVar(value=str((cwd / "Cu-master_20kV.h5").resolve()))
        self.export_path_var = tk.StringVar(value=str((cwd / "reindexed_output.h5oina").resolve()))

        self.sample_tilt_var = tk.DoubleVar(value=70.0)
        self.detector_tilt_var = tk.DoubleVar(value=0.0)

        self.phase_id_var = tk.IntVar(value=1)
        self.index_var = tk.IntVar(value=0)
        self.row_var = tk.IntVar(value=0)
        self.col_var = tk.IntVar(value=0)
        self.roi_r0_var = tk.IntVar(value=0)
        self.roi_c0_var = tk.IntVar(value=0)
        self.roi_nrows_var = tk.IntVar(value=31)
        self.roi_ncols_var = tk.IntVar(value=31)
        self.roi_r1_var = self.roi_nrows_var
        self.roi_c1_var = self.roi_ncols_var
        self.euler1_deg_var = tk.DoubleVar(value=0.0)
        self.euler2_deg_var = tk.DoubleVar(value=0.0)
        self.euler3_deg_var = tk.DoubleVar(value=0.0)
        self.pcx_var = tk.DoubleVar(value=0.0)
        self.pcy_var = tk.DoubleVar(value=0.0)
        self.pcz_var = tk.DoubleVar(value=0.0)
        self.pc_conv_label_var = tk.StringVar(value="PC convention: -")

        self.trust_euler_var = tk.DoubleVar(value=1.0)
        self.trust_pc_var = tk.DoubleVar(value=0.03)
        self.maxfev_var = tk.IntVar(value=25)

        self.di_res_deg_var = tk.DoubleVar(value=12.0)
        self.di_binning_var = tk.IntVar(value=4)
        self.dictionary_keep_n_var = tk.IntVar(value=1)
        self.dictionary_status_var = tk.StringVar(value="No dictionary generated or loaded.")
        self.dictionary_progress_var = tk.DoubleVar(value=0.0)
        self.reindex_progress_var = tk.DoubleVar(value=0.0)
        self.reindex_progress_status_var = tk.StringVar(value="Re-indexing not started.")
        self.overlap_progress_var = tk.DoubleVar(value=0.0)
        self.overlap_progress_status_var = tk.StringVar(value="Residual ROI workflow not started.")
        self.overlap_optimization_progress_var = tk.DoubleVar(value=0.0)
        self.overlap_optimization_status_var = tk.StringVar(value="Overlap optimization not started.")
        self.refinement_progress_var = tk.DoubleVar(value=0.0)
        self.refinement_progress_status_var = tk.StringVar(value="Orientation refinement not started.")
        self.dictionary_path_var = tk.StringVar(value=str((cwd / "ebsd_dictionary_binned.h5").resolve()))
        self.primary_roi_export_path_var = tk.StringVar(value=str((cwd / "primary_roi_map.h5oina").resolve()))
        self.residual_roi_export_path_var = tk.StringVar(value=str((cwd / "residual_roi_map.h5oina").resolve()))
        self.blur_sigma_var = tk.DoubleVar(value=0.0)
        self.fit_blur_gain_var = tk.BooleanVar(value=True)
        self.gain_fit_maxiter_var = tk.IntVar(value=80)
        self.gain_fit_popsize_var = tk.IntVar(value=15)
        self.residual_trust_euler_var = tk.DoubleVar(value=2.0)
        self.residual_maxfev_var = tk.IntVar(value=100)
        self.residual_keep_n_var = tk.IntVar(value=1)
        self.overlap_mixture_trust_euler_var = tk.DoubleVar(value=1.0)
        self.overlap_mixture_maxfev_var = tk.IntVar(value=80)
        self.overlap_min_ncc_var = tk.StringVar(value="0.15")
        self.residual_ipf_ncc_var = tk.StringVar(value="0.15")
        self.overlap_mixture_residual_ncc_var = tk.StringVar(value=self.residual_ipf_ncc_var.get())
        self.write_residual_patterns_var = tk.BooleanVar(value=False)
        self.residual_pattern_path_var = tk.StringVar(value=str((cwd / "residual_patterns.h5oina").resolve()))
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
        self.use_scan_pc_shift_var = tk.BooleanVar(value=True)
        self.detector_px_size_var = tk.DoubleVar(value=1.0)
        self.detector_binning_var = tk.DoubleVar(value=1.0)
        self.calibration_summary_var = tk.StringVar(value="No calibration points selected.")
        self.pattern_mask_option_var = tk.IntVar(value=-1)
        self.pattern_mask_status_var = tk.StringVar(value=f"Mask: {self.session.pattern_mask_description()}")
        self.dynamic_bg_enabled_var = tk.BooleanVar(value=False)
        self.dynamic_bg_std_var = tk.StringVar(value="0")
        self.dynamic_bg_status_var = tk.StringVar(value=f"Dynamic BG: {self.session.dynamic_background_description()}")

        self.map_layer_var = tk.StringVar(value=ORIENTATION_LAYER_LABEL)
        self.index_quality_layer_var = tk.StringVar(value="CI")
        self.status_var = tk.StringVar(value="Load data to begin.")
        self.overlap_progress_bar: ttk.Progressbar | None = None

        self._build_ui()
        self._attach_point_value_traces()
        self._attach_threshold_value_traces()
        self._attach_entry_commit_handlers()

    # ---------------------------- UI ---------------------------- #

    def _build_ui(self) -> None:
        self.workflow_notebook = ttk.Notebook(self)
        self.workflow_notebook.pack(fill=tk.BOTH, expand=True)

        tab_calibration = ttk.Frame(self.workflow_notebook)
        tab_indexing = ttk.Frame(self.workflow_notebook)
        tab_overlap = ttk.Frame(self.workflow_notebook)
        tab_overlap_optimization = ttk.Frame(self.workflow_notebook)
        self.workflow_notebook.add(tab_calibration, text="1. Load and PC Calibration")
        self.workflow_notebook.add(tab_indexing, text="2. Dictionary Indexing")
        self.workflow_notebook.add(tab_overlap, text="3. Overlap Indexing")
        self.workflow_notebook.add(tab_overlap_optimization, text="4. Overlap Optimization")

        self._build_calibration_workspace(tab_calibration)
        self._build_indexing_workspace(tab_indexing)
        self._build_overlap_workspace(tab_overlap)
        self._build_overlap_optimization_workspace(tab_overlap_optimization)
        self.workflow_notebook.bind("<<NotebookTabChanged>>", self._on_workspace_changed)

        ttk.Label(self, textvariable=self.status_var, padding=(8, 4), anchor="w").pack(fill=tk.X)
        self._activate_plot_view(0)
        self._draw_instruction("Load data and a master pattern to start.")

    def _workspace_panes(self, parent: ttk.Frame) -> tuple[ttk.Frame, ttk.Frame]:
        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, width=440)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        return left, right

    def _scrollable_controls(self, parent: ttk.Frame) -> ttk.Frame:
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0, width=430)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls = ttk.Frame(canvas, padding=8)
        window_id = canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))

        def wheel(event) -> None:
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            elif getattr(event, "delta", 0):
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        controls.bind("<Enter>", lambda _e: self.bind_all("<MouseWheel>", wheel))
        controls.bind("<Leave>", lambda _e: self.unbind_all("<MouseWheel>"))
        return controls

    def _build_calibration_workspace(self, parent: ttk.Frame) -> None:
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        self._build_input_controls(controls)
        calibration = ttk.LabelFrame(controls, text="PC Calibration", padding=8)
        calibration.pack(fill=tk.X, pady=4)
        self._build_refine_tab(calibration)
        self._build_selection_controls(controls, include_roi=False)
        self._build_point_editor_controls(controls)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 0)

    def _build_indexing_workspace(self, parent: ttk.Frame) -> None:
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        self._build_selection_controls(controls, include_roi=True)
        indexing = ttk.LabelFrame(controls, text="Kikuchipy Dictionary Indexing", padding=8)
        indexing.pack(fill=tk.X, pady=4)
        self._build_index_tab(indexing)
        refinement = ttk.LabelFrame(controls, text="Post-index Orientation Refinement", padding=8)
        refinement.pack(fill=tk.X, pady=4)
        ttk.Label(refinement, text="trust Euler (deg)").grid(row=0, column=0, sticky="w")
        ttk.Entry(refinement, textvariable=self.trust_euler_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(refinement, text="max evaluations").grid(row=1, column=0, sticky="w")
        ttk.Entry(refinement, textvariable=self.maxfev_var, width=10).grid(row=1, column=1, sticky="w")
        self.btn_refine_indexed = ttk.Button(
            refinement,
            text="Refine Last Indexed Orientations",
            command=self._refine_last_indexed,
        )
        self.btn_refine_indexed.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Progressbar(
            refinement,
            variable=self.refinement_progress_var,
            maximum=100.0,
            mode="determinate",
        ).grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Label(
            refinement,
            textvariable=self.refinement_progress_status_var,
            wraplength=390,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))
        refinement.columnconfigure(0, weight=1)
        output = ttk.LabelFrame(controls, text="Save and Reopen Results", padding=8)
        output.pack(fill=tk.X, pady=4)
        ttk.Entry(output, textvariable=self.export_path_var, width=42).grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Button(output, text="Browse", command=self._browse_export).grid(row=0, column=2, padx=4)
        ttk.Button(output, text="Export Re-indexed Map", command=self._export_results).grid(row=1, column=0, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Button(output, text="Save Workflow State", command=self._save_workflow).grid(row=2, column=0, sticky="we", pady=(4, 0))
        ttk.Button(output, text="Open Workflow State", command=self._restore_workflow).grid(row=2, column=1, columnspan=2, sticky="we", pady=(4, 0))
        output.columnconfigure(0, weight=1)
        output.columnconfigure(1, weight=1)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 1)

    def _build_overlap_workspace(self, parent: ttk.Frame) -> None:
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        overlap = ttk.LabelFrame(controls, text="Primary Subtraction and Residual Indexing", padding=8)
        overlap.pack(fill=tk.X, pady=4)
        self._build_overlap_tab(overlap)
        ttk.Label(
            controls,
            text="Paper model: fit Gaussian σ and an elliptical power-law gain mask, normalize S′, then subtract Zexp − NCC(E,S′)·S′. The residual is indexed with the dictionary retained from tab 2.",
            wraplength=390,
        ).pack(fill=tk.X, pady=(0, 4))
        self._build_selection_controls(controls, include_roi=True)
        self._build_info_and_log(controls, log_height=10)
        self._build_plot_area(right, 2, fixed_ipf=True)

    def _build_overlap_optimization_workspace(self, parent: ttk.Frame) -> None:
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        optimization = ttk.LabelFrame(controls, text="Shared Gain/Blur Mixture Fit", padding=8)
        optimization.pack(fill=tk.X, pady=4)
        self._build_overlap_optimization_tab(optimization)
        self._build_selection_controls(controls, include_roi=True)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 3, fixed_ipf=True)

    def _build_input_controls(self, parent: ttk.Frame) -> None:
        io_box = ttk.LabelFrame(parent, text="Input Data", padding=8)
        io_box.pack(fill=tk.X, pady=4)
        ttk.Label(io_box, text="Patterns (.h5oina / .up1 / .up2)").grid(row=0, column=0, sticky="w")
        ttk.Entry(io_box, textvariable=self.pattern_path_var, width=42).grid(row=1, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_patterns).grid(row=1, column=1, padx=4)
        ttk.Label(io_box, text="Orientations (.ang for UP files)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(io_box, textvariable=self.orientation_path_var, width=42).grid(row=3, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_orientation).grid(row=3, column=1, padx=4)
        ttk.Label(io_box, text="Master pattern").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(io_box, textvariable=self.master_path_var, width=42).grid(row=5, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_master).grid(row=5, column=1, padx=4)
        io_box.columnconfigure(0, weight=1)

        geometry = ttk.LabelFrame(parent, text="Geometry and Loading", padding=8)
        geometry.pack(fill=tk.X, pady=4)
        ttk.Label(geometry, text="sample tilt").grid(row=0, column=0, sticky="w")
        ttk.Entry(geometry, textvariable=self.sample_tilt_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(geometry, text="detector tilt").grid(row=1, column=0, sticky="w")
        ttk.Entry(geometry, textvariable=self.detector_tilt_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(geometry, text="pattern mask (-1/0/N px)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(geometry, textvariable=self.pattern_mask_option_var, width=10).grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(geometry, textvariable=self.pattern_mask_status_var, wraplength=390).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        bg_controls = ttk.Frame(geometry)
        bg_controls.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            bg_controls,
            text="dynamic BG subtraction",
            variable=self.dynamic_bg_enabled_var,
            command=self._on_value_commit,
        ).pack(side=tk.LEFT)
        ttk.Label(bg_controls, text="std px (0=auto)").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(bg_controls, textvariable=self.dynamic_bg_std_var, width=10).pack(side=tk.LEFT)
        ttk.Label(geometry, textvariable=self.dynamic_bg_status_var, wraplength=390).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        ttk.Button(geometry, text="Load Input Data", command=self._load_input).grid(row=6, column=0, sticky="we", pady=(8, 0))
        ttk.Button(geometry, text="Load Master Pattern", command=self._load_master).grid(row=6, column=1, sticky="we", pady=(8, 0))
        ttk.Button(geometry, text="Open Existing Workflow", command=self._restore_workflow).grid(row=7, column=0, columnspan=2, sticky="we", pady=(4, 0))

    def _build_selection_controls(self, parent: ttk.Frame, *, include_roi: bool) -> None:
        box = ttk.LabelFrame(parent, text="Map Selection", padding=8)
        box.pack(fill=tk.X, pady=4)
        ttk.Label(box, text="phase ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.phase_id_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="index").grid(row=1, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.index_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(box, text="Index → Row/Col", command=self._sync_row_col_from_index).grid(row=1, column=2, padx=4)
        ttk.Label(box, text="row").grid(row=2, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.row_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(box, text="column").grid(row=3, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.col_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Button(box, text="Row/Col → Index", command=self._sync_index_from_row_col).grid(row=3, column=2, padx=4)
        if include_roi:
            ttk.Label(box, text="ROI r0, c0, nrows, ncols").grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
            roi = ttk.Frame(box)
            roi.grid(row=5, column=0, columnspan=3, sticky="w")
            for variable in (self.roi_r0_var, self.roi_c0_var, self.roi_nrows_var, self.roi_ncols_var):
                ttk.Entry(roi, textvariable=variable, width=7).pack(side=tk.LEFT, padx=2)
            ttk.Button(box, text="Center ROI on Selected Point", command=self._center_roi_on_selected).grid(
                row=6, column=0, columnspan=3, sticky="we", pady=(4, 0)
            )
            ttk.Button(box, text="Apply ROI to Map", command=self._apply_roi_selection).grid(
                row=7, column=0, columnspan=3, sticky="we", pady=(4, 0)
            )

    def _build_point_editor_controls(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Selected Point Orientation and PC", padding=8)
        box.pack(fill=tk.X, pady=4)
        labels = ("Euler φ1 (deg)", "Euler Φ (deg)", "Euler φ2 (deg)")
        variables = (self.euler1_deg_var, self.euler2_deg_var, self.euler3_deg_var)
        for row, (label, variable) in enumerate(zip(labels, variables)):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w")
            ttk.Spinbox(box, textvariable=variable, from_=-720, to=720, increment=self._euler_step_deg, width=12).grid(row=row, column=1, sticky="w")
        for offset, (label, variable) in enumerate(zip(("PC x", "PC y", "PC z"), (self.pcx_var, self.pcy_var, self.pcz_var)), start=3):
            ttk.Label(box, text=label).grid(row=offset, column=0, sticky="w")
            ttk.Spinbox(box, textvariable=variable, from_=-2, to=2, increment=self._pc_step, width=12).grid(row=offset, column=1, sticky="w")
        ttk.Label(box, textvariable=self.pc_conv_label_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(box, text="Read Current Values", command=self._load_selected_point_values).grid(row=7, column=0, sticky="we", pady=(6, 0))
        ttk.Button(box, text="Apply Edited Values", command=self._apply_selected_point_values).grid(row=7, column=1, sticky="we", pady=(6, 0))

    def _build_info_and_log(self, parent: ttk.Frame, *, log_height: int) -> None:
        info_box = ttk.LabelFrame(parent, text="Current Status", padding=4)
        info_box.pack(fill=tk.X, pady=4)
        info = tk.Text(info_box, height=7, wrap=tk.WORD)
        info.pack(fill=tk.X)
        info.configure(state=tk.DISABLED)
        self.info_texts.append(info)
        if self.info_text is None:
            self.info_text = info
        log_box = ttk.LabelFrame(parent, text="Log", padding=4)
        log_box.pack(fill=tk.BOTH, expand=True, pady=4)
        log = tk.Text(log_box, height=log_height, wrap=tk.WORD)
        log.pack(fill=tk.BOTH, expand=True)
        self.log_texts.append(log)
        if not hasattr(self, "log_text"):
            self.log_text = log

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
        if view_index == 1:
            ttk.Label(top, text="Quality layer").pack(side=tk.LEFT)
            combo = ttk.Combobox(top, textvariable=self.index_quality_layer_var, values=[], state="readonly", width=18)
            combo.pack(side=tk.LEFT, padx=4)
            combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
            ttk.Button(top, text="Refresh", command=self._refresh_plot).pack(side=tk.LEFT, padx=4)
            ttk.Label(top, text="ROI selection maps").pack(side=tk.LEFT, padx=(10, 0))
        elif single_map or fixed_ipf:
            title = "Primary and residual IPF-Z diagnostics" if fixed_ipf else "Re-indexed IPF-Z orientation map"
            ttk.Label(top, text=title).pack(side=tk.LEFT)
            combo = ttk.Combobox(top, textvariable=self.map_layer_var, values=[ORIENTATION_LAYER_LABEL], state="disabled", width=1)
        else:
            ttk.Label(top, text="Map layer").pack(side=tk.LEFT)
            combo = ttk.Combobox(top, textvariable=self.map_layer_var, values=[ORIENTATION_LAYER_LABEL, "Phase"], state="readonly", width=20)
            combo.pack(side=tk.LEFT, padx=4)
            combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
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
        canvas.mpl_connect("button_press_event", lambda event, i=view_index: self._on_plot_click(event, i))
        self._plot_views[view_index] = {
            "figure": figure,
            "axes": axes,
            "canvas": canvas,
            "combo": combo,
            "colorbar": None,
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
        self._refresh_plot()

    def _build_scrollable_left(self, parent: ttk.Frame) -> None:
        self._left_canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        vscroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._left_canvas.yview)
        self._left_canvas.configure(yscrollcommand=vscroll.set)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._left_controls_frame = ttk.Frame(self._left_canvas, padding=8)
        self._left_canvas_window_id = self._left_canvas.create_window(
            (0, 0),
            window=self._left_controls_frame,
            anchor="nw",
        )

        self._left_controls_frame.bind("<Configure>", self._on_left_content_configure)
        self._left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self._left_controls_frame.bind("<Enter>", self._bind_left_mousewheel)
        self._left_controls_frame.bind("<Leave>", self._unbind_left_mousewheel)
        self._left_canvas.bind("<Enter>", self._bind_left_mousewheel)
        self._left_canvas.bind("<Leave>", self._unbind_left_mousewheel)

        self._build_left(self._left_controls_frame)

    def _on_left_content_configure(self, _event: tk.Event) -> None:
        if self._left_canvas is None:
            return
        self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event: tk.Event) -> None:
        if self._left_canvas is None or self._left_canvas_window_id is None:
            return
        self._left_canvas.itemconfigure(self._left_canvas_window_id, width=event.width)

    def _bind_left_mousewheel(self, _event: tk.Event) -> None:
        self.bind_all("<MouseWheel>", self._on_left_mousewheel)
        self.bind_all("<Button-4>", self._on_left_mousewheel)
        self.bind_all("<Button-5>", self._on_left_mousewheel)

    def _unbind_left_mousewheel(self, _event: tk.Event) -> None:
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_left_mousewheel(self, event: tk.Event) -> None:
        if self._left_canvas is None:
            return
        if hasattr(event, "num") and event.num == 4:
            self._left_canvas.yview_scroll(-1, "units")
            return
        if hasattr(event, "num") and event.num == 5:
            self._left_canvas.yview_scroll(1, "units")
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = -1 if delta > 0 else 1
        self._left_canvas.yview_scroll(step, "units")

    def _build_left(self, parent: ttk.Frame) -> None:
        io_box = ttk.LabelFrame(parent, text="Input / Output", padding=8)
        io_box.pack(fill=tk.X, pady=4)

        ttk.Label(io_box, text="Patterns (.h5oina / .up1 / .up2)").grid(row=0, column=0, sticky="w")
        ttk.Entry(io_box, textvariable=self.pattern_path_var, width=52).grid(row=1, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_patterns).grid(row=1, column=1, padx=4)

        ttk.Label(io_box, text="Orientation (.ang, for .up1/.up2)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(io_box, textvariable=self.orientation_path_var, width=52).grid(row=3, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_orientation).grid(row=3, column=1, padx=4)

        ttk.Label(io_box, text="Master pattern (.h5/.hdf5/.sdf5)").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(io_box, textvariable=self.master_path_var, width=52).grid(row=5, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_master).grid(row=5, column=1, padx=4)

        ttk.Label(io_box, text="Export path").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(io_box, textvariable=self.export_path_var, width=52).grid(row=7, column=0, sticky="we")
        ttk.Button(io_box, text="Browse", command=self._browse_export).grid(row=7, column=1, padx=4)

        io_box.columnconfigure(0, weight=1)

        geom_box = ttk.LabelFrame(parent, text="Geometry / Loading", padding=8)
        geom_box.pack(fill=tk.X, pady=4)
        ttk.Label(geom_box, text="sample tilt").grid(row=0, column=0, sticky="w")
        ttk.Entry(geom_box, textvariable=self.sample_tilt_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(geom_box, text="detector tilt").grid(row=1, column=0, sticky="w")
        ttk.Entry(geom_box, textvariable=self.detector_tilt_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(geom_box, text="Load Input Data", command=self._load_input).grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Button(geom_box, text="Load Master", command=self._load_master).grid(row=2, column=1, sticky="we", pady=(8, 0))
        ttk.Button(geom_box, text="Export Re-indexed Results", command=self._export_results).grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Button(geom_box, text="Save Workflow State", command=self._save_workflow).grid(row=4, column=0, sticky="we", pady=(4, 0))
        ttk.Button(geom_box, text="Open Workflow State", command=self._restore_workflow).grid(row=4, column=1, sticky="we", pady=(4, 0))

        select_box = ttk.LabelFrame(parent, text="Selection", padding=8)
        select_box.pack(fill=tk.X, pady=4)
        ttk.Label(select_box, text="phase ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(select_box, textvariable=self.phase_id_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(select_box, text="index").grid(row=1, column=0, sticky="w")
        ttk.Entry(select_box, textvariable=self.index_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(select_box, text="Index -> Row/Col", command=self._sync_row_col_from_index).grid(row=1, column=2, padx=4)
        ttk.Label(select_box, text="row").grid(row=2, column=0, sticky="w")
        ttk.Entry(select_box, textvariable=self.row_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(select_box, text="col").grid(row=3, column=0, sticky="w")
        ttk.Entry(select_box, textvariable=self.col_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Button(select_box, text="Row/Col -> Index", command=self._sync_index_from_row_col).grid(row=3, column=2, padx=4)

        ttk.Label(select_box, text="ROI r0, c0, nrows, ncols").grid(row=4, column=0, sticky="w", pady=(6, 0))
        roi_frame = ttk.Frame(select_box)
        roi_frame.grid(row=5, column=0, columnspan=3, sticky="w")
        ttk.Entry(roi_frame, textvariable=self.roi_r0_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Entry(roi_frame, textvariable=self.roi_c0_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Entry(roi_frame, textvariable=self.roi_nrows_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Entry(roi_frame, textvariable=self.roi_ncols_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(select_box, text="Center ROI on Selected Point", command=self._center_roi_on_selected).grid(
            row=6, column=0, columnspan=3, sticky="we", pady=(4, 0)
        )

        point_box = ttk.LabelFrame(parent, text="Current Point Values", padding=8)
        point_box.pack(fill=tk.X, pady=4)
        ttk.Label(point_box, text="Euler phi1 (deg)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            point_box,
            textvariable=self.euler1_deg_var,
            from_=-720.0,
            to=720.0,
            increment=self._euler_step_deg,
            format="%.2f",
            width=12,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(point_box, text="Euler Phi (deg)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            point_box,
            textvariable=self.euler2_deg_var,
            from_=-720.0,
            to=720.0,
            increment=self._euler_step_deg,
            format="%.2f",
            width=12,
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(point_box, text="Euler phi2 (deg)").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            point_box,
            textvariable=self.euler3_deg_var,
            from_=-720.0,
            to=720.0,
            increment=self._euler_step_deg,
            format="%.2f",
            width=12,
        ).grid(row=2, column=1, sticky="w")
        ttk.Label(point_box, text="PC x").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            point_box,
            textvariable=self.pcx_var,
            from_=-2.0,
            to=2.0,
            increment=self._pc_step,
            format="%.3f",
            width=12,
        ).grid(row=3, column=1, sticky="w", pady=(6, 0))
        ttk.Label(point_box, text="PC y").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(
            point_box,
            textvariable=self.pcy_var,
            from_=-2.0,
            to=2.0,
            increment=self._pc_step,
            format="%.3f",
            width=12,
        ).grid(row=4, column=1, sticky="w")
        ttk.Label(point_box, text="PC z").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(
            point_box,
            textvariable=self.pcz_var,
            from_=-2.0,
            to=2.0,
            increment=self._pc_step,
            format="%.3f",
            width=12,
        ).grid(row=5, column=1, sticky="w")
        ttk.Label(point_box, text="steps: Euler 0.01 deg, PC 0.001").grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(point_box, textvariable=self.pc_conv_label_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(point_box, text="Read Selected Point Values", command=self._load_selected_point_values).grid(
            row=8, column=0, columnspan=2, sticky="we", pady=(6, 0)
        )
        ttk.Button(point_box, text="Apply Edited Values", command=self._apply_selected_point_values).grid(
            row=9, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Button(
            point_box,
            text="Apply This PC To Map (No Averaging)",
            command=self._apply_selected_pc_to_full_map,
        ).grid(row=10, column=0, columnspan=2, sticky="we", pady=(4, 0))

        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.X, pady=4)
        self.workflow_notebook = notebook

        tab_refine = ttk.Frame(notebook, padding=6)
        tab_index = ttk.Frame(notebook, padding=6)
        tab_overlap = ttk.Frame(notebook, padding=6)
        notebook.add(tab_refine, text="Step 1: PC Refine")
        notebook.add(tab_index, text="Step 2: Re-index")
        notebook.add(tab_overlap, text="Step 3: Overlap")

        self._build_refine_tab(tab_refine)
        self._build_index_tab(tab_index)
        self._build_overlap_tab(tab_overlap)
        notebook.bind("<<NotebookTabChanged>>", lambda _e: self._refresh_plot())

        ttk.Label(parent, textvariable=self.status_var, wraplength=520).pack(fill=tk.X, pady=4)

        info_box = ttk.LabelFrame(parent, text="Info", padding=4)
        info_box.pack(fill=tk.X, pady=(2, 4))
        self.info_text = tk.Text(info_box, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, expand=False)
        self._set_info_lines(["Load data and master pattern to start."])

        log_box = ttk.LabelFrame(parent, text="Log", padding=4)
        log_box.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_box, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_refine_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="trust euler (deg)").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.trust_euler_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(parent, text="trust PC").grid(row=1, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.trust_pc_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(parent, text="maxfev").grid(row=2, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.maxfev_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Button(parent, text="Add Selected Calibration Point", command=self._add_calibration_point).grid(
            row=3, column=0, columnspan=2, sticky="we", pady=(6, 0)
        )
        ttk.Button(parent, text="Remove Selected Calibration Point", command=self._remove_calibration_point).grid(
            row=4, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Button(parent, text="Clear Calibration Points", command=self._clear_calibration_points).grid(
            row=5, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Label(parent, textvariable=self.calibration_summary_var, wraplength=460).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Button(parent, text="Optimize Selected Point (orientation + PC)", command=self._refine_selected_point).grid(
            row=7, column=0, columnspan=2, sticky="we", pady=(6, 0)
        )
        self.btn_refine_roi = ttk.Button(parent, text="Optimize All Calibration Points", command=self._refine_calibration_points)
        self.btn_refine_roi.grid(row=8, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Checkbutton(
            parent,
            text="Correct PC for scan position with Kikuchipy",
            variable=self.use_scan_pc_shift_var,
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(parent, text="detector pixel size (scan-step unit)").grid(row=10, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.detector_px_size_var, width=8).grid(row=10, column=1, sticky="w")
        ttk.Label(parent, text="detector binning").grid(row=11, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.detector_binning_var, width=8).grid(row=11, column=1, sticky="w")
        ttk.Button(parent, text="Average PCs and Apply to Map", command=self._apply_average_calibration_pc).grid(
            row=12, column=0, columnspan=2, sticky="we", pady=(8, 0)
        )
        ttk.Label(
            parent,
            text="For maps below ~50 µm, compare the reported max |dPC|; the correction may be negligible.",
            wraplength=460,
        ).grid(row=13, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_index_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="orientation resolution (deg)").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.di_res_deg_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(parent, text="software pattern binning").grid(row=1, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.di_binning_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Button(parent, text="Generate Binned Dictionary", command=self._generate_dictionary).grid(
            row=2, column=0, columnspan=2, sticky="we", pady=(6, 0)
        )
        self.dictionary_progress_bar = ttk.Progressbar(
            parent,
            variable=self.dictionary_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.dictionary_progress_bar.grid(row=3, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Label(parent, textvariable=self.dictionary_status_var, wraplength=390).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Entry(parent, textvariable=self.dictionary_path_var, width=36).grid(row=5, column=0, sticky="we", pady=(6, 0))
        ttk.Button(parent, text="Browse", command=self._browse_dictionary).grid(row=5, column=1, sticky="we", padx=(4, 0), pady=(6, 0))
        ttk.Button(parent, text="Save Dictionary", command=self._save_dictionary).grid(row=6, column=0, sticky="we", pady=(4, 0))
        ttk.Button(parent, text="Load Dictionary", command=self._load_dictionary).grid(row=6, column=1, sticky="we", pady=(4, 0))
        ttk.Button(parent, text="Load Indexed Data", command=self._restore_workflow).grid(
            row=7, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=8, column=0, columnspan=2, sticky="we", pady=8)
        ttk.Label(parent, text="keep_n (top matches)").grid(row=9, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.dictionary_keep_n_var, width=8).grid(row=9, column=1, sticky="w")
        ttk.Button(parent, text="Index Selected Point (NCC)", command=self._index_selected_point).grid(row=10, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.btn_index_roi = ttk.Button(parent, text="Re-index ROI (NCC)", command=self._index_roi)
        self.btn_index_roi.grid(row=11, column=0, columnspan=2, sticky="we", pady=(4, 0))
        self.btn_index_full = ttk.Button(parent, text="Re-index Full Map (NCC)", command=self._index_full)
        self.btn_index_full.grid(row=12, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Progressbar(
            parent,
            variable=self.reindex_progress_var,
            maximum=100.0,
            mode="determinate",
        ).grid(row=13, column=0, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Label(
            parent,
            textvariable=self.reindex_progress_status_var,
            wraplength=390,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(2, 0))
        parent.columnconfigure(0, weight=1)

    def _build_overlap_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            parent,
            text="Fit Gaussian blur + elliptical power-law gain mask",
            variable=self.fit_blur_gain_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(parent, text="manual blur sigma (fit disabled)").grid(row=1, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.blur_sigma_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(parent, text="gain-fit max iterations").grid(row=2, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gain_fit_maxiter_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(parent, text="gain-fit population size").grid(row=3, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gain_fit_popsize_var, width=8).grid(row=3, column=1, sticky="w")
        ttk.Label(parent, text="residual refinement trust (deg)").grid(row=4, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.residual_trust_euler_var, width=8).grid(row=4, column=1, sticky="w")
        ttk.Label(parent, text="residual refinement max evaluations").grid(row=5, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.residual_maxfev_var, width=8).grid(row=5, column=1, sticky="w")
        ttk.Label(parent, text="keep_n (top matches)").grid(row=6, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.residual_keep_n_var, width=8).grid(row=6, column=1, sticky="w")
        ttk.Button(parent, text="Fit Selected Point and Build Residual", command=self._analyze_overlap).grid(row=7, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Button(parent, text="Index Residual for Selected Point", command=self._index_overlap_residual).grid(
            row=8, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Button(parent, text="Refine Residual for Selected Point", command=self._refine_overlap_residual).grid(
            row=9, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=10, column=0, columnspan=2, sticky="we", pady=8)
        ttk.Label(parent, text="Full ROI residual workflow (uses the shared ROI selection)").grid(
            row=11, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(parent, text="minimum primary NCC for residual work").grid(row=12, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.overlap_min_ncc_var, width=8).grid(row=12, column=1, sticky="w")
        ttk.Checkbutton(
            parent,
            text="Write residual patterns to file",
            variable=self.write_residual_patterns_var,
        ).grid(row=13, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(parent, text="residual pattern file").grid(row=14, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Entry(parent, textvariable=self.residual_pattern_path_var, width=34).grid(row=15, column=0, sticky="we", pady=(2, 0))
        ttk.Button(parent, text="Browse", command=self._browse_residual_pattern_output).grid(
            row=15, column=1, sticky="we", padx=(4, 0), pady=(2, 0)
        )
        ttk.Button(parent, text="Compute Residuals for ROI", command=self._compute_overlap_residual_roi).grid(
            row=16, column=0, columnspan=2, sticky="we", pady=(6, 0)
        )
        ttk.Button(parent, text="Index Residual ROI", command=self._index_overlap_residual_roi).grid(
            row=17, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Button(parent, text="Refine Residual ROI", command=self._refine_overlap_residual_roi).grid(
            row=18, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Label(parent, text="residual IPF white threshold (KP NCC)").grid(row=19, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.residual_ipf_ncc_var, width=8).grid(row=19, column=1, sticky="w")
        self.overlap_progress_bar = ttk.Progressbar(parent, variable=self.overlap_progress_var, maximum=100.0, mode="determinate")
        self.overlap_progress_bar.grid(row=20, column=0, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Label(parent, textvariable=self.overlap_progress_status_var, wraplength=390).grid(
            row=21, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        bounds_box = ttk.LabelFrame(parent, text="Primary fit bounds", padding=8)
        bounds_box.grid(row=22, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(
            bounds_box,
            text="Used when Gaussian blur + gain are fitted. These defaults mirror the reference script and can be edited here.",
            wraplength=380,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        for row, (label, low_var, high_var) in enumerate(self.primary_fit_bound_specs, start=1):
            ttk.Label(bounds_box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
            ttk.Entry(bounds_box, textvariable=low_var, width=8).grid(row=row, column=1, sticky="w")
            ttk.Entry(bounds_box, textvariable=high_var, width=8).grid(row=row, column=2, sticky="w", padx=(6, 0))

        export_box = ttk.LabelFrame(parent, text="Export ROI indexing results", padding=8)
        export_box.grid(row=23, column=0, columnspan=2, sticky="we", pady=(10, 0))
        export_box.columnconfigure(0, weight=1)
        ttk.Label(export_box, text="primary ROI export").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(export_box, textvariable=self.primary_roi_export_path_var, width=34).grid(
            row=1, column=0, sticky="we", pady=(2, 0)
        )
        ttk.Button(export_box, text="Browse", command=self._browse_primary_roi_export).grid(
            row=1, column=1, sticky="we", padx=(4, 0), pady=(2, 0)
        )
        ttk.Button(export_box, text="Export Primary ROI Map", command=self._export_primary_roi_map).grid(
            row=2, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )
        ttk.Separator(export_box, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=2, sticky="we", pady=8)
        ttk.Label(export_box, text="residual ROI export").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Entry(export_box, textvariable=self.residual_roi_export_path_var, width=34).grid(
            row=5, column=0, sticky="we", pady=(2, 0)
        )
        ttk.Button(export_box, text="Browse", command=self._browse_residual_roi_export).grid(
            row=5, column=1, sticky="we", padx=(4, 0), pady=(2, 0)
        )
        ttk.Button(export_box, text="Export Residual ROI Map", command=self._export_residual_roi_map).grid(
            row=6, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )

    def _build_overlap_optimization_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="shared fit max iterations").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gain_fit_maxiter_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(parent, text="shared fit population size").grid(row=1, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.gain_fit_popsize_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(parent, text="minimum residual NCC").grid(row=2, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.overlap_mixture_residual_ncc_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(parent, text="orientation trust (deg)").grid(row=3, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.overlap_mixture_trust_euler_var, width=8).grid(row=3, column=1, sticky="w")
        ttk.Label(parent, text="orientation max evaluations").grid(row=4, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.overlap_mixture_maxfev_var, width=8).grid(row=4, column=1, sticky="w")
        ttk.Button(parent, text="Fit Selected Point Mixture", command=self._fit_overlap_mixture).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="we",
            pady=(8, 0),
        )
        ttk.Button(parent, text="Fit Mixture for ROI", command=self._fit_overlap_mixture_roi).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="we",
            pady=(4, 0),
        )
        ttk.Button(parent, text="Refine Selected Mixture Orientations", command=self._refine_overlap_mixture_orientations).grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="we",
            pady=(4, 0),
        )
        ttk.Progressbar(
            parent,
            variable=self.overlap_optimization_progress_var,
            maximum=100.0,
            mode="determinate",
        ).grid(row=8, column=0, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Label(parent, textvariable=self.overlap_optimization_status_var, wraplength=390).grid(
            row=9,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(2, 0),
        )

        bounds_box = ttk.LabelFrame(parent, text="Primary fit bounds", padding=8)
        bounds_box.grid(row=10, column=0, columnspan=2, sticky="we", pady=(10, 0))
        for row, (label, low_var, high_var) in enumerate(self.primary_fit_bound_specs):
            ttk.Label(bounds_box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
            ttk.Entry(bounds_box, textvariable=low_var, width=8).grid(row=row, column=1, sticky="w")
            ttk.Entry(bounds_box, textvariable=high_var, width=8).grid(row=row, column=2, sticky="w", padx=(6, 0))

    def _build_right(self, parent: ttk.Frame) -> None:
        top_bar = ttk.Frame(parent)
        top_bar.pack(fill=tk.X)
        ttk.Label(top_bar, text="Map layer").pack(side=tk.LEFT)
        self.map_layer_combo = ttk.Combobox(
            top_bar,
            textvariable=self.map_layer_var,
            values=[ORIENTATION_LAYER_LABEL, "Phase"],
            state="readonly",
            width=18,
        )
        self.map_layer_combo.pack(side=tk.LEFT, padx=4)
        self.map_layer_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plot())
        ttk.Button(top_bar, text="Refresh", command=self._refresh_plot).pack(side=tk.LEFT, padx=4)

        self.figure = Figure(figsize=(10.5, 8.5), dpi=100)
        self.axes = self.figure.subplots(2, 3)
        for ax in self.axes.flat:
            ax.set_axis_off()
        self._safe_tight_layout(self.figure)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self._draw_instruction("Load data and master pattern to start.")

    # ------------------------ UI handlers ------------------------ #

    def _set_busy(self, flag: bool) -> None:
        self.busy = bool(flag)

    def _attach_point_value_traces(self) -> None:
        vars_to_watch = (
            self.euler1_deg_var,
            self.euler2_deg_var,
            self.euler3_deg_var,
            self.pcx_var,
            self.pcy_var,
            self.pcz_var,
            self.roi_r0_var,
            self.roi_c0_var,
            self.roi_nrows_var,
            self.roi_ncols_var,
        )
        for var in vars_to_watch:
            var.trace_add("write", self._on_point_values_changed)

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

    def _browse_patterns(self) -> None:
        fn = filedialog.askopenfilename(filetypes=[("Pattern files", "*.h5oina *.up1 *.up2"), ("All files", "*.*")])
        if fn:
            self.pattern_path_var.set(str(Path(fn).resolve()))

    def _browse_orientation(self) -> None:
        fn = filedialog.askopenfilename(filetypes=[("ANG files", "*.ang"), ("All files", "*.*")])
        if fn:
            self.orientation_path_var.set(str(Path(fn).resolve()))

    def _browse_master(self) -> None:
        fn = filedialog.askopenfilename(filetypes=[("Master patterns", "*.h5 *.hdf5 *.sdf5"), ("All files", "*.*")])
        if fn:
            self.master_path_var.set(str(Path(fn).resolve()))

    def _browse_export(self) -> None:
        fn = filedialog.asksaveasfilename(defaultextension=".h5oina", filetypes=[("All files", "*.*")])
        if fn:
            self.export_path_var.set(str(Path(fn).resolve()))

    def _default_residual_pattern_path(self) -> str:
        pattern = self.pattern_path_var.get().strip()
        if pattern:
            src = Path(pattern).expanduser().resolve()
            ext = src.suffix.lower()
            if ext in {".h5oina", ".up1", ".up2"}:
                return str(src.with_name(f"{src.stem}_residuals{ext}"))
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
            if self.session.data.source_type == "h5oina":
                source = Path(self.session.data.pattern_path)
                suffix = ".h5oina"
            elif self.session.data.source_type == "up_ang":
                source_path = self.session.data.orientation_path or self.orientation_path_var.get().strip()
                source = Path(source_path) if source_path else None
                suffix = ".ang"
        if source is not None and source.name:
            return str(source.with_name(f"{source.stem}_{tag}_roi{suffix}").resolve())
        return str((Path.cwd() / f"{tag}_roi{suffix}").resolve())

    def _browse_roi_export(self, var: tk.StringVar, *, residual: bool) -> None:
        current = Path(var.get().strip() or self._default_roi_export_path(residual=residual))
        ext = current.suffix.lower()
        if ext not in {".ang", ".h5oina"}:
            ext = ".h5oina" if (self.session.data is None or self.session.data.source_type == "h5oina") else ".ang"
        fn = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialfile=current.name,
            initialdir=str(current.parent),
            filetypes=[
                ("Pattern/orientation files", "*.ang *.h5oina"),
                ("ANG files", "*.ang"),
                ("H5OINA files", "*.h5oina"),
                ("All files", "*.*"),
            ],
        )
        if fn:
            var.set(str(Path(fn).resolve()))

    def _browse_primary_roi_export(self) -> None:
        self._browse_roi_export(self.primary_roi_export_path_var, residual=False)

    def _browse_residual_roi_export(self) -> None:
        self._browse_roi_export(self.residual_roi_export_path_var, residual=True)

    def _browse_dictionary(self) -> None:
        fn = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("Binned EBSD dictionary", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if fn:
            self.dictionary_path_var.set(str(Path(fn).resolve()))

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

    def _set_dictionary_progress(self, value: float, message: str) -> None:
        self.dictionary_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.dictionary_status_var.set(message)

    def _set_reindex_progress(self, value: float, message: str) -> None:
        self.reindex_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.reindex_progress_status_var.set(message)

    def _set_refinement_progress(self, value: float, message: str) -> None:
        self.refinement_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.refinement_progress_status_var.set(message)

    def _set_overlap_progress(self, value: float, message: str) -> None:
        self.overlap_progress_var.set(float(np.clip(value, 0.0, 100.0)))
        self.overlap_progress_status_var.set(message)

    def _set_overlap_optimization_progress(self, value: float, message: str) -> None:
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
            return 0.0

    def _residual_ipf_ncc_threshold(self) -> float:
        try:
            return float(str(self.residual_ipf_ncc_var.get()).strip())
        except Exception:
            return 0.0

    def _overlap_mixture_residual_ncc_threshold(self) -> float:
        try:
            return float(str(self.overlap_mixture_residual_ncc_var.get()).strip())
        except Exception:
            return 0.0

    def _primary_threshold_mask(self) -> np.ndarray | None:
        threshold = self._residual_ncc_threshold()
        if threshold <= 0.0 or self.session.data is None:
            return None
        score_map = self.session.last_scores_map
        if score_map is None or score_map.shape != (self.session.data.rows, self.session.data.cols):
            return None
        return np.isfinite(score_map) & (np.asarray(score_map, dtype=np.float32) < threshold)

    def _index_quality_layer_choices(self) -> list[str]:
        if self.session.data is None:
            return ["CI", "BC", "IQ", "BS", "DP", "NCC", "MAD", "Phase"]
        unavailable = {ORIENTATION_LAYER_LABEL, "X", "Y"}
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
        for view_index, plot_view in self._plot_views.items():
            if view_index == 1 and plot_view.get("combo") is not None:
                plot_view["combo"]["values"] = choices
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
        if score is not None:
            return float(score)
        if self.last_overlap is not None and self.last_overlap.index == int(index):
            try:
                value = float(self.last_overlap.ncc_es)
            except Exception:
                return None
            return value if np.isfinite(value) else None
        return None

    def _filter_roi_indices_by_threshold(self, indices: np.ndarray) -> tuple[np.ndarray, int]:
        threshold = self._residual_ncc_threshold()
        if threshold <= 0.0:
            return np.asarray(indices, dtype=np.int64).ravel(), 0
        filtered: list[int] = []
        skipped = 0
        for idx in np.asarray(indices, dtype=np.int64).ravel().tolist():
            score = self.session.get_primary_index_ncc(int(idx))
            if score is None or score >= threshold:
                filtered.append(int(idx))
            else:
                skipped += 1
        return np.asarray(filtered, dtype=np.int64), skipped

    def _update_dictionary_status(self) -> None:
        cache = self.session.dictionary_cache
        if cache is None:
            self.dictionary_status_var.set("No dictionary generated or loaded.")
            self.dictionary_progress_var.set(0.0)
            return
        self.dictionary_status_var.set(
            f"Ready: {cache.rotation_count} patterns, {cache.pattern_shape[0]}x{cache.pattern_shape[1]}, "
            f"binning={cache.software_binning}, resolution={cache.resolution_deg:g}°"
        )
        self.dictionary_progress_var.set(100.0)

    def _sync_residual_keep_n_to_dictionary(self, keep_n: int) -> None:
        self.residual_keep_n_var.set(max(1, int(keep_n)))

    def _run_threaded(self, fn) -> None:
        if self.busy:
            return
        try:
            self._sync_pattern_conditioning_settings()
            self.session.last_action_note = ""
        except Exception as exc:
            messagebox.showerror("Invalid pattern conditioning", str(exc))
            return
        self._set_busy(True)
        self.status_var.set("Running...")
        self._set_info_lines(["Running...", "Check the log for detailed step updates."])

        def worker() -> None:
            try:
                msg = fn()
                self.after(0, lambda msg=msg: self._on_action_done(msg))
            except Exception as exc:
                detail = traceback.format_exc()
                self.after(0, lambda exc=exc, detail=detail: self._on_action_error(exc, detail))

        threading.Thread(target=worker, daemon=True).start()

    def _update_mode_controls(self) -> None:
        state = tk.NORMAL
        for btn in (self.btn_refine_roi, self.btn_index_roi, self.btn_index_full, self.btn_refine_indexed):
            if btn is not None:
                try:
                    btn.configure(state=state)
                except Exception:
                    pass

    def _on_action_done(self, msg: str) -> None:
        self._set_busy(False)
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

    def _on_action_error(self, exc: Exception, detail: str) -> None:
        self._set_busy(False)
        self.status_var.set(f"Error: {exc}")
        self._log(detail)
        self._set_info_lines([f"Error: {exc}", "See Log for traceback details."])
        messagebox.showerror("Error", str(exc))

    def _load_input(self) -> None:
        def action() -> str:
            geom = GeometryConfig(
                pc_convention="edax",
                sample_tilt_deg=float(self.sample_tilt_var.get()),
                detector_tilt_deg=float(self.detector_tilt_var.get()),
                azimuthal_deg=0.0,
                twist_deg=0.0,
                phi1_offset_deg=0.0,
            )
            msg = self.session.load_input(
                pattern_path=self.pattern_path_var.get().strip(),
                orientation_path=self.orientation_path_var.get().strip() or None,
                geom=geom,
            )
            self.last_overlap = None
            self.last_overlap_mixture = None
            self.index_var.set(0)
            self.row_var.set(0)
            self.col_var.set(0)
            if self.session.data is not None:
                layers = self.session.available_layers() or ["Phase"]
                self._sync_index_quality_layer_choices()
                for view_index, plot_view in self._plot_views.items():
                    if view_index == 1:
                        plot_view["combo"]["values"] = self._index_quality_layer_choices()
                    else:
                        plot_view["combo"]["values"] = layers
                if self.map_layer_var.get() not in layers:
                    self.map_layer_var.set(layers[0])
                if self.index_quality_layer_var.get() not in self._index_quality_layer_choices():
                    self.index_quality_layer_var.set(self._default_index_quality_layer())
                self.roi_nrows_var.set(min(31, self.session.data.rows))
                self.roi_ncols_var.set(min(31, self.session.data.cols))
                unique_phases = np.unique(self.session.data.phases)
                if unique_phases.size > 0:
                    nonneg = unique_phases[unique_phases >= 0]
                    if nonneg.size > 0:
                        counts = [(int(ph), int(np.sum(self.session.data.phases == ph))) for ph in nonneg.tolist()]
                        counts.sort(key=lambda t: t[1], reverse=True)
                        chosen = counts[0][0]
                    else:
                        chosen = int(unique_phases[0])
                    self.phase_id_var.set(int(chosen))
                    msg = (
                        f"{msg} Auto-set phase ID to {int(chosen)}. "
                        f"Available phases: {unique_phases.tolist()}"
                    )
                if self.session.data.source_type == "up_ang":
                    msg = (
                        f"{msg} UP mode keeps patterns on disk and batches ROI/full-map indexing."
                    )
                self.residual_pattern_path_var.set(self._default_residual_pattern_path())
                self.primary_roi_export_path_var.set(self._default_roi_export_path(residual=False))
                self.residual_roi_export_path_var.set(self._default_roi_export_path(residual=True))
                if self.session.data.detector_px_size is not None:
                    self.detector_px_size_var.set(float(self.session.data.detector_px_size))
                if self.session.data.detector_binning is not None:
                    self.detector_binning_var.set(float(self.session.data.detector_binning))
                self._update_mode_controls()
                self._update_calibration_summary()
                self._populate_point_vars()
            return msg

        self._run_threaded(action)

    def _load_master(self) -> None:
        def action() -> str:
            return self.session.load_master(
                master_path=self.master_path_var.get().strip(),
                energy_kv=None,
            )

        self._run_threaded(action)

    def _export_results(self) -> None:
        def action() -> str:
            return self.session.export_reindexed_results(self.export_path_var.get().strip())

        self._run_threaded(action)

    def _save_workflow(self) -> None:
        fn = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("Overlap workflow", "*.npz"), ("All files", "*.*")],
        )
        if fn:
            self._run_threaded(lambda: self.session.save_workflow_state(fn))

    def _restore_workflow(self) -> None:
        fn = filedialog.askopenfilename(
            filetypes=[("Overlap workflow", "*.npz"), ("All files", "*.*")],
        )
        if not fn:
            return

        def action() -> str:
            msg = self.session.restore_workflow_state(fn)
            self.last_overlap = None
            self.last_overlap_mixture = None
            candidate_store = self.session.indexed_candidate_eulers_rad
            candidate_count = int(candidate_store.shape[1]) if candidate_store is not None else 1
            self.dictionary_keep_n_var.set(candidate_count)
            self._sync_residual_keep_n_to_dictionary(candidate_count)
            if self.session.data is not None:
                self.pattern_path_var.set(self.session.data.pattern_path)
                self.orientation_path_var.set(self.session.data.orientation_path or "")
                self.master_path_var.set(self.session.master.path if self.session.master is not None else "")
                self.pattern_mask_option_var.set(int(self.session.pattern_mask_option))
                self.dynamic_bg_enabled_var.set(bool(self.session.dynamic_bg_config.enabled))
                self.dynamic_bg_std_var.set(f"{float(self.session.dynamic_bg_config.std_px):g}")
                layers = self.session.available_layers()
                self._sync_index_quality_layer_choices()
                for view_index, plot_view in self._plot_views.items():
                    if view_index == 1:
                        plot_view["combo"]["values"] = self._index_quality_layer_choices()
                    else:
                        plot_view["combo"]["values"] = layers
                if self.map_layer_var.get() not in layers and layers:
                    self.map_layer_var.set(layers[0])
                if self.index_quality_layer_var.get() not in self._index_quality_layer_choices():
                    self.index_quality_layer_var.set(self._default_index_quality_layer())
                if self.session.data.detector_px_size is not None:
                    self.detector_px_size_var.set(float(self.session.data.detector_px_size))
                if self.session.data.detector_binning is not None:
                    self.detector_binning_var.set(float(self.session.data.detector_binning))
                self.residual_pattern_path_var.set(
                    self.session.residual_pattern_output_path or self._default_residual_pattern_path()
                )
                self.primary_roi_export_path_var.set(self._default_roi_export_path(residual=False))
                self.residual_roi_export_path_var.set(self._default_roi_export_path(residual=True))
            self._update_mode_controls()
            return msg

        self._run_threaded(action)

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
        self.roi_r0_var.set(r0)
        self.roi_c0_var.set(c0)
        self.roi_nrows_var.set(height)
        self.roi_ncols_var.set(width)
        self._refresh_plot()

    def _apply_roi_selection(self) -> None:
        if self.session.data is None:
            return
        self._refresh_plot()

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

    def _apply_selected_point_values(self) -> None:
        def action() -> str:
            idx = int(self.index_var.get())
            e1 = self._quantize(float(self.euler1_deg_var.get()), self._euler_step_deg)
            e2 = self._quantize(float(self.euler2_deg_var.get()), self._euler_step_deg)
            e3 = self._quantize(float(self.euler3_deg_var.get()), self._euler_step_deg)
            pcx = self._quantize(float(self.pcx_var.get()), self._pc_step)
            pcy = self._quantize(float(self.pcy_var.get()), self._pc_step)
            pcz = self._quantize(float(self.pcz_var.get()), self._pc_step)
            self._suspend_point_trace = True
            try:
                self.euler1_deg_var.set(e1)
                self.euler2_deg_var.set(e2)
                self.euler3_deg_var.set(e3)
                self.pcx_var.set(pcx)
                self.pcy_var.set(pcy)
                self.pcz_var.set(pcz)
            finally:
                self._suspend_point_trace = False
            return self.session.set_point_state(
                idx,
                euler_deg=(e1, e2, e3),
                pc_custom=(pcx, pcy, pcz),
            )

        self._run_threaded(action)

    # -------------------------- Step 1 -------------------------- #

    def _update_calibration_summary(self) -> None:
        if self.session.data is None:
            self.calibration_summary_var.set("No calibration points selected.")
            return
        self.calibration_summary_var.set(self.session.calibration_point_summary())

    def _add_calibration_point(self) -> None:
        try:
            msg = self.session.add_calibration_point(int(self.index_var.get()))
            self.status_var.set(msg)
            self._log(msg)
            self._update_calibration_summary()
            self._refresh_plot()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _remove_calibration_point(self) -> None:
        try:
            msg = self.session.remove_calibration_point(int(self.index_var.get()))
            self.status_var.set(msg)
            self._log(msg)
            self._update_calibration_summary()
            self._refresh_plot()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _clear_calibration_points(self) -> None:
        msg = self.session.clear_calibration_points()
        self.status_var.set(msg)
        self._log(msg)
        self._update_calibration_summary()
        self._refresh_plot()

    def _refine_selected_point(self) -> None:
        def action() -> str:
            idx = int(self.index_var.get())
            return self.session.refine_indices(
                indices=np.array([idx], dtype=np.int64),
                phase_id=int(self.phase_id_var.get()),
                trust_euler_deg=float(self.trust_euler_var.get()),
                trust_pc=float(self.trust_pc_var.get()),
                maxfev=int(self.maxfev_var.get()),
            )

        self._run_threaded(action)

    def _refine_calibration_points(self) -> None:
        def action() -> str:
            if not self.session.calibration_indices:
                raise ValueError("Add calibration points from the map first.")
            msg = self.session.refine_indices(
                indices=np.asarray(self.session.calibration_indices, dtype=np.int64),
                phase_id=int(self.phase_id_var.get()),
                trust_euler_deg=float(self.trust_euler_var.get()),
                trust_pc=float(self.trust_pc_var.get()),
                maxfev=int(self.maxfev_var.get()),
            )
            return f"{msg}\n{self.session.calibration_point_summary()}"

        self._run_threaded(action)

    def _apply_average_calibration_pc(self) -> None:
        def action() -> str:
            return self.session.apply_average_calibration_pc(
                use_scan_geometry=bool(self.use_scan_pc_shift_var.get()),
                detector_px_size=float(self.detector_px_size_var.get()),
                detector_binning=float(self.detector_binning_var.get()),
            )

        self._run_threaded(action)

    def _refine_roi(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()

        def action() -> str:
            idx = self.session.roi_indices(*bounds)
            return self.session.refine_indices(
                indices=idx,
                phase_id=int(self.phase_id_var.get()),
                trust_euler_deg=float(self.trust_euler_var.get()),
                trust_pc=float(self.trust_pc_var.get()),
                maxfev=int(self.maxfev_var.get()),
            )

        self._run_threaded(action)

    def _apply_selected_pc_to_full_map(self) -> None:
        def action() -> str:
            idx = int(self.index_var.get())
            return self.session.apply_point_pc_to_full_map(idx)

        self._run_threaded(action)

    # -------------------------- Step 2 -------------------------- #

    def _generate_dictionary(self) -> None:
        self._set_dictionary_progress(0.0, "Starting dictionary generation...")
        phase_id = int(self.phase_id_var.get())
        resolution_deg = float(self.di_res_deg_var.get())
        software_binning = int(self.di_binning_var.get())

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_dictionary_progress(v, m))

        def action() -> str:
            return self.session.generate_dictionary(
                phase_id=phase_id,
                resolution_deg=resolution_deg,
                software_binning=software_binning,
                progress_callback=progress,
            )

        self._run_threaded(action)

    def _save_dictionary(self) -> None:
        path = self.dictionary_path_var.get().strip()
        self._run_threaded(lambda: self.session.save_dictionary(path))

    def _load_dictionary(self) -> None:
        path = self.dictionary_path_var.get().strip()
        if not path or not Path(path).exists():
            selected = filedialog.askopenfilename(
                filetypes=[("Binned EBSD dictionary", "*.h5 *.hdf5"), ("All files", "*.*")],
            )
            if not selected:
                return
            path = str(Path(selected).resolve())
            self.dictionary_path_var.set(path)

        def action() -> str:
            msg = self.session.load_dictionary(path)
            cache = self.session.dictionary_cache
            if cache is not None:
                self.after(
                    0,
                    lambda: (
                        self.phase_id_var.set(cache.phase_id),
                        self.di_res_deg_var.set(cache.resolution_deg),
                        self.di_binning_var.set(cache.software_binning),
                    ),
                )
            return msg

        self._run_threaded(action)

    def _index_selected_point(self) -> None:
        idx = int(self.index_var.get())
        phase_id = int(self.phase_id_var.get())
        resolution_deg = float(self.di_res_deg_var.get())
        keep_n = max(1, int(self.dictionary_keep_n_var.get()))
        self._sync_residual_keep_n_to_dictionary(keep_n)
        self._set_reindex_progress(0.0, "Starting selected-point re-indexing...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_reindex_progress(v, m))

        def action() -> str:
            return self.session.dictionary_index_indices(
                indices=np.array([idx], dtype=np.int64),
                phase_id=phase_id,
                keep_n=keep_n,
                resolution_deg=resolution_deg,
                progress_callback=progress,
            )

        self._run_threaded(action)

    def _index_roi(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        phase_id = int(self.phase_id_var.get())
        resolution_deg = float(self.di_res_deg_var.get())
        keep_n = max(1, int(self.dictionary_keep_n_var.get()))
        self._sync_residual_keep_n_to_dictionary(keep_n)
        self._set_reindex_progress(0.0, "Starting ROI re-indexing...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_reindex_progress(v, m))

        def action() -> str:
            idx = self.session.roi_indices(*bounds)
            return self.session.dictionary_index_indices(
                indices=idx,
                phase_id=phase_id,
                keep_n=keep_n,
                resolution_deg=resolution_deg,
                progress_callback=progress,
            )

        self._run_threaded(action)

    def _index_full(self) -> None:
        phase_id = int(self.phase_id_var.get())
        resolution_deg = float(self.di_res_deg_var.get())
        keep_n = max(1, int(self.dictionary_keep_n_var.get()))
        self._sync_residual_keep_n_to_dictionary(keep_n)
        self._set_reindex_progress(0.0, "Starting full-map re-indexing...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_reindex_progress(v, m))

        def action() -> str:
            if self.session.data is None:
                raise RuntimeError("Load input data first.")
            idx = np.arange(self.session.data.count, dtype=np.int64)
            return self.session.dictionary_index_indices(
                indices=idx,
                phase_id=phase_id,
                keep_n=keep_n,
                resolution_deg=resolution_deg,
                progress_callback=progress,
            )

        self._run_threaded(action)

    def _refine_last_indexed(self) -> None:
        phase_id = int(self.phase_id_var.get())
        trust_euler = float(self.trust_euler_var.get())
        maxfev = int(self.maxfev_var.get())
        self._set_refinement_progress(0.0, "Starting orientation refinement...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_refinement_progress(v, m))

        def action() -> str:
            indices = self.session.last_indexed_indices
            if indices is None or indices.size == 0:
                raise ValueError("Run dictionary indexing on a point, ROI, or full map first.")
            return self.session.refine_orientations_indices(
                indices,
                phase_id=phase_id,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
                progress_callback=progress,
            )

        self._run_threaded(action)

    # -------------------------- Step 3 -------------------------- #

    def _analyze_overlap(self) -> None:
        index = int(self.index_var.get())
        blur_sigma = float(self.blur_sigma_var.get())
        fit_blur_gain = bool(self.fit_blur_gain_var.get())
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
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
            )
            self.last_overlap = result
            self.last_overlap_mixture = None
            resid_rms = float(np.sqrt(np.mean(np.square(result.residual))))
            self.after(0, lambda: self._set_overlap_progress(100.0, "Selected-point residual fit complete."))
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

        self._run_threaded(action)

    def _index_overlap_residual(self) -> None:
        index = int(self.index_var.get())
        blur_sigma = float(self.blur_sigma_var.get())
        keep_n = max(1, int(self.residual_keep_n_var.get()))
        residual_result = self.last_overlap
        threshold = self._residual_ncc_threshold()
        primary_ncc = self._selected_primary_ncc(index)
        if residual_result is None or residual_result.index != index:
            messagebox.showerror(
                "Residual unavailable",
                "Fit the primary pattern and build the residual for this point before indexing it.",
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
            result = self.session.index_overlap_residual(
                index,
                blur_sigma=blur_sigma,
                keep_n=keep_n,
                residual_result=residual_result,
            )
            self.last_overlap = result
            self.last_overlap_mixture = None
            self.after(0, lambda: self._set_overlap_progress(100.0, "Selected-point residual indexed."))
            return (
                f"Indexed residual at idx={result.index} with the step 2 dictionary: "
                f"KP NCC={result.secondary_ncc_kp:.4f}, full-pattern NCC={result.secondary_ncc_full:.4f}, "
                f"keep_n={keep_n}."
            )

        self._run_threaded(action)

    def _refine_overlap_residual(self) -> None:
        index = int(self.index_var.get())
        result = self.last_overlap
        if result is None or result.index != index or result.secondary_euler_rad is None:
            messagebox.showerror(
                "Residual match unavailable",
                "Build and index the residual for the selected point before refinement.",
            )
            return
        trust_euler = float(self.residual_trust_euler_var.get())
        maxfev = int(self.residual_maxfev_var.get())
        threshold = self._residual_ncc_threshold()
        primary_ncc = self._selected_primary_ncc(index)
        if primary_ncc is not None and threshold > 0.0 and primary_ncc < threshold:
            messagebox.showinfo(
                "Below NCC threshold",
                f"Primary NCC {primary_ncc:.4f} is below the minimum {threshold:.4f}; residual refinement is skipped.",
            )
            return
        self._set_overlap_progress(0.0, "Refining the selected-point residual match...")

        def action() -> str:
            refined = self.session.refine_overlap_residual(
                result,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
            )
            self.last_overlap = refined
            self.last_overlap_mixture = None
            self.after(0, lambda: self._set_overlap_progress(100.0, "Selected-point residual refined."))
            return (
                f"Refined residual orientation at idx={refined.index}: "
                f"KP NCC={refined.secondary_ncc_kp:.4f}, full-pattern NCC={refined.secondary_ncc_full:.4f}. "
                f"{refined.secondary_refinement_note}"
            )

        self._run_threaded(action)

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
                "Below NCC threshold",
                f"No ROI points meet the minimum primary NCC of {threshold:.3f} for residual work.",
            )
            return
        selected_index = int(self.index_var.get())
        fit_blur_gain = bool(self.fit_blur_gain_var.get())
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
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
            self.after(0, lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.compute_overlap_residual_indices(
                indices,
                fit_blur_gain=fit_blur_gain,
                fit_maxiter=fit_maxiter,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
                write_patterns=write_patterns,
                residual_output_path=residual_output_path if write_patterns else None,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
            )
            if np.any(indices == selected_index):
                selected_result = self.session.get_residual_point_result(selected_index)
                if selected_result is not None:
                    self.last_overlap = selected_result
            self.last_overlap_mixture = None
            self.after(0, lambda: self._set_overlap_progress(100.0, "Residuals computed for the ROI."))
            skipped_note = f" Skipped {skipped} point(s) below NCC {threshold:.3f}." if threshold > 0.0 and skipped > 0 else ""
            return f"{msg}{skipped_note} ROI bounds r0={bounds[0]}, c0={bounds[1]}, nrows={bounds[2]}, ncols={bounds[3]}."

        self._run_threaded(action)

    def _index_overlap_residual_roi(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        if self.session.dictionary_cache is None:
            messagebox.showerror("Error", "Generate or load a dictionary in tab 2 first.")
            return
        bounds = self._roi_bounds()
        indices = self.session.roi_indices(*bounds)
        threshold = self._residual_ncc_threshold()
        indices, skipped = self._filter_roi_indices_by_threshold(indices)
        if indices.size == 0:
            messagebox.showinfo(
                "Below NCC threshold",
                f"No ROI points meet the minimum primary NCC of {threshold:.3f} for residual work.",
            )
            return
        selected_index = int(self.index_var.get())
        write_patterns = bool(self.write_residual_patterns_var.get())
        keep_n = max(1, int(self.residual_keep_n_var.get()))
        self._set_overlap_progress(0.0, f"Indexing residuals for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.index_overlap_residual_indices(
                indices,
                keep_n=keep_n,
                write_patterns=write_patterns,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
            )
            if np.any(indices == selected_index):
                selected_result = self.session.get_residual_point_result(selected_index)
                if selected_result is not None:
                    self.last_overlap = selected_result
            self.last_overlap_mixture = None
            self.after(0, lambda: self._set_overlap_progress(100.0, "Residual ROI indexing complete."))
            skipped_note = f" Skipped {skipped} point(s) below NCC {threshold:.3f}." if threshold > 0.0 and skipped > 0 else ""
            return f"{msg}{skipped_note}"

        self._run_threaded(action)

    def _refine_overlap_residual_roi(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        if self.session.dictionary_cache is None:
            messagebox.showerror("Error", "Generate or load a dictionary in tab 2 first.")
            return
        bounds = self._roi_bounds()
        indices = self.session.roi_indices(*bounds)
        threshold = self._residual_ncc_threshold()
        indices, skipped = self._filter_roi_indices_by_threshold(indices)
        if indices.size == 0:
            messagebox.showinfo(
                "Below NCC threshold",
                f"No ROI points meet the minimum primary NCC of {threshold:.3f} for residual work.",
            )
            return
        selected_index = int(self.index_var.get())
        trust_euler = float(self.residual_trust_euler_var.get())
        maxfev = int(self.residual_maxfev_var.get())
        write_patterns = bool(self.write_residual_patterns_var.get())
        self._set_overlap_progress(0.0, f"Refining residuals for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_overlap_progress(v, m))

        def action() -> str:
            msg = self.session.refine_overlap_residual_indices(
                indices,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
                write_patterns=write_patterns,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
            )
            if np.any(indices == selected_index):
                selected_result = self.session.get_residual_point_result(selected_index)
                if selected_result is not None:
                    self.last_overlap = selected_result
            self.last_overlap_mixture = None
            self.after(0, lambda: self._set_overlap_progress(100.0, "Residual ROI refinement complete."))
            skipped_note = f" Skipped {skipped} point(s) below NCC {threshold:.3f}." if threshold > 0.0 and skipped > 0 else ""
            return f"{msg} trust Euler={trust_euler:g}°, maxfev={maxfev}.{skipped_note}"

        self._run_threaded(action)

    def _fit_overlap_mixture(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        index = int(self.index_var.get())
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        residual_result = self.last_overlap if self.last_overlap is not None and self.last_overlap.index == index else None
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
            result = self.session.fit_overlap_mixture_point(
                index,
                residual_result=residual_result,
                fit_maxiter=fit_maxiter,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
            )
            self.last_overlap_mixture = result
            self.after(0, lambda: self._set_overlap_optimization_progress(100.0, "Selected-point mixture fit complete."))
            return (
                f"idx={result.index} overlap mixture: primary={result.primary_fraction:.3f}, "
                f"secondary={result.secondary_fraction:.3f}, NCC={result.ncc_mixture:.4f}, "
                f"RMS={result.residual_rms:.4f}."
            )

        self._run_threaded(action)

    def _refine_overlap_mixture_orientations(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        index = int(self.index_var.get())
        result = (
            self.last_overlap_mixture
            if self.last_overlap_mixture is not None and self.last_overlap_mixture.index == index
            else self.session.get_overlap_mixture_result(index)
        )
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
            refined = self.session.refine_overlap_mixture_orientations(
                result,
                trust_euler_deg=trust_euler,
                maxfev=maxfev,
            )
            self.last_overlap_mixture = refined
            if refined.orientation_refined:
                self.last_overlap = None
            self.after(0, lambda: self._set_overlap_optimization_progress(100.0, "Selected mixture orientation refinement complete."))
            initial = refined.initial_mixture_ncc if refined.initial_mixture_ncc is not None else float("nan")
            return (
                f"Refined mixture orientations at idx={refined.index}: NCC {initial:.4f} -> "
                f"{refined.ncc_mixture:.4f}, primary={refined.primary_fraction:.3f}, "
                f"residual={refined.secondary_fraction:.3f}. {refined.orientation_refinement_note}"
            )

        self._run_threaded(action)

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
        fit_maxiter = int(self.gain_fit_maxiter_var.get())
        fit_popsize = int(self.gain_fit_popsize_var.get())
        try:
            fit_bounds = self._primary_fit_bounds()
        except Exception as exc:
            messagebox.showerror("Invalid fit bounds", str(exc))
            return
        self._set_overlap_optimization_progress(0.0, f"Fitting overlap mixtures for {indices.size} ROI point(s)...")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda v=value, m=message: self._set_overlap_optimization_progress(v, m))

        def action() -> str:
            msg = self.session.compute_overlap_mixture_indices(
                indices,
                fit_maxiter=fit_maxiter,
                fit_popsize=fit_popsize,
                fit_bounds=fit_bounds,
                selected_index=selected_index if np.any(indices == selected_index) else None,
                progress_callback=progress,
            )
            selected_result = self.session.get_overlap_mixture_result(selected_index)
            if selected_result is not None:
                self.last_overlap_mixture = selected_result
            self.after(0, lambda: self._set_overlap_optimization_progress(100.0, "Overlap mixture ROI fit complete."))
            threshold_note = (
                f" Skipped {skipped_low_residual} point(s) below residual NCC {threshold:.3f}."
                if threshold > 0.0 and skipped_low_residual > 0
                else ""
            )
            return f"{msg}{threshold_note} ROI bounds r0={bounds[0]}, c0={bounds[1]}, nrows={bounds[2]}, ncols={bounds[3]}."

        self._run_threaded(action)

    def _export_primary_roi_map(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        output_path = self.primary_roi_export_path_var.get().strip() or self._default_roi_export_path(residual=False)
        self._set_overlap_progress(0.0, "Exporting primary ROI map...")

        def action() -> str:
            msg = self.session.export_primary_roi_results(bounds, output_path)
            self.after(0, lambda: self._set_overlap_progress(100.0, "Primary ROI export complete."))
            return msg

        self._run_threaded(action)

    def _export_residual_roi_map(self) -> None:
        if self.session.data is None:
            messagebox.showerror("Error", "Load input data first.")
            return
        bounds = self._roi_bounds()
        output_path = self.residual_roi_export_path_var.get().strip() or self._default_roi_export_path(residual=True)
        threshold = self._residual_ncc_threshold()
        self._set_overlap_progress(0.0, "Exporting residual ROI map...")

        def action() -> str:
            msg = self.session.export_residual_roi_results(
                bounds,
                output_path,
                primary_ncc_threshold=threshold,
            )
            self.after(0, lambda: self._set_overlap_progress(100.0, "Residual ROI export complete."))
            return msg

        self._run_threaded(action)

    def _show_current_overlap_inspection(self) -> None:
        index = int(self.index_var.get())
        if self.last_overlap is None or self.last_overlap.index != index:
            messagebox.showinfo(
                "No inspection available",
                "Fit and index the selected residual point first.",
            )
            return
        self._show_overlap_inspection(self.last_overlap)

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
        axes[0, 1].set_title(f"Primary simulated pattern — NCC={result.ncc_es:.4f}")
        rabs = max(float(np.max(np.abs(result.residual))), 1e-8)
        axes[1, 0].imshow(result.residual, cmap="bwr", vmin=-rabs, vmax=rabs)
        axes[1, 0].set_title(f"Residual: Zexp − {result.scale:.4f}·Zsim")
        if result.secondary_simulated is not None:
            axes[1, 1].imshow(result.secondary_simulated, cmap="gray")
            if result.secondary_refined:
                seed = result.secondary_dictionary_ncc_kp
                seed_note = f"; binned dictionary seed={seed:.4f}" if seed is not None else ""
                axes[1, 1].set_title(
                    f"Full-resolution refined residual match — KP NCC={result.secondary_ncc_kp:.4f}{seed_note}"
                )
            else:
                axes[1, 1].set_title(f"Binned dictionary residual match — KP NCC={result.secondary_ncc_kp:.4f}")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Index the residual with the tab 2 dictionary",
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
                self._quantize(float(self.euler1_deg_var.get()), self._euler_step_deg),
                self._quantize(float(self.euler2_deg_var.get()), self._euler_step_deg),
                self._quantize(float(self.euler3_deg_var.get()), self._euler_step_deg),
            ],
            dtype=np.float64,
        )
        pc_custom = np.array(
            [
                self._quantize(float(self.pcx_var.get()), self._pc_step),
                self._quantize(float(self.pcy_var.get()), self._pc_step),
                self._quantize(float(self.pcz_var.get()), self._pc_step),
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

    def _refresh_plot(self) -> None:
        view_index = 0
        if self.workflow_notebook is not None:
            view_index = int(self.workflow_notebook.index(self.workflow_notebook.select()))
        self._activate_plot_view(view_index)
        if not self.busy:
            self._sync_pattern_conditioning_for_refresh()
        view = self._plot_views[view_index]
        if self.session.data is None:
            self._draw_instruction("Load data and master pattern to start.")
            self._set_info_lines(["Load data and master pattern to start."])
            return
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

        data = self.session.data
        row = max(0, min(int(self.row_var.get()), data.rows - 1))
        col = max(0, min(int(self.col_var.get()), data.cols - 1))
        idx = self.session.index_from_row_col(row, col)
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
            for plot_view in self._plot_views.values():
                plot_view["combo"]["values"] = available
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
        ax_map.scatter([col], [row], marker="+", color="red", s=160, linewidths=2.0)
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
                self.axes[0, 2].set_title(f"Primary Simulation | NCC={ncc_es:.4f}")
                self.axes[0, 2].set_axis_off()
                sim_shown = True
                live_ncc = float(ncc_es)
            except Exception:
                sim_shown = False

        if not sim_shown:
            self.axes[0, 2].text(0.5, 0.5, "Load a master pattern", ha="center", va="center")
            self.axes[0, 2].set_title("Primary Simulation")

        if sim_shown and live_residual is not None:
            rmin = float(np.nanmin(live_residual))
            rmax = float(np.nanmax(live_residual))
            rabs = max(abs(rmin), abs(rmax), 1e-8)
            rrms = float(np.sqrt(np.mean(np.square(live_residual))))
            im_resid = self.axes[1, 0].imshow(live_residual, cmap="bwr", vmin=-rabs, vmax=rabs)
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
            im_resid = self.axes[1, 0].imshow(self.last_overlap.residual, cmap="bwr", vmin=-rabs, vmax=rabs)
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
            self.axes[1, 1].set_title(f"Residual-indexed Simulation | KP NCC={self.last_overlap.secondary_ncc_kp:.4f}")
            self.axes[1, 1].set_axis_off()
        else:
            self.axes[1, 1].text(0.5, 0.5, "Index residual with step 2 dictionary", ha="center", va="center")
            self.axes[1, 1].set_title("Secondary Simulation")

        if self.session.last_scores_map is not None:
            sm = self.session.last_scores_map
            self.axes[1, 2].imshow(sm, cmap="viridis", origin="upper")
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
        quality_cmap = "tab20" if quality_layer.lower() == "phase" else "viridis"
        quality_vals = quality_full[np.isfinite(quality_full)]
        if quality_vals.size > 0:
            qvmin = float(np.nanpercentile(quality_vals, 2))
            qvmax = float(np.nanpercentile(quality_vals, 98))
            if qvmax <= qvmin:
                qvmax = qvmin + 1e-8
        else:
            qvmin, qvmax = 0.0, 1.0

        preliminary_ipf = self.session.get_preliminary_ipf_color_map(direction="z")
        indexed = self.session.last_indexed_indices is not None and self.session.last_indexed_indices.size > 0
        updated_ipf = self.session.get_ipf_color_map(direction="z") if indexed and self.session.current_eulers_rad is not None else None
        ncc_map = self.session.last_scores_map if indexed else None

        def _draw_rgb(ax, image: np.ndarray | None, title: str, *, zoom: bool) -> None:
            if image is None:
                ax.text(0.5, 0.5, "No IPF available", ha="center", va="center")
                ax.set_title(title)
                return
            ax.imshow(np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0), origin="upper")
            ax.scatter([col], [row], marker="+", color="red", s=180, linewidths=2.0)
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
                ax.set_title(title)
                return
            ax.imshow(np.asarray(image, dtype=np.float32), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            ax.scatter([col], [row], marker="+", color="red", s=180, linewidths=2.0)
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

        _draw_rgb(axes[0, 0], preliminary_ipf, "Preliminary IPF-Z (full map)", zoom=False)
        _draw_numeric(
            axes[0, 1],
            quality_full,
            f"Initial {quality_layer} map (full map)",
            zoom=False,
            cmap=quality_cmap,
            vmin=qvmin,
            vmax=qvmax,
        )
        _draw_rgb(axes[0, 2], preliminary_ipf, "Preliminary IPF-Z (ROI zoom)", zoom=True)
        _draw_numeric(
            axes[1, 0],
            quality_full,
            f"Initial {quality_layer} map (ROI zoom)",
            zoom=True,
            cmap=quality_cmap,
            vmin=qvmin,
            vmax=qvmax,
        )
        _draw_rgb(axes[1, 1], updated_ipf, "Re-indexed ROI IPF-Z", zoom=True)
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
            "Click any map to move the selected point.",
        ]
        if self.session.last_indexed_indices is not None and self.session.last_indexed_indices.size > 0:
            info_lines.append(f"Indexed points in current session: {self.session.last_indexed_indices.size}")
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
            ax.scatter([col], [row], marker="+", color="red", s=180, linewidths=2.0)
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

        primary_ipf = self.session.get_ipf_color_map(direction="z")
        primary_threshold_mask = self._primary_threshold_mask()
        primary_ipf = self._apply_white_mask(primary_ipf, primary_threshold_mask)
        residual_ipf = None
        residual_map_note = "Residual IPF-Z will appear after residual workflow runs"
        threshold_mask = self._residual_threshold_mask()
        if residual_indexed:
            try:
                residual_ipf = self.session.get_residual_ipf_color_map(direction="z")
                residual_map_note = "Residual IPF-Z (reindexed only)"
            except Exception:
                residual_ipf = None

        if primary_threshold_mask is not None:
            threshold_mask = (
                primary_threshold_mask
                if threshold_mask is None
                else np.logical_or(threshold_mask, primary_threshold_mask)
            )
        residual_ipf = self._apply_white_mask(residual_ipf, threshold_mask)

        _decorate_map_axis(primary_ax, primary_ipf, "Primary IPF-Z", "Primary IPF-Z")
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

        result = self.last_overlap if self.last_overlap is not None and self.last_overlap.index == index else self.session.get_residual_point_result(index)
        if result is not None and result.index == index:
            exp_ax.imshow(normalize_for_view(result.experimental), cmap="gray")
            self._overlay_pattern_mask(exp_ax)
            exp_ax.set_title("Experimental pattern")
            sim_ax.imshow(normalize_for_view(result.simulated), cmap="gray")
            pre = result.ncc_unfitted if result.ncc_unfitted is not None else float("nan")
            sim_ax.set_title(f"Primary simulation — NCC {pre:.4f} → {result.ncc_es:.4f}")
            if result.gain_map is not None:
                gain_ax.imshow(result.gain_map, cmap="viridis")
                gain_ax.set_title("Fitted gain mask")
            else:
                gain_ax.text(0.5, 0.5, "Gain fitting disabled", ha="center", va="center")
                gain_ax.set_title("Gain mask")
            rabs = max(float(np.max(np.abs(result.residual))), 1e-8)
            primary_residual_ax.imshow(result.residual, cmap="bwr", vmin=-rabs, vmax=rabs)
            primary_residual_ax.set_title(f"Residual E − {result.scale:.4f}·S′")
            if result.secondary_simulated is not None:
                residual_sim_ax.imshow(normalize_for_view(result.secondary_simulated), cmap="gray")
                match_label = "Refined residual simulation" if result.secondary_refined else "Residual simulation"
                residual_sim_ax.set_title(f"{match_label} — KP NCC={result.secondary_ncc_kp:.4f}")
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
                sim_ax.set_title(f"Indexed solution simulation — NCC={preview_ncc:.4f}")
            else:
                sim_ax.text(0.5, 0.5, "Run step 2 indexing first", ha="center", va="center")
                sim_ax.set_title("Indexed solution simulation")
            if preview_residual is not None and preview_scale is not None:
                rabs = max(float(np.max(np.abs(preview_residual))), 1e-8)
                primary_residual_ax.imshow(preview_residual, cmap="bwr", vmin=-rabs, vmax=rabs)
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

        info_lines = [
            "Re-indexed IPF-Z / residual view",
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
            ax.scatter([col], [row], marker="+", color="red", s=180, linewidths=2.0)
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
            ax.scatter([col], [row], marker="+", color="red", s=180, linewidths=2.0)
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

        primary_ipf = self.session.get_ipf_color_map(direction="z")
        primary_threshold_mask = self._primary_threshold_mask()
        primary_ipf = self._apply_white_mask(primary_ipf, primary_threshold_mask)
        residual_ipf = None
        residual_note = "Residual IPF-Z"
        try:
            residual_ipf = self.session.get_residual_ipf_color_map(direction="z")
        except Exception:
            residual_note = "Residual IPF-Z after step 3"
        threshold_mask = self._overlap_mixture_residual_threshold_mask()
        residual_ipf = self._apply_white_mask(residual_ipf, threshold_mask)

        _decorate_map_axis(primary_ax, primary_ipf, "Primary IPF-Z", "Primary IPF-Z")
        _decorate_map_axis(residual_ax, residual_ipf, residual_note, residual_note)

        result = (
            self.last_overlap_mixture
            if self.last_overlap_mixture is not None and self.last_overlap_mixture.index == index
            else self.session.get_overlap_mixture_result(index)
        )

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
            primary_sim_ax.set_title(primary_title)
            secondary_sim_ax.imshow(normalize_for_view(result.secondary_simulated), cmap="gray")
            secondary_sim_ax.set_title(secondary_title)
            rabs = max(float(np.max(np.abs(result.residual))), 1e-8)
            final_residual_ax.imshow(result.residual, cmap="bwr", vmin=-rabs, vmax=rabs)
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

        _decorate_fraction_axis(primary_fraction_ax, self.session.overlap_primary_fraction_map, "Primary fraction map")
        _decorate_fraction_axis(secondary_fraction_ax, self.session.overlap_secondary_fraction_map, "Residual fraction map")

        info_lines = [
            "Overlap optimization",
            f"Selected point: idx={index}, row={row}, col={col}",
            f"Residual NCC threshold: {_fmt(self._overlap_mixture_residual_ncc_threshold(), 3)}",
        ]
        if result is not None and result.index == index:
            info_lines.extend(
                [
                    f"Fractions: primary={_fmt(result.primary_fraction, 3)}, residual={_fmt(result.secondary_fraction, 3)}",
                    f"NCC: old primary={_fmt(result.old_primary_ncc)}, old residual={_fmt(result.old_secondary_ncc)}, combined={_fmt(result.ncc_mixture)}",
                    f"Fitted sigma={result.fitted_sigma:.4f}; gain (gmin, gmax, p)={result.gain_params}",
                    f"Ellipse (a, b, y offset, x offset)={result.ellipse_params}",
                    f"Coefficients: primary={result.primary_coefficient:.4f}, residual={result.secondary_coefficient:.4f}",
                ]
            )
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

    def _on_plot_click(self, event, view_index: int | None = None) -> None:
        if self.session.data is None:
            return
        active_view = int(view_index) if view_index is not None else None
        if view_index is not None:
            self._activate_plot_view(int(view_index))
        if event.inaxes is None:
            return
        if active_view == 1:
            allowed_axes = [ax for ax in np.asarray(self.axes, dtype=object).flat]
        elif active_view == 3 and self.axes.shape == (2, 4):
            allowed_axes = [self.axes[0, 0], self.axes[0, 1], self.axes[1, 2], self.axes[1, 3]]
        elif active_view == 2 and self.axes.shape[1] > 1:
            allowed_axes = [self.axes[0, 0], self.axes[0, 1]]
        else:
            allowed_axes = [self.axes[0, 0]]
        if event.inaxes not in allowed_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        row = int(np.clip(round(event.ydata), 0, self.session.data.rows - 1))
        col = int(np.clip(round(event.xdata), 0, self.session.data.cols - 1))
        self.row_var.set(row)
        self.col_var.set(col)
        self._sync_index_from_row_col()

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
            self.euler1_deg_var.set(self._quantize(float(e_deg[0]), self._euler_step_deg))
            self.euler2_deg_var.set(self._quantize(float(e_deg[1]), self._euler_step_deg))
            self.euler3_deg_var.set(self._quantize(float(e_deg[2]), self._euler_step_deg))
            self.pcx_var.set(self._quantize(float(pc[0]), self._pc_step))
            self.pcy_var.set(self._quantize(float(pc[1]), self._pc_step))
            self.pcz_var.set(self._quantize(float(pc[2]), self._pc_step))
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
