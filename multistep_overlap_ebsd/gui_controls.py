"""Tk control layout for the four-stage overlap workflow."""
from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from .gui_phases import PhaseControls
from .cpu_fitting import FIT_METHOD_LABELS
from .gui_theme import BACKGROUND, configure_theme


class CollapsibleSection(ttk.Frame):
    """Disclosure that preserves the values and widgets inside it."""

    def __init__(self, parent, title: str, *, expanded: bool = False):
        super().__init__(parent)
        self.title = title
        self.toggle = ttk.Button(self, command=self.toggle_open, style="Disclosure.TButton")
        self.toggle.pack(fill=tk.X)
        self.content = ttk.Frame(self, padding=(8, 6))
        self.set_open(expanded)

    def set_open(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self.toggle.configure(text=f"{'▾' if self.expanded else '▸'} {self.title}")
        if self.expanded:
            self.content.pack(fill=tk.X)
        else:
            self.content.pack_forget()

    def toggle_open(self) -> None:
        self.set_open(not self.expanded)


class GUIControls(PhaseControls):
    """Layout and presentation helpers; processing actions live in gui.py."""

    def _box(self, parent, title):
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        return box

    def _advanced(self, parent, title, *, expanded=False):
        section = CollapsibleSection(parent, title, expanded=expanded)
        section.pack(fill=tk.X, pady=3)
        return section

    def _field(self, parent, label, variable, *, width=10, state="normal"):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, wraplength=255).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable, width=width, state=state)
        entry.pack(side=tk.RIGHT, padx=(8, 0))
        return entry

    def _action(self, parent, label, callback):
        button = ttk.Button(parent, text=label, command=callback)
        button.pack(fill=tk.X, pady=3)
        return button

    def _hint(self, parent, text=None, *, variable=None):
        options = {"textvariable": variable} if variable is not None else {"text": text}
        label = ttk.Label(parent, wraplength=370, style="Hint.TLabel", **options)
        label.pack(fill=tk.X, pady=(3, 5))
        return label

    def _progress(self, parent, variable, status):
        bar = ttk.Progressbar(parent, variable=variable, maximum=100.0)
        bar.pack(fill=tk.X, pady=(5, 0))
        self._hint(parent, variable=status)
        return bar

    def _build_ui(self):
        configure_theme(self)
        ttk.Style(self).configure("Disclosure.TButton", anchor="w")
        for name, color in (("PendingCalibration", "#9f1239"), ("AppliedCalibration", "#166534")):
            ttk.Style(self).configure(f"{name}.TButton", foreground=color)
            ttk.Style(self).map(f"{name}.TButton", foreground=[("disabled", "#777777"), ("!disabled", color)])
        self._build_workflow_controls(self)
        self._calibration_warning_banner = tk.Frame(self, background="#fff1f2", padx=10, pady=7)
        tk.Label(
            self._calibration_warning_banner,
            text="PC calibration needs applying. Apply the average in tab 1 before continuing with indexing or mixture analysis.",
            background="#fff1f2", foreground="#9f1239", anchor="w", justify=tk.LEFT, wraplength=760,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._calibration_warning_button = ttk.Button(
            self._calibration_warning_banner, text="Review & apply PC…", command=self._show_calibration_application,
            style="PendingCalibration.TButton",
        )
        self._calibration_warning_button.pack(side=tk.RIGHT, padx=(10, 0))
        self.workflow_notebook = ttk.Notebook(self, style="Workflow.TNotebook")
        self.workflow_notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(3, 6))
        builders = (
            ("1. Load & PC Calibration", self._build_calibration_workspace),
            ("2. Dictionary Indexing", self._build_indexing_workspace),
            ("3. Residual Indexing", self._build_overlap_workspace),
            ("4. Mixture Optimization", self._build_overlap_optimization_workspace),
        )
        for title, builder in builders:
            tab = ttk.Frame(self.workflow_notebook)
            self.workflow_notebook.add(tab, text=title)
            builder(tab)
        self.workflow_notebook.bind("<<NotebookTabChanged>>", self._on_workspace_changed)
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(fill=tk.X)
        self.bind("<Control-s>", lambda _e: self._save_workflow())
        self.bind("<Command-s>", lambda _e: self._save_workflow())
        self.bind("<Control-o>", lambda _e: self._restore_workflow())
        self.bind("<Command-o>", lambda _e: self._restore_workflow())
        self.bind("<MouseWheel>", self._on_controls_mousewheel, add="+")
        self.bind("<Button-4>", self._on_controls_mousewheel, add="+")
        self.bind("<Button-5>", self._on_controls_mousewheel, add="+")
        self._activate_plot_view(0)
        self._draw_instruction("Load data and a master pattern to start.")

    def _build_workflow_controls(self, parent):
        bar = ttk.Frame(parent, padding=(8, 6))
        bar.pack(fill=tk.X)
        for col, (label, command) in enumerate((
            ("Open workflow…", self._restore_workflow),
            ("Save", self._save_workflow),
            ("Save as…", self._save_workflow_as),
        )):
            ttk.Button(bar, text=label, command=command).grid(row=0, column=col, padx=(0, 4))
        ttk.Entry(bar, textvariable=self.workflow_path_var, width=32).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(bar, text="Pattern conditioning…", command=self._show_conditioning).grid(row=0, column=4, padx=4)
        ttk.Label(bar, text="Max worker cores").grid(row=0, column=5, padx=(8, 4))
        ttk.Spinbox(bar, textvariable=self.parallel_cores_var, from_=1, to=max(1, os.cpu_count() or 1), width=4).grid(row=0, column=6)
        self.btn_cancel = ttk.Button(bar, text="Cancel", command=self._cancel_current_action, state="disabled")
        self.btn_cancel.grid(row=0, column=7, padx=(8, 0))
        ttk.Label(bar, textvariable=self.context_summary_var, anchor="w", wraplength=800).grid(row=1, column=0, columnspan=5, sticky="ew", pady=(5, 0))
        ttk.Label(bar, text="Cores: indexing, refinement + fitting", anchor="e").grid(row=1, column=5, columnspan=3, sticky="e", pady=(5, 0))
        ttk.Label(bar, textvariable=self.live_update_status_var, anchor="w").grid(
            row=2, column=0, columnspan=8, sticky="ew", pady=(3, 0))
        bar.columnconfigure(3, weight=1)

    def _workspace_panes(self, parent):
        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left, right = ttk.Frame(paned, width=420), ttk.Frame(paned, padding=6)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        return left, right

    def _scrollable_controls(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0, width=415, background=BACKGROUND)
        canvas._workflow_controls = True
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls = ttk.Frame(canvas, padding=8)
        window = canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        return controls

    def _on_controls_mousewheel(self, event):
        widget = event.widget
        if isinstance(widget, (tk.Text, tk.Listbox)):
            return
        while widget is not None:
            if isinstance(widget, tk.Canvas) and getattr(widget, "_workflow_controls", False):
                direction = -1 if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0 else 1
                widget.yview_scroll(direction, "units")
                return "break"
            widget = getattr(widget, "master", None)

    def _build_calibration_workspace(self, parent):
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        self._build_input_controls(controls)
        self._build_phase_controls(controls)
        self._build_refine_tab(self._box(controls, "Pattern-center calibration"))
        self._conditioning_section = self._advanced(controls, "Pattern conditioning · all stages")
        self._build_conditioning_controls(self._conditioning_section.content)
        self._build_selection_controls(controls, include_roi=False)
        self._build_point_editor_controls(controls)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 0)

    def _build_input_controls(self, parent):
        box = self._box(parent, "Input data")
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(row, text="Input type").pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=self.source_type_var, values=("H5OINA", "UP + ANG"), state="readonly", width=19)
        combo.pack(side=tk.RIGHT)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_input_type_controls(reset_up_tilt=True))
        self._hint(box, variable=self.pattern_input_label_var)
        self._file_row(box, self.pattern_path_var)
        self._ang_input_row = ttk.Frame(box)
        self._ang_input_row.pack(fill=tk.X)
        self._hint(self._ang_input_row, "Orientations (.ang)")
        self._file_row(self._ang_input_row, self.orientation_path_var)
        self._input_load_button = self._action(box, "Load input data…", self._choose_and_load_input)
        self._sample_tilt_entry = self._field(box, "Sample tilt (°)", self.sample_tilt_var)
        self._hint(box, variable=self.pc_conv_label_var)
        geometry = self._advanced(box, "Acquisition geometry").content
        self._detector_tilt_entry = self._field(geometry, "Detector tilt (°)", self.detector_tilt_var)
        self._hint(geometry, "H5OINA supplies tilt and PC convention. UP + ANG starts at 70° sample tilt and uses Oxford PC scaling for compatibility; tilt edits apply on the next input load.")

    def _file_row(self, parent, variable, browse=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=variable, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if browse is not None:
            ttk.Button(row, text="Browse…", command=browse).pack(side=tk.RIGHT, padx=(4, 0))

    def _build_refine_tab(self, parent):
        self._hint(parent, "Recalibration takes two steps: optimize selected points, then apply their average to the entire map.")
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        for text, command in (("Add point", self._add_calibration_point), ("Remove point", self._remove_calibration_point), ("Clear", self._clear_calibration_points)):
            ttk.Button(row, text=text, command=command).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_refine_roi = self._action(parent, "1. Optimize calibration points", self._refine_calibration_points)
        self._calibration_statistics_label = self._hint(parent, variable=self.calibration_statistics_var)
        self._calibration_statistics_label.pack_forget()
        self._calibration_report_section = self._advanced(parent, "Detailed calibration report")
        self._hint(self._calibration_report_section.content, variable=self.calibration_summary_var)
        application = self._box(parent, "2. Apply PC to the entire map")
        self._calibration_apply_panel = application
        self._calibration_apply_status_label = tk.Label(
            application, textvariable=self.calibration_apply_status_var,
            anchor="w", justify=tk.LEFT, wraplength=340, padx=8, pady=8,
        )
        self._calibration_apply_status_label.pack(fill=tk.X, pady=(0, 5))
        self._hint(application, "Optimization changes only the selected points. Apply updates the PC used across the entire scan.")
        row = ttk.Frame(application)
        row.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(row, text="Correct PC for scan position", variable=self.use_scan_pc_shift_var).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.detector_px_size_var, width=7).pack(side=tk.LEFT, padx=(5, 3))
        ttk.Label(row, text="µm/px").pack(side=tk.LEFT)
        self._hint(application, variable=self.detector_pixel_source_var)
        self._hint(application, "Pixel size refers to a stored pattern pixel on the detector. With correction off, the average PC is applied uniformly.")
        self.btn_apply_calibration = self._action(application, "2. Apply average PC to map", self._apply_average_calibration_pc)
        advanced = self._advanced(parent, "Calibration search limits & selected point").content
        self._field(advanced, "Orientation search half-range (°)", self.calibration_trust_euler_var)
        self._field(advanced, "PC search half-range", self.trust_pc_var)
        self._field(advanced, "Maximum evaluations", self.calibration_maxfev_var)
        self._action(advanced, "Optimize selected point (orientation + PC)", self._refine_selected_point)

    def _build_conditioning_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Pattern mask").pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=self.pattern_mask_mode_var, values=("Automatic", "None", "Custom diameter"), state="readonly", width=16)
        combo.pack(side=tk.RIGHT)
        combo.bind("<<ComboboxSelected>>", self._on_mask_mode_changed)
        self._mask_diameter_entry = self._field(parent, "Custom circular diameter (px)", self.pattern_mask_option_var, state="disabled")
        self._hint(parent, variable=self.pattern_mask_status_var)
        ttk.Checkbutton(parent, text="Dynamic background subtraction", variable=self.dynamic_bg_enabled_var, command=self._on_value_commit).pack(anchor="w", pady=(8, 2))
        self._field(parent, "Background blur σ (px; 0 = automatic)", self.dynamic_bg_std_var)
        self._hint(parent, variable=self.dynamic_bg_status_var)
        self._hint(parent, "Shared by calibration, indexing, residuals and mixture fitting. Changing conditioning invalidates affected results.")

    def _sync_mask_mode(self, *_args):
        try:
            value = int(self.pattern_mask_option_var.get())
        except (tk.TclError, ValueError):
            return
        self.pattern_mask_mode_var.set("None" if value < 0 else "Automatic" if value == 0 else "Custom diameter")
        if hasattr(self, "_mask_diameter_entry"):
            self._mask_diameter_entry.configure(state="normal" if value > 0 else "disabled")

    def _on_mask_mode_changed(self, _event=None):
        mode = self.pattern_mask_mode_var.get()
        if mode == "Automatic":
            self.pattern_mask_option_var.set(0)
        elif mode == "None":
            self.pattern_mask_option_var.set(-1)
        else:
            data = self.session.data
            self.pattern_mask_option_var.set(max(1, min(data.h, data.w)) if data is not None else 128)
        self._on_value_commit()

    def _build_indexing_workspace(self, parent):
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        self._build_roi_controls(controls)
        self._build_index_tab(self._box(controls, "Dictionary & indexing"))
        settings = self._advanced(controls, "Indexing & refinement settings · tabs 2 and 3")
        self._refinement_section = settings
        self._field(settings.content, "Retained matches", self.dictionary_keep_n_var)
        ttk.Checkbutton(settings.content, text="Search range follows dictionary spacing", variable=self.follow_dictionary_trust_var).pack(anchor="w", pady=3)
        self._trust_euler_entry = self._field(settings.content, "Euler search half-range (°)", self.trust_euler_var)
        self._field(settings.content, "Maximum evaluations", self.maxfev_var)
        ttk.Checkbutton(settings.content, text="Use full-resolution patterns for refinement", variable=self.refine_full_resolution_var).pack(anchor="w", pady=3)
        self._hint(settings.content, "The dictionary spacing sets a starting search range for each Euler angle. A manual range is available for difficult cases.")
        self._build_selection_controls(controls, include_roi=False)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 1)

    def _build_index_tab(self, parent):
        self._hint(parent, variable=self.phase_summary_var)
        self._hint(parent, "Indexing searches all enabled phases using the dictionaries linked in tab 1.")
        self._action(parent, "Edit phases / dictionaries…", lambda: self.workflow_notebook.select(0))
        self._action(parent, "Phase maps / IPF keys…", self._show_phase_maps)
        self._keep_imported_phases_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Keep imported phase assignments", variable=self._keep_imported_phases_var,
                        command=self._set_keep_imported_phases).pack(anchor="w")
        ttk.Checkbutton(parent, text="Refine automatically after indexing (tabs 2 and 3)", variable=self.auto_refine_var).pack(anchor="w", pady=(8, 3))
        self.btn_index_roi = self._action(parent, "Index ROI", self._index_roi)
        self.btn_refine_indexed = self._action(parent, "Refine last indexed ROI again", self._refine_last_indexed)
        self._progress(parent, self.reindex_progress_var, self.reindex_progress_status_var)
        self.refinement_progress_bar = self._progress(parent, self.refinement_progress_var, self.refinement_progress_status_var)
        automation = self._box(parent, "Run multiple stages on this ROI")
        self.btn_steps_2_3_analysis = self._action(automation, "Run steps 2–3 · primary + residual", lambda: self._run_complete_roi_analysis(include_step4=False))
        self.btn_complete_analysis = self._action(automation, "Run steps 2–4 · include mixture fit", self._run_complete_roi_analysis)
        self._hint(automation, "Uses the shared refinement settings, thresholds in tabs 3–4, and the global worker limit. Automatic refinement follows the checkbox above.")
        self._progress(automation, self.complete_analysis_progress_var, self.complete_analysis_status_var)
        manual = self._advanced(parent, "Selected-point indexing").content
        self._action(manual, "Index selected point", self._index_selected_point)

    def _build_overlap_workspace(self, parent):
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        self._build_shared_roi_summary(controls)
        self._build_overlap_tab(self._box(controls, "Residual indexing"))
        self._build_selection_controls(controls, include_roi=False)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 2, fixed_ipf=True)

    def _build_overlap_tab(self, parent):
        self._action(parent, "Phase maps / IPF keys…", self._show_phase_maps)
        self._hint(parent, "Retained matches, refinement range, evaluation limit and full-resolution choice are shared with tab 2.")
        self._action(parent, "Edit shared indexing settings…", lambda: self._show_section(self._refinement_section, 1))
        self._field(parent, "Minimum primary NCC for residual work", self.overlap_min_ncc_var)
        self._build_fit_method_controls(parent)
        self._action(parent, "Analyze residual ROI", self._run_residual_roi_analysis)
        self.btn_steps_3_4_analysis = self._action(
            parent, "Run steps 3–4 · residual + mixture fit",
            lambda: self._run_residual_roi_analysis(include_step4=True),
        )
        self._hint(parent, "Uses existing primary indexing, computes and indexes residuals, then fits mixtures using the tab 4 NCC threshold.")
        self._field(parent, "Minimum residual NCC shown in maps", self.residual_ipf_ncc_var)
        self._hint(parent, "Display threshold: hides weak residual orientations. The processing threshold for mixture fitting is in tab 4.")
        self.overlap_progress_bar = self._progress(parent, self.overlap_progress_var, self.overlap_progress_status_var)
        manual = self._advanced(parent, "Individual steps & selected-point inspection").content
        for text, callback in (
            ("Fit selected point and build residual", self._analyze_overlap),
            ("Index selected residual only", self._index_overlap_residual),
            ("Refine selected residual again", self._refine_overlap_residual),
            ("Open residual inspection", self._show_current_overlap_inspection),
            ("Compute ROI residuals only", self._compute_overlap_residual_roi),
            ("Index residual ROI only", self._index_overlap_residual_roi),
            ("Refine residual ROI again", self._refine_overlap_residual_roi),
        ):
            self._action(manual, text, callback)
        fit = self._advanced(parent, "Blur / gain fit settings · tabs 3 and 4").content
        ttk.Checkbutton(fit, text="Fit Gaussian blur and elliptical gain", variable=self.fit_blur_gain_var).pack(anchor="w")
        self._field(fit, "Manual blur σ (when fitting is off)", self.blur_sigma_var)
        self._hint(fit, "The checkbox and manual blur apply to residual generation. Mixture fitting always fits blur and gain; it shares the limits below.")
        self._field(fit, "Maximum fit iterations", self.gain_fit_maxiter_var)
        self._field(fit, "Population multiplier (minimum 4)", self.gain_fit_popsize_var)
        self._build_fit_bounds(fit)
        writing = self._advanced(parent, "Write residual patterns during processing").content
        ttk.Checkbutton(writing, text="Write residual patterns to disk", variable=self.write_residual_patterns_var).pack(anchor="w")
        self._file_row(writing, self.residual_pattern_path_var, self._browse_residual_pattern_output)
        self._build_map_export_controls(parent)

    def _build_map_export_controls(self, parent):
        box = self._box(parent, "Export maps & optional patterns")
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Output format").pack(side=tk.LEFT)
        self._roi_export_format_combo = ttk.Combobox(row, textvariable=self.roi_export_format_var, values=("H5OINA", "ANG"), state="readonly", width=12)
        self._roi_export_format_combo.pack(side=tk.RIGHT)
        self._roi_export_format_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_roi_export_paths_to_source())
        self._hint(box, "Exports keep full scan dimensions with ROI results. H5OINA input exports H5OINA; UP + ANG supports either format.")
        ttk.Checkbutton(box, text="Include primary patterns", variable=self.include_primary_patterns_export_var).pack(anchor="w")
        self._action(box, "Export primary ROI map…", self._export_primary_roi_map)
        ttk.Checkbutton(box, text="Include residual patterns", variable=self.include_residual_patterns_export_var).pack(anchor="w")
        self._action(box, "Export residual ROI map…", self._export_residual_roi_map)
        self._hint(box, "Patterns are stored inside H5OINA or in a companion UP1 file for ANG.")

    def _build_fit_bounds(self, parent):
        self._hint(parent, "Shared blur/gain bounds (lower / upper)")
        for label, low, high in self.primary_fit_bound_specs:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=high, width=8).pack(side=tk.RIGHT, padx=(4, 0))
            ttk.Entry(row, textvariable=low, width=8).pack(side=tk.RIGHT)

    def _build_fit_method_controls(self, parent):
        ttk.Label(parent, text="Blur / gain fitting method · tabs 3 and 4").pack(anchor="w", pady=(5, 2))
        combo = ttk.Combobox(
            parent, textvariable=self.fit_method_var, values=tuple(FIT_METHOD_LABELS.values()),
            state="readonly", width=32,
        )
        combo.pack(fill=tk.X, pady=2)
        self._hint(parent, "Blur then gain is fastest. Joint refinement takes longer and can improve the fit. The choice is shared by both tabs.")

    def _build_overlap_optimization_workspace(self, parent):
        left, right = self._workspace_panes(parent)
        controls = self._scrollable_controls(left)
        self._build_shared_roi_summary(controls)
        self._build_overlap_optimization_tab(self._box(controls, "Mixture optimization"))
        self._build_selection_controls(controls, include_roi=False)
        self._build_info_and_log(controls, log_height=8)
        self._build_plot_area(right, 3, fixed_ipf=True)

    def _build_overlap_optimization_tab(self, parent):
        self._action(parent, "Phase maps / IPF keys…", self._show_phase_maps)
        self._field(parent, "Minimum residual NCC for mixture fitting", self.overlap_mixture_residual_ncc_var)
        self._build_fit_method_controls(parent)
        self._action(parent, "Fit mixture for ROI", self._fit_overlap_mixture_roi)
        self._action(parent, "Fit selected-point mixture", self._fit_overlap_mixture)
        self._progress(parent, self.overlap_optimization_progress_var, self.overlap_optimization_status_var)
        self._action(parent, "Export full-map mixture results…", self._export_overlap_optimization_results)
        self._hint(parent, "Export keeps full scan dimensions; missing fits are marked by a mask and NaN values.")
        settings = self._advanced(parent, "Shared blur / gain optimizer settings").content
        self._field(settings, "Maximum fit iterations", self.gain_fit_maxiter_var)
        self._field(settings, "Population multiplier (minimum 4)", self.gain_fit_popsize_var)
        self._build_fit_bounds(settings)

    def _build_roi_controls(self, parent):
        box = self._box(parent, "Region of interest")
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        for text, var in (("Start row", self.roi_r0_var), ("Start col", self.roi_c0_var), ("Rows", self.roi_nrows_var), ("Cols", self.roi_ncols_var)):
            field = ttk.Frame(row)
            field.pack(side=tk.LEFT, expand=True, fill=tk.X)
            ttk.Label(field, text=text).pack(anchor="w")
            ttk.Entry(field, textvariable=var, width=7).pack(fill=tk.X, padx=(0, 4))
        actions = ttk.Frame(box)
        actions.pack(fill=tk.X, pady=3)
        ttk.Button(actions, text="Center on selected point", command=self._center_roi_on_selected).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(actions, text="Full map", command=self._use_full_map_roi).pack(side=tk.RIGHT, padx=(4, 0))
        self._hint(box, "Shift-drag on a map to draw the ROI. Used by tabs 2, 3 and 4. Coordinates are zero-based.")

    def _build_shared_roi_summary(self, parent):
        box = self._box(parent, "Shared ROI")
        self._hint(box, variable=self.context_summary_var)
        self._action(box, "Edit ROI in tab 2…", lambda: self.workflow_notebook.select(1))

    def _build_selection_controls(self, parent, *, include_roi=False):
        box = self._advanced(parent, "Selected map point").content
        for name, var, callback in (("Row", self.row_var, self._sync_index_from_row_col), ("Column", self.col_var, self._sync_index_from_row_col), ("Point index", self.index_var, self._sync_row_col_from_index)):
            entry = self._field(box, name, var)
            entry.bind("<Return>", lambda _e, fn=callback: (fn(), "break")[1])
            entry.bind("<KP_Enter>", lambda _e, fn=callback: (fn(), "break")[1])
        self._hint(box, "Click a map to select a point, or enter a coordinate and press Enter.")

    def _build_point_editor_controls(self, parent):
        box = self._advanced(parent, "Manual orientation / PC preview").content
        for text, variable in (("Euler φ1 (°)", self.euler1_deg_var), ("Euler Φ (°)", self.euler2_deg_var), ("Euler φ2 (°)", self.euler3_deg_var), ("PC x", self.pcx_var), ("PC y", self.pcy_var), ("PC z", self.pcz_var)):
            self._field(box, text, variable)
        self._hint(box, "Edits preview the selected pattern. Apply commits them and invalidates dependent results.")
        self._action(box, "Revert to saved point values", self._load_selected_point_values)
        self._action(box, "Apply edited values", self._apply_selected_point_values)

    def _build_info_and_log(self, parent, *, log_height):
        section = self._advanced(parent, "Point details & log")
        info = tk.Text(section.content, height=7, width=35, wrap=tk.WORD)
        info.pack(fill=tk.X)
        info.configure(state=tk.DISABLED)
        self.info_texts.append(info)
        if self.info_text is None:
            self.info_text = info
        log = tk.Text(section.content, height=log_height, width=35, wrap=tk.WORD)
        log.pack(fill=tk.X, pady=(5, 0))
        self.log_texts.append(log)
        if not hasattr(self, "log_text"):
            self.log_text = log

    def _show_section(self, section, tab_index):
        self.workflow_notebook.select(tab_index)
        section.set_open(True)
        self._scroll_control_into_view(section)

    def _scroll_control_into_view(self, control):
        self.update_idletasks()
        widget = control.master
        while widget is not None and not isinstance(widget, tk.Canvas):
            widget = getattr(widget, "master", None)
        if widget is not None:
            bbox = widget.bbox("all")
            if bbox and bbox[3] > 0:
                y = control.winfo_rooty() - widget.winfo_rooty() + widget.canvasy(0)
                widget.yview_moveto(max(0, y / bbox[3]))

    def _show_calibration_application(self):
        self.workflow_notebook.select(0)
        self._scroll_control_into_view(self._calibration_apply_panel)
        self.btn_apply_calibration.focus_set()

    def _show_conditioning(self):
        self._show_section(self._conditioning_section, 0)

    def _sync_input_type_controls(self, *, reset_up_tilt=False):
        is_h5 = self.source_type_var.get() == "H5OINA"
        self.pattern_input_label_var.set("Patterns + orientations (.h5oina)" if is_h5 else "Patterns (.up1 / .up2)")
        if is_h5:
            self._ang_input_row.pack_forget()
        else:
            self._ang_input_row.pack(fill=tk.X, before=self._input_load_button)
            if reset_up_tilt:
                self.sample_tilt_var.set(70.0)
                self.detector_tilt_var.set(0.0)
        self._sample_tilt_entry.configure(state="readonly" if is_h5 else "normal")
        self._detector_tilt_entry.configure(state="readonly" if is_h5 else "normal")
        data = self.session.data
        if data is not None:
            self.pc_conv_label_var.set(f"PC convention: {data.pc_output_convention.capitalize()} (loaded data)")
        if hasattr(self, "_roi_export_format_combo"):
            loaded_h5 = data is not None and data.source_type == "h5oina"
            self._roi_export_format_combo.configure(values=("H5OINA",) if loaded_h5 else ("H5OINA", "ANG"))
            if loaded_h5:
                self.roi_export_format_var.set("H5OINA")
        self._sync_mask_mode()

    def _update_dictionary_binned_size(self, *_args):
        try:
            factor = int(self.di_binning_var.get())
            if factor < 1:
                raise ValueError
        except (ValueError, TypeError, tk.TclError):
            self.dictionary_binned_size_var.set("Enter a positive whole-number binning factor.")
            return
        data = self.session.data
        if data is None:
            self.dictionary_binned_size_var.set("Load input data to see the binned pattern size.")
            return
        try:
            top, bottom, left, right = self.session._binning_crop_extent(factor)
        except ValueError:
            self.dictionary_binned_size_var.set(f"Binning {factor} is too large for {data.w} × {data.h} px patterns.")
            return
        height, width = bottom - top, right - left
        message = f"Binned size for generation: {width // factor} × {height // factor} px (width × height)."
        if (height, width) != (data.h, data.w):
            message += f" Center crop to {width} × {height} px before binning."
        self.dictionary_binned_size_var.set(message)

    def _sync_refinement_settings(self, *_args):
        try:
            follow = self.follow_dictionary_trust_var.get()
            if follow:
                cache = self.session.dictionary_cache
                spacing = float(cache.resolution_deg if cache is not None else self.di_res_deg_var.get())
                if spacing > 0:
                    self.trust_euler_var.set(spacing)
            if hasattr(self, "_trust_euler_entry"):
                self._trust_euler_entry.configure(state="readonly" if follow else "normal")
        except (ValueError, tk.TclError):
            pass

    def _refresh_context_summary(self):
        if hasattr(self, "phase_table"):
            self._refresh_phase_table()
        if hasattr(self, "_keep_imported_phases_var"):
            self._keep_imported_phases_var.set(self.session.keep_imported_phase_assignments)
        data = self.session.data
        master = self.session.master
        cache = self.session.dictionary_cache
        parts = [f"Data: {data.rows} × {data.cols}" if data is not None else "No data"]
        parts.append(f"MP: {Path(master.path).stem}" if master is not None else "No master pattern")
        parts.append("Dictionary ready" if cache is not None else "No dictionary")
        if data is not None:
            try:
                parts.append(
                    f"ROI: {self.roi_nrows_var.get()} × {self.roi_ncols_var.get()} "
                    f"from row {self.roi_r0_var.get()}, col {self.roi_c0_var.get()}"
                )
            except tk.TclError:
                pass
            parts.append(f"PC: {data.pc_output_convention}")
        self.context_summary_var.set(" · ".join(parts))
        phase_name = getattr(getattr(master, "phase", None), "name", "") if master is not None else ""
        phase_text = (
            f"Phase: {phase_name} · output ID {self.phase_id_var.get()}" if phase_name
            else f"Single phase from {Path(master.path).stem}" if master is not None
            else "Load a master pattern to define the indexing phase."
        )
        registry = getattr(self.session, "phase_registry", None)
        if registry is not None and registry.entries:
            enabled = [entry.name for entry in registry.entries if entry.enabled]
            phase_text = "Enabled phases: " + (", ".join(enabled) or "none")
            revision = getattr(self.session, "phase_search_revision", None)
            if revision is not None and revision != registry.search_revision:
                phase_text += " · results use an earlier phase selection; re-index to compare the current set"
        self.phase_summary_var.set(phase_text)
