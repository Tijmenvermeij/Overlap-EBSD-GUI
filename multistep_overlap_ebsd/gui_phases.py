"""Phase and dictionary controls for the input tab."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class PhaseControls:
    def _build_phase_controls(self, parent):
        box = self._box(parent, "Phases, master patterns & dictionaries")
        self.phase_table = ttk.Treeview(box, columns=("enabled", "name", "master", "dictionary"),
                                       show="headings", height=4, selectmode="browse")
        for key, title, width in (("enabled", "Use", 35), ("name", "Phase", 100),
                                  ("master", "Master", 80), ("dictionary", "Dictionary", 130)):
            self.phase_table.heading(key, text=title)
            self.phase_table.column(key, width=width, stretch=True)
        self.phase_table.column("enabled", anchor=tk.CENTER, stretch=False)
        self.phase_table.pack(fill=tk.X)
        self.phase_table.bind("<<TreeviewSelect>>", self._select_phase_row)
        self.phase_table.bind("<Button-1>", self._click_phase_use)
        self.phase_table.bind("<space>", self._toggle_phase_from_keyboard)
        self._hint(box, "Select an imported phase to link its master. Add a phase for a material absent from the input.")
        self._hint(box, "Click a checkmark in Use to include or exclude a phase.")
        self._action(box, "Add new phase (master pattern)", lambda: self._choose_phase_master(add=True))
        self._action(box, "Link master to selected phase…", self._choose_phase_master)
        self._action(box, "Edit phase name / color…", self._edit_phase_appearance)
        self._action(box, "Phase maps / IPF keys…", self._show_phase_maps)

        self._field(box, "Shared orientation spacing (°)", self.di_res_deg_var)
        self._field(box, "Shared dictionary binning", self.di_binning_var)
        self._hint(box, variable=self.dictionary_binned_size_var)
        self._action(box, "Generate missing / incompatible dictionaries", self._generate_phase_dictionaries)
        self._action(box, "Load dictionary for selected phase…", self._load_phase_dictionary)
        self.btn_save_dictionary = self._action(box, "Save dictionary for selected phase…", self._save_phase_dictionary)
        self.dictionary_progress_bar = self._progress(box, self.dictionary_progress_var, self.dictionary_status_var)

    def _set_keep_imported_phases(self):
        if getattr(self, "_worker_thread", None) is not None:
            self._keep_imported_phases_var.set(self.session.keep_imported_phase_assignments)
            return
        self.session.keep_imported_phase_assignments = self._keep_imported_phases_var.get()

    def _selected_phase_key(self):
        selected = self.phase_table.selection()
        if not selected:
            raise ValueError("Select a phase in tab 1 first.")
        return selected[0]

    def _refresh_phase_table(self):
        if not hasattr(self, "phase_table"):
            return
        selected = self.phase_table.selection()
        self.phase_table.delete(*self.phase_table.get_children())
        for entry in self.session.phase_registry.entries:
            master = self.session.phase_masters.get(entry.key)
            cache = self.session.phase_dictionaries.get(entry.key)
            expected = self.session.phase_dictionary_provenance(entry.key, cache) if master and cache else None
            status = entry.status(expected)
            self.phase_table.insert("", "end", iid=entry.key,
                                    values=("☑" if entry.enabled else "☐", entry.name,
                                            "Loaded" if master else "Missing", status))
        valid = [key for key in selected if self.phase_table.exists(key)]
        if valid:
            self.phase_table.selection_set(valid)
            self.phase_table.focus(valid[0])
        elif self.phase_registry_entries():
            self.phase_table.selection_set(self.phase_registry_entries()[0].key)

    def phase_registry_entries(self):
        return self.session.phase_registry.entries

    def _select_phase_row(self, _event=None):
        if getattr(self, "_worker_thread", None) is not None:
            return
        selected = self.phase_table.selection()
        if not selected:
            return
        entry = self.session.phase_registry.by_key(selected[0])
        self.phase_id_var.set(entry.output_id)
        self.master_path_var.set(entry.master_path)
        # Selection is a UI default; computation dispatches by each point's phase.
        if entry.key in self.session.phase_masters:
            self.session.master = self.session.phase_masters[entry.key]
            self.session.dictionary_cache = self.session.phase_dictionaries.get(entry.key)

    def _choose_phase_master(self, *, add=False):
        if getattr(self, "_worker_thread", None) is not None:
            return
        try:
            key = None if add else self._selected_phase_key()
        except ValueError as exc:
            messagebox.showerror("Select a phase", str(exc))
            return
        paths = filedialog.askopenfilenames(title="Add phase master patterns" if add else "Link phase master pattern",
                                           filetypes=[("Master patterns", "*.h5 *.hdf5 *.mat"), ("All files", "*")])
        if not paths:
            return
        if not add and len(paths) != 1:
            messagebox.showerror("Link master", "Choose one master for the selected phase.")
            return
        def action():
            notes = []
            for path in paths:
                self._check_job_cancelled()
                notes.append(self.session.attach_phase_master(path, phase_key=key))
            return " ".join(notes)
        self._run_threaded(action, on_success=lambda _msg: self._refresh_phase_table())

    def _click_phase_use(self, event):
        if (self.phase_table.identify_region(event.x, event.y) != "cell"
                or self.phase_table.identify_column(event.x) != "#1"):
            return
        key = self.phase_table.identify_row(event.y)
        if not key:
            return
        if getattr(self, "_worker_thread", None) is None:
            self.phase_table.selection_set(key)
            self.phase_table.focus(key)
            self.phase_table.focus_set()
            self._toggle_phase()
        return "break"

    def _toggle_phase_from_keyboard(self, _event):
        self._toggle_phase()
        return "break"

    def _toggle_phase(self):
        if getattr(self, "_worker_thread", None) is not None:
            return
        try:
            key = self._selected_phase_key()
        except ValueError:
            return
        entry = self.session.phase_registry.by_key(key)
        self.session.phase_registry.set_enabled(key, not entry.enabled)
        self._refresh_phase_table()
        self._refresh_context_summary()

    def _generate_phase_dictionaries(self):
        if getattr(self, "_worker_thread", None) is not None:
            return
        try:
            spacing, binning = float(self.di_res_deg_var.get()), int(self.di_binning_var.get())
            pending = []
            entries = [e for e in self.session.phase_registry.entries if e.enabled]
            if not entries:
                raise ValueError("Add or enable a phase first.")
            for entry in entries:
                if entry.key not in self.session.phase_masters:
                    raise ValueError(f"{entry.name}: link a master pattern first.")
                cache = self.session.phase_dictionaries.get(entry.key)
                if cache is not None and cache.resolution_deg == spacing and cache.software_binning == binning:
                    expected = self.session.phase_dictionary_provenance(entry.key, cache)
                    if entry.status(expected) == "Ready":
                        continue
                pending.append(entry)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Generate dictionaries", str(exc), parent=self)
            return
        if pending:
            self._update_calibration_application_controls()
            if self._calibration_apply_state != "applied":
                message = (
                    "PC calibration changes have not been applied to the entire map."
                    if self._calibration_apply_state == "pending" else
                    "The pattern center (PC) has not been refined and applied in this workflow."
                )
                if not messagebox.askyesno(
                    "Generate dictionaries before PC calibration?",
                    message + "\n\nRefining or changing the PC later will require regenerating the dictionaries."
                    "\n\nGenerate dictionaries using the current PC anyway?",
                    parent=self, icon="warning", default="no",
                ):
                    return
        def action():
            notes = []
            for number, entry in enumerate(pending, start=1):
                label = f"Dictionary {number}/{len(pending)} — {entry.name} (phase {entry.output_id})"
                def progress(value, message, label=label):
                    self._check_job_cancelled()
                    text = f"{label}: {message}"
                    self._post_ui(lambda: self._set_dictionary_progress(value, text))
                progress(0, "Starting generation…")
                notes.append(self.session.generate_phase_dictionary(entry.key, resolution_deg=spacing,
                             software_binning=binning, progress_callback=progress))
            return " ".join(notes) or "All enabled phase dictionaries are ready."
        self._run_threaded(action, on_success=lambda _msg: self._refresh_phase_table())

    def _load_phase_dictionary(self):
        if getattr(self, "_worker_thread", None) is not None:
            return
        try:
            key = self._selected_phase_key()
        except ValueError as exc:
            messagebox.showerror("Select a phase", str(exc))
            return
        path = filedialog.askopenfilename(title="Load dictionary for selected phase", filetypes=[("Dictionary", "*.h5 *.hdf5")])
        if path:
            self._run_threaded(lambda: self.session.load_phase_dictionary(key, path),
                               on_success=lambda _msg: self._refresh_phase_table())

    def _save_phase_dictionary(self):
        if getattr(self, "_worker_thread", None) is not None:
            return
        try:
            key = self._selected_phase_key()
        except ValueError as exc:
            messagebox.showerror("Select a phase", str(exc))
            return
        entry = self.session.phase_registry.by_key(key)
        cache = self.session.phase_dictionaries.get(key)
        if cache is None:
            messagebox.showerror("Save dictionary", "Generate or load a dictionary for this phase first.")
            return
        master_stem = Path(entry.master_path).stem if entry.master_path else entry.name
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_"
                            for ch in master_stem).strip("_") or "master_pattern"
        resolution = f"{cache.resolution_deg:g}".replace(".", "p")
        energy_suffix = "_globalMC" if cache.master_energy_mode == "global_weighted" else ""
        filename = f"{safe_stem}_dictionary{energy_suffix}_bin{cache.software_binning}_{resolution}deg"
        path = filedialog.asksaveasfilename(title=f"Save {entry.name} dictionary", defaultextension=".h5",
                                           initialfile=filename,
                                           filetypes=[("HDF5 dictionary", "*.h5")])
        if path:
            self._run_threaded(lambda: self.session.save_phase_dictionary(key, path),
                               on_success=lambda _msg: self._refresh_phase_table())

    def _edit_phase_appearance(self):
        from tkinter import simpledialog, colorchooser
        if getattr(self, "_worker_thread", None) is not None:
            return
        try:
            entry = self.session.phase_registry.by_key(self._selected_phase_key())
        except ValueError:
            return
        name = simpledialog.askstring("Phase display name", "Name", initialvalue=entry.name, parent=self)
        if name is None:
            return
        if name.strip():
            entry.name = name.strip()
        _rgb, color = colorchooser.askcolor(entry.color, title=f"Color for {entry.name}", parent=self)
        if color:
            entry.color = color
        self._refresh_context_summary()

    def _show_phase_maps(self):
        if self.session.data is None:
            return
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from orix.plot import IPFColorKeyTSL
        from orix.vector import Vector3d
        window = tk.Toplevel(self)
        window.title("Phase maps, fitted contributions and IPF keys")
        window.geometry("1050x700")
        controls = ttk.Frame(window, padding=8); controls.pack(fill=tk.X)
        layer = tk.StringVar(value="Primary phase")
        layers = [*self.session.phase_map_layers(), "Primary IPF", "Residual IPF"]
        combo = ttk.Combobox(controls, textvariable=layer, values=layers, state="readonly", width=38)
        combo.pack(side=tk.LEFT, padx=4)
        phase_labels = {f"{entry.name} [{entry.output_id}]":entry for entry in self.session.phase_registry.entries}
        selected = tk.StringVar(value="All phases")
        filter_box = ttk.Combobox(controls, textvariable=selected, values=["All phases", *phase_labels], state="readonly", width=26)
        filter_box.pack(side=tk.LEFT, padx=4)
        body=ttk.Frame(window); body.pack(fill=tk.BOTH, expand=True)
        fig=Figure(figsize=(7,5)); axis=fig.add_subplot(111)
        canvas=FigureCanvasTkAgg(fig,master=body);canvas.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        key_frame=ttk.Frame(body);key_frame.pack(side=tk.RIGHT,fill=tk.Y)
        key_figures=[]
        colorbars=[]
        def clear_key():
            import matplotlib.pyplot as plt
            for child in key_frame.winfo_children():child.destroy()
            for figure in key_figures:plt.close(figure)
            key_figures.clear()
        def draw(_event=None):
            if getattr(self, "_worker_thread", None) is not None:
                return
            for bar in colorbars:
                bar.remove()
            colorbars.clear()
            axis.clear();clear_key()
            entry=phase_labels.get(selected.get())
            label=layer.get()
            direction,_=self._selected_ipf_direction()
            secondary=label.startswith("Residual")
            ids=self.session.residual_phases if secondary else self.session.current_phases
            if label.endswith("IPF"):
                image=(None if secondary and self.session.residual_eulers_rad is None else
                       self.session.get_residual_ipf_color_map(direction=direction) if secondary else
                       self.session.get_ipf_color_map(direction=direction))
                image = np.ones((self.session.data.rows,self.session.data.cols,3)) if image is None else image.copy()
                if entry is not None and ids is not None:
                    image[np.asarray(ids).reshape(image.shape[:2]) != entry.output_id] = 1
                axis.imshow(image)
            else:
                values=self.session.phase_layer_map(label).copy()
                if entry is not None and ids is not None:
                    values[np.asarray(ids).reshape(values.shape) != entry.output_id] = np.nan
                legend=self.session.phase_map_legend(label)
                if legend:
                    categories=np.full_like(values,np.nan)
                    for i,(pid,_,_) in enumerate(legend):categories[values==pid]=i
                    axis.imshow(categories,cmap=ListedColormap([color for _,_,color in legend]),vmin=-.5,vmax=len(legend)-.5)
                    axis.legend(handles=[Patch(color=color,label=name) for _,name,color in legend],loc="upper left",fontsize=8)
                else:
                    plot = axis.imshow(values,cmap="viridis",vmin=0 if label.startswith("Contribution:") else None,
                                       vmax=1 if label.startswith("Contribution:") else None)
                    colorbars.append(fig.colorbar(plot, ax=axis, fraction=.046, pad=.04))
            axis.set_title(label);axis.set_axis_off();canvas.draw_idle()
            if entry is not None:
                symmetry=self.session.data.phase_symmetries.get(entry.output_id)
                if symmetry is not None:
                    key=IPFColorKeyTSL(symmetry, direction=getattr(Vector3d,direction+'vector')())
                    key_fig=key.plot(return_figure=True);key_figures.append(key_fig)
                    key_fig.set_size_inches(3,3)
                    key_canvas=FigureCanvasTkAgg(key_fig,master=key_frame)
                    key_canvas.get_tk_widget().pack();key_canvas.draw_idle()
                    ttk.Label(key_frame,text=f"{entry.name} · {direction.upper()} IPF key",wraplength=240).pack()
            else:
                ttk.Label(key_frame,text="Select a phase to show its IPF color key.",wraplength=220).pack(padx=8,pady=10)
            ttk.Label(key_frame,text="Contributions are fitted pattern intensities, not phase volume fractions.\n\nOverlap acceptance uses provisional fit diagnostics; it is not a calibrated probability.",wraplength=220).pack(padx=8,pady=10)
        combo.bind("<<ComboboxSelected>>",draw);filter_box.bind("<<ComboboxSelected>>",draw)
        ttk.Button(controls,text="Refresh",command=draw).pack(side=tk.LEFT,padx=4)
        def close():
            clear_key();window.destroy()
        window.protocol("WM_DELETE_WINDOW",close)
        draw()
