"""Phase-specific session contexts and multi-phase candidate competition.

Views borrow input handles but own mutable analysis state. They never replace
an active session's master during a run, and never close borrowed resources.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np

from .phases import DictionaryAsset, DictionaryProvenance, PhaseRegistry
from .cpu_indexing import cpu_job


def phase_structure(phase) -> dict:
    """Only record known metadata; never supply fictitious unit cells."""
    result = {}
    if phase is None:
        return result
    group = getattr(phase, "space_group", None)
    if group is not None:
        result["space_group"] = int(group.number)
    group = getattr(phase, "point_group", None)
    if group is not None:
        result["point_group"] = str(group.name)
    structure = getattr(phase, "structure", None)
    lattice = getattr(structure, "lattice", None)
    if lattice is not None:
        # These properties work across diffpy versions, including releases
        # predating cell_parms() (which used abcABG() instead).
        result["lattice_angstrom_degrees"] = [
            float(getattr(lattice, name)) for name in ("a", "b", "c", "alpha", "beta", "gamma")
        ]
    if structure is not None:
        result["atoms"] = [dict(element=str(a.element), xyz=[float(v) for v in a.xyz],
                                 occupancy=float(a.occupancy)) for a in structure]
    return result


class MultiPhaseSession:
    def _init_phases(self):
        self.phase_registry = PhaseRegistry()
        self.phase_masters = {}
        self.phase_dictionaries = {}
        self._phase_preparations = {}
        self.phase_candidates = {}
        self.residual_phase_candidates = {}
        self.phase_score_gap = None
        self.phase_search_revision = None
        self.residual_phase_search_revision = None
        self.keep_imported_phase_assignments = False
        self._borrowed_phase_context = False

    def _phase_context(self, key: str, *, private_arrays=True):
        entry = self.phase_registry.by_key(key)
        master = self.phase_masters.get(key)
        if master is None:
            raise ValueError(f"{entry.name}: load its master pattern in tab 1.")
        view = copy(self)
        view._borrowed_phase_context = True
        view.master = copy(master)
        view.dictionary_cache = self.phase_dictionaries.get(key)
        view._dictionary_preparation = self._phase_preparations.get(key)
        view._refinement_master_cache = None
        view._orientation_color_cache = {}
        view.residual_point_results = {}
        view.overlap_mixture_results = {}
        view._residual_pattern_store = None
        view._residual_pattern_source_cache = None
        view._residual_inspection_indices = OrderedDict()
        view._mixture_inspection_indices = OrderedDict()
        view.last_overlap = view.last_overlap_mixture = None
        if not private_arrays:
            return view
        # Mutating legacy routines operate only on private arrays.
        for name in ("current_eulers_rad", "current_phases", "current_pc_bruker", "current_pc_custom", "last_scores_map", "indexed_mask",
                     "indexed_candidate_eulers_rad", "residual_eulers_rad", "residual_phases",
                     "last_residual_scores_map", "residual_candidate_eulers_rad",
                     "overlap_primary_fraction_map", "overlap_secondary_fraction_map", "overlap_mixture_ncc_map"):
            value = getattr(self, name, None)
            setattr(view, name, None if value is None else value.copy())
        return view

    def _import_phase_registry(self):
        if self.phase_masters:
            self.master = None
        for cache in self.phase_dictionaries.values():
            if cache.owns_storage and cache.storage_path:
                Path(cache.storage_path).unlink(missing_ok=True)
        self.phase_masters.clear()
        self.phase_dictionaries.clear()
        self._phase_preparations.clear()
        self.phase_registry = PhaseRegistry()
        self.phase_candidates = {}
        self.residual_phase_candidates = {}
        self.phase_score_gap = None
        self.phase_search_revision = None
        self.residual_phase_search_revision = None
        if self.data is None:
            return
        metadata = {}
        if self.data.source_type == "h5oina":
            with h5py.File(self.data.pattern_path, "r") as h5:
                root = self.data.h5_analysis_root
                paths = [f"{root}/{header}/Phases" for header in ("Data Processing/Header", "EBSD/Header")]
                path = next((p for p in paths if p in h5), "")
                header = h5.get(f"{root}/EBSD/Header")
                saved = header.attrs.get("Overlap Phase Registry") if header is not None else None
                if saved:
                    self.phase_registry = PhaseRegistry.from_json(saved)
                    for entry in self.phase_registry.entries:
                        entry.input_ids = [entry.output_id]
                    return
                if path in h5:
                    for label, group in h5[path].items():
                        try:
                            pid = int(label)
                        except ValueError:
                            continue
                        values = {}
                        for name, dataset in group.items():
                            if not isinstance(dataset, h5py.Dataset):
                                continue
                            raw = np.asarray(dataset[()])
                            if raw.dtype.kind in "SO":
                                values[name] = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                                                for v in raw.reshape(-1)]
                            else:
                                values[name] = raw.tolist()
                        metadata[pid] = values
        else:
            from .core import _ang_phase_metadata, _symmetry_from_ang_name
            from .phase_export import LAUE
            for pid, phase in _ang_phase_metadata(self.data.ang_header_lines or []).items():
                values = {"Phase Name": phase.get("name", f"Phase {pid}"), "Reference": phase.get("reference", "Imported from ANG")}
                if "lattice_dimensions" in phase:
                    values["Lattice Dimensions"] = np.asarray(phase["lattice_dimensions"]).tolist()
                if "lattice_angles" in phase:
                    values["Lattice Angles"] = np.asarray(phase["lattice_angles"]).tolist()
                if "symmetry" in phase:
                    group = _symmetry_from_ang_name(phase["symmetry"])
                    symbol = getattr(getattr(group, "laue", None), "name", None)
                    if symbol in LAUE:
                        values["Laue Group"] = LAUE[symbol]
                metadata[pid] = values
        ids = sorted(set(int(v) for v in self.current_phases if not self._phase_is_unindexed(v)) | set(metadata))
        for pid in ids:
            if self._phase_is_unindexed(pid):
                continue
            fields = metadata.get(pid, {})
            name = fields.get("Phase Name", fields.get("Name", [f"Phase {pid}"]))
            if isinstance(name, list):
                name = name[0] if name else f"Phase {pid}"
            self.phase_registry.add(str(name), output_id=pid if pid > 0 else max(ids, default=0) + 1, input_ids=[pid],
                                    structure={"h5oina": fields} if fields else {})

    def attach_phase_master(self, path: str, *, phase_key: str | None = None, **energy_options) -> str:
        """Explicit association preserves imported results on initial attachment."""
        if self.data is None:
            raise RuntimeError("Load input data first.")
        # A temporary owner loads the asset without invalidating the live session.
        view = copy(self)
        view._borrowed_phase_context = True
        view.master = None
        view.dictionary_cache = None
        view._dictionary_preparation = None
        view._refinement_master_cache = None
        view._orientation_color_cache = {}
        note = view.load_master(path, **energy_options)
        if view.master.kind != "kikuchipy":
            raise ValueError("Multi-phase indexing requires a Kikuchipy-readable master pattern.")
        structure = phase_structure(view.master.phase)
        if view.master.phase is None or getattr(view.master.phase, "point_group", None) is None:
            raise ValueError("The master is missing crystal phase/symmetry metadata required for indexing.")
        if phase_key is not None:
            imported = self.phase_registry.by_key(phase_key).structure.get("h5oina", {})
            from .phase_export import LAUE
            group = view.master.phase.point_group.laue.name
            if "Laue Group" in imported and group in LAUE:
                imported_group = int(np.asarray(imported["Laue Group"]).reshape(-1)[0])
                if imported_group != LAUE[group]:
                    raise ValueError("This master has a different crystal symmetry from the selected imported phase. Add it as a separate phase or select the matching phase row.")
        if phase_key is None:
            entry = self.phase_registry.add(str(getattr(view.master.phase, "name", "") or Path(path).stem),
                                             structure=structure)
        else:
            entry = self.phase_registry.by_key(phase_key)
        previous = entry.master_sha256
        self.phase_registry.bind_master(entry.key, path)
        entry.structure["master"] = structure
        view.master.phase_id = entry.output_id
        self.phase_masters[entry.key] = view.master
        if previous and previous != entry.master_sha256:
            old = self.phase_dictionaries.pop(entry.key, None)
            if old is not None and old.owns_storage and old.storage_path:
                Path(old.storage_path).unlink(missing_ok=True)
            # Changing a candidate affects the competition, including other winners.
            self._invalidate_primary_results()
            self.phase_candidates.clear()
            self.residual_phase_candidates.clear()
        if self.master is None or getattr(self.master, "phase_id", None) == entry.output_id:
            self.master = view.master
        symmetry = getattr(view.master.phase, "point_group", None)
        if symmetry is not None:
            self.data.phase_symmetries[entry.output_id] = symmetry
        self._invalidate_orientation_cache()
        return f"{entry.name}: {note}"

    def phase_dictionary_provenance(self, key, cache):
        entry = self.phase_registry.by_key(key)
        master = self.phase_masters[key]
        def values(value):
            return None if value is None else np.asarray(value).tolist()
        center_idx = self.index_from_row_col(int(round((self.data.rows - 1) / 2.0)), int(round((self.data.cols - 1) / 2.0)))
        settings = dict(pc_bruker=values(self.current_pc_bruker[center_idx]), software_binning=cache.software_binning,
                        crop_extent=list(cache.crop_extent), pattern_shape=list(cache.pattern_shape),
                        resolution_deg=cache.resolution_deg, dtype=cache.pattern_dtype,
                        sample_tilt=self.data.sample_tilt_deg, detector_tilt=self.data.detector_tilt_deg,
                        azimuthal=self.data.azimuthal_deg, twist=self.data.twist_deg,
                        energy_mode=master.energy_mode, energy_kv=master.energy_kv,
                        energy_values=values(master.energy_values_kv), energy_weights=values(master.energy_weights),
                        energy_reference_pc=values(master.energy_reference_pc_bruker),
                        orientation_convention="kikuchipy Bunge radians", generator=1)
        return DictionaryProvenance.create(entry.master_sha256, phase_structure(master.phase), settings)

    @cpu_job
    def generate_phase_dictionary(self, key, *, resolution_deg, software_binning, progress_callback=None, parallel_cores=0):
        entry = self.phase_registry.by_key(key)
        view = self._phase_context(key)
        # Generation may dispose its old cache. The live asset remains valid until success.
        view.dictionary_cache = None
        note = view.generate_dictionary(phase_id=entry.output_id, resolution_deg=resolution_deg,
                                        software_binning=software_binning, progress_callback=progress_callback)
        cache = view.dictionary_cache
        provenance = self.phase_dictionary_provenance(key, cache)
        with h5py.File(cache.storage_path, "r+") as h5:
            h5.attrs["phase_provenance_json"] = json.dumps(asdict(provenance))
        old = self.phase_dictionaries.get(key)
        self.phase_dictionaries[key] = cache
        asset = DictionaryAsset(path=cache.storage_path, provenance=provenance, persistent=False)
        entry.dictionaries.append(asset)
        entry.active_dictionary_key = asset.key
        self.phase_registry.search_revision += 1
        if old is not None and old.owns_storage and old.storage_path != cache.storage_path:
            Path(old.storage_path).unlink(missing_ok=True)
        self.master, self.dictionary_cache = self.phase_masters[key], cache
        self.dictionary_settings = view.dictionary_settings
        return note

    def save_phase_dictionary(self, key, path):
        view = self._phase_context(key)
        note = view.save_dictionary(path)
        entry = self.phase_registry.by_key(key)
        entry.active_dictionary.path = view.dictionary_cache.storage_path
        entry.active_dictionary.persistent = True
        return note

    def dictionary_pc_difference(self, key, path):
        difference = self.dictionary_geometry_difference(key, path)
        if difference is not None:
            saved, current = difference
            if saved["detector_tilt"] == current["detector_tilt"]:
                return saved["pc_bruker"], current["pc_bruker"]
        return None

    def dictionary_geometry_difference(self, key, path):
        """Inspect reusable geometry differences without loading patterns."""
        from types import SimpleNamespace
        with h5py.File(path, "r") as h5:
            value = h5.attrs.get("phase_provenance_json")
            if not value:
                return None
            provenance = DictionaryProvenance(**json.loads(value))
            cache = SimpleNamespace(software_binning=int(h5.attrs["software_binning"]),
                crop_extent=h5["crop_extent"][()].tolist(), pattern_shape=h5["patterns"].shape[-2:],
                resolution_deg=float(h5.attrs["resolution_deg"]), pattern_dtype=np.dtype(h5["patterns"].dtype).name)
        expected = self.phase_dictionary_provenance(key, cache)
        if provenance.reusable_geometry_difference(expected):
            return tuple({key: json.loads(p.settings_json)[key] for key in ("pc_bruker", "detector_tilt")}
                         for p in (provenance, expected))
        return None

    def load_phase_dictionary(self, key, path, *, allow_legacy=False, accepted_pc_bruker=None, accepted_geometry=None):
        entry = self.phase_registry.by_key(key)
        with h5py.File(path, "r") as h5:
            value = h5.attrs.get("phase_provenance_json")
            provenance = DictionaryProvenance(**json.loads(value)) if value else None
        if provenance is None and not allow_legacy:
            raise ValueError("This dictionary has no master fingerprint. Regenerate it or explicitly link it as legacy/unverified.")
        if provenance is not None and provenance.master_sha256 != entry.master_sha256:
            raise ValueError("Dictionary belongs to a different master pattern.")
        view = self._phase_context(key)
        view.dictionary_cache = None
        note = view.load_dictionary(path)
        cache = view.dictionary_cache
        asset = DictionaryAsset(path=str(Path(path).resolve()), provenance=provenance,
                                legacy_linked=allow_legacy, accepted_pc_bruker=accepted_pc_bruker,
                                accepted_geometry=accepted_geometry)
        if provenance is not None and not asset.compatible_with(self.phase_dictionary_provenance(key, cache)):
            raise ValueError("Dictionary simulation settings or crystal metadata are incompatible.")
        # Stored numeric IDs are relocatable; the owning phase key is authoritative.
        cache.phase_id = entry.output_id
        self.phase_dictionaries[key] = cache
        entry.dictionaries.append(asset)
        entry.active_dictionary_key = asset.key
        self.phase_registry.search_revision += 1
        self.master, self.dictionary_cache = self.phase_masters[key], cache
        self.dictionary_settings = view.dictionary_settings
        return note

    def _enabled_phase_contexts(self):
        entries = sorted((e for e in self.phase_registry.entries if e.enabled), key=lambda e: e.output_id)
        if not entries:
            raise ValueError("Enable at least one phase in tab 1.")
        settings = None
        legacy_single = False
        expected_by_key = {}
        for entry in entries:
            cache = self.phase_dictionaries.get(entry.key)
            if entry.key not in self.phase_masters or cache is None:
                raise ValueError(f"{entry.name}: load its master and generate/load its dictionary in tab 1.")
            from .phases import file_fingerprint
            if file_fingerprint(entry.master_path) != entry.master_sha256:
                raise ValueError(f"{entry.name}: master file changed; relink it and regenerate its dictionary.")
            expected = self.phase_dictionary_provenance(entry.key, cache)
            expected_by_key[entry.key] = expected
            asset = entry.active_dictionary
            legacy_allowed = len(entries) == 1 and asset is not None and asset.legacy_linked and asset.provenance is None
            legacy_single = legacy_single or legacy_allowed
            if not legacy_allowed and (asset is None or entry.status(expected) != "Ready"):
                raise ValueError(f"{entry.name}: dictionary is incompatible or unverified; regenerate it.")
            shared = json.loads(expected.settings_json)
            # Structure-dependent dictionary length is deliberately not compared.
            if settings is not None and shared != settings:
                raise ValueError("Phase dictionaries must share geometry, energy and sampling settings. Regenerate them with shared settings.")
            settings = shared
        self._last_phase_run = () if legacy_single else self.phase_registry.snapshot(expected_by_key)
        return entries

    def index_enabled_phases(self, indices, *, keep_n=4, residual=False, progress_callback=None):
        """Compare every phase on identical pixels; commit only complete batches."""
        entries = self._enabled_phase_contexts()
        keep_imported = self.keep_imported_phase_assignments
        selected = np.unique(np.asarray(indices, dtype=np.int64).ravel())
        self._pattern_selection_from_indices(selected)
        stores = self.residual_phase_candidates if residual else self.phase_candidates
        if self.phase_score_gap is None:
            self.phase_score_gap = np.full(self.data.count, np.nan, dtype=np.float32)
        if residual:
            self._ensure_residual_state()
        completed = []
        # Orientation candidates are small; simulated/experimental images remain batched.
        # Bound score/selection temporaries as well as full-size input images.
        # Larger batches amortize dictionary reads and expose more matching tasks.
        batch_size = min(self._dictionary_index_batch_size(
                self.phase_dictionaries[entry.key], len(selected),
                n_per_iteration=self._dictionary_n_per_iteration(
                    self.phase_dictionaries[entry.key],
                    self._signal_mask_for_dictionary_cache(self.phase_dictionaries[entry.key])))
              for entry in entries)
        batch_count = (len(selected) + batch_size - 1) // batch_size
        for start in range(0, len(selected), batch_size):
            batch_number = start // batch_size + 1
            batch = selected[start:start + batch_size]
            outcomes = []
            signal = None
            for number, entry in enumerate(entries):
                if progress_callback:
                    progress_callback(100 * (start + number * len(batch) / len(entries)) / len(selected),
                                      f"Batch {batch_number}/{batch_count} — matching {entry.name} "
                                      f"(phase {number + 1}/{len(entries)}): points {start + 1}–{start + len(batch)}/{len(selected)}")
                view = self._phase_context(entry.key, private_arrays=False)
                cache = view.dictionary_cache
                if signal is None:
                    # Enabled dictionaries have identical geometry and binning.
                    # Materialize once so lazy reads/background/binning are not
                    # repeated for every phase in this batch.
                    signal = (self._residual_signal_from_indices(batch, dictionary_cache=cache)
                              if residual else self._signal_from_indices(batch, software_binning=cache.software_binning,
                                                                         crop_extent=cache.crop_extent))
                    signal = self._materialize_signal_batch(signal)
                mask = view._signal_mask_for_dictionary_cache(cache)
                def matching_progress(fraction):
                    if progress_callback:
                        progress_callback(100 * (start + (number + fraction) * len(batch) / len(entries)) / len(selected),
                                          f"Batch {batch_number}/{batch_count} — matching {entry.name} "
                                          f"(phase {number + 1}/{len(entries)}): {fraction:.0%} of current batch")
                eulers, scores, candidates, candidate_scores = view._dictionary_index_kikuchipy_signal(
                    signal, cache=cache, keep_n=min(keep_n, cache.rotation_count), signal_mask=mask,
                    n_per_iteration=view._dictionary_n_per_iteration(cache, mask), progress_callback=matching_progress)
                self._phase_preparations[entry.key] = view._dictionary_preparation
                scores = np.asarray(scores[:len(batch)], dtype=float)
                if keep_imported and not residual:
                    assigned = self.data.phases[batch]
                    scores[(~self._phase_unindexed_mask(assigned)) & (~np.isin(assigned, entry.input_ids or [entry.output_id]))] = np.nan
                outcomes.append((view._eulers_from_kikuchipy_frame(eulers[:len(batch)]), scores,
                                 candidates[:len(batch)].copy(), candidate_scores[:len(batch)].copy()))
            scores = np.stack([o[1] for o in outcomes], axis=1)
            comparable = np.where(np.isfinite(scores), scores, -np.inf)
            winners = comparable.argmax(axis=1)
            valid = np.isfinite(comparable.max(axis=1))
            if not np.all(valid):
                raise ValueError("No finite phase match for some points; the incomplete batch was not committed.")
            if residual:
                self._invalidate_overlap_mixture_cache(batch)
            else:
                self._invalidate_residual_cache(batch)
            for n, entry in enumerate(entries):
                records = stores.setdefault(entry.key, {})
                for pos, idx in enumerate(batch):
                    records[int(idx)] = (outcomes[n][2][pos].copy(), outcomes[n][3][pos].copy())
            for pos, idx in enumerate(batch):
                winner = winners[pos]
                entry = entries[winner]
                euler, score = outcomes[winner][0][pos], scores[pos, winner]
                if residual:
                    self.residual_eulers_rad[idx] = euler
                    self.residual_phases[idx] = entry.output_id
                    self.last_residual_scores_map.reshape(-1)[idx] = score
                    result = self.residual_point_results.get(int(idx))
                    if result is not None:
                        result.secondary_euler_rad = euler.copy()
                        result.secondary_phase_key = entry.key
                        result.secondary_dictionary_ncc_kp = result.secondary_ncc_kp = float(score)
                        result.secondary_simulated = None
                        result.secondary_ncc_full = None
                        result.secondary_refined = False
                else:
                    self.current_eulers_rad[idx] = euler
                    self.current_phases[idx] = entry.output_id
                    self.last_scores_map.reshape(-1)[idx] = score
                    self.indexed_mask[idx] = True
                    ranked = np.sort(comparable[pos])
                    self.phase_score_gap[idx] = ranked[-1] - ranked[-2] if len(entries) > 1 else np.nan
            completed.extend(batch.tolist())
            if residual:
                self.last_residual_indexed_indices = np.asarray(completed, dtype=np.int64)
            else:
                self.last_indexed_indices = np.asarray(completed, dtype=np.int64)
            if residual:
                self.residual_phase_search_revision = self.phase_registry.search_revision
            else:
                self.phase_search_revision = self.phase_registry.search_revision
            self._invalidate_orientation_cache()
            self._invalidate_residual_color_cache()
            if progress_callback:
                progress_callback(100 * len(completed) / len(selected),
                                  f"Batch {batch_number}/{batch_count} complete — indexed "
                                  f"{len(completed)}/{len(selected)} points across {len(entries)} phases")
        return f"Indexed {len(selected)} points across {len(entries)} enabled phases."

    def _entry_for_phase_id(self, phase_id):
        if int(phase_id) == 0 and self.data is not None and self.data.source_type == "up_ang":
            return next((e for e in self.phase_registry.entries if 0 in e.input_ids), None)
        return self.phase_registry.by_output_id(int(phase_id))

    def _phase_context_for_index(self, index, *, secondary=False):
        ids = self.residual_phases if secondary else self.current_phases
        phase_id = int(ids[int(index)])
        entry = self._entry_for_phase_id(phase_id)
        if entry is None:
            raise ValueError(f"Phase {phase_id} has no master association. Link it in tab 1.")
        return self._phase_context(entry.key, private_arrays=False)

    def refine_enabled_phases(self, indices, *, residual=False, trust_euler_deg=1.0,
                              maxfev=50, use_full_resolution=True, progress_callback=None, parallel_cores=0):
        entries = self._enabled_phase_contexts()
        selected = np.unique(np.asarray(indices, dtype=np.int64).ravel())
        self._pattern_selection_from_indices(selected)
        stores = self.residual_phase_candidates if residual else self.phase_candidates
        if not any(stores.get(e.key) for e in entries):
            # Imported orientations remain valid seeds without a dictionary search.
            ids = self.residual_phases if residual else self.current_phases
            angles = self.residual_eulers_rad if residual else self.current_eulers_rad
            for entry in entries:
                stores.setdefault(entry.key, {}).update({int(i): (angles[i:i+1].copy(), np.array([np.nan]))
                                                         for i in selected if self._entry_for_phase_id(ids[i]) is entry and np.all(np.isfinite(angles[i]))})
        # Keep only prepared masters across batches in this run. Mutable point
        # state stays private; a later run rechecks the source and energy model.
        prepared_masters = {}
        for start in range(0, len(selected), 128):
            batch = selected[start:start + 128]
            outcomes = []
            for n, entry in enumerate(entries):
                rows = stores.get(entry.key, {})
                work = np.array([i for i in batch if int(i) in rows], dtype=np.int64)
                if not work.size:
                    continue
                if progress_callback:
                    progress_callback(100 * (start + n * len(batch) / len(entries)) / len(selected),
                                      f"Refining {entry.name}: {len(work)} points")
                view = self._phase_context(entry.key)
                view._refinement_master_cache = prepared_masters.get(entry.key)
                k = max(len(rows[int(i)][0]) for i in work)
                candidates = np.stack([np.pad(rows[int(i)][0], ((0, k-len(rows[int(i)][0])),(0,0)), mode="edge") for i in work])
                k = candidates.shape[1]
                view.current_phases[work] = entry.output_id
                if residual:
                    from copy import deepcopy
                    view.residual_point_results = {int(i): deepcopy(self.residual_point_results[int(i)])
                                                   for i in work if int(i) in self.residual_point_results}
                    # Residual reconstruction always uses the original primary master.
                    view._simulate_patterns_for_eulers = self._simulate_patterns_for_eulers
                    view._residual_pattern_store = self._residual_pattern_store
                    view.residual_candidate_eulers_rad = np.full((self.data.count, k, 3), np.nan)
                    view.residual_candidate_eulers_rad[work] = candidates
                    view.residual_eulers_rad[work] = candidates[:, 0]
                    refined = view._batch_refine_residual_points(work, trust_euler_deg=trust_euler_deg,
                                                                 maxfev=maxfev, use_full_resolution=use_full_resolution)
                    values = [(r.index, r.secondary_euler_rad, r.secondary_ncc_kp, r) for r in refined]
                else:
                    view.current_eulers_rad[work] = candidates[:, 0]
                    view.indexed_candidate_eulers_rad = np.full((self.data.count, k, 3), np.nan)
                    view.indexed_candidate_eulers_rad[work] = candidates
                    view.refine_orientations_indices(work, phase_id=entry.output_id,
                                                     trust_euler_deg=trust_euler_deg, maxfev=maxfev,
                                                     use_full_resolution=use_full_resolution, parallel_cores=parallel_cores)
                    values = [(int(i), view.current_eulers_rad[i], view.last_scores_map.reshape(-1)[i], None) for i in work]
                prepared_masters[entry.key] = view._refinement_master_cache
                for i, euler, score, result in values:
                    if score is not None and np.isfinite(score):
                        stores[entry.key][int(i)] = (np.asarray(euler).reshape(1,3).copy(), np.asarray([score]))
                        outcomes.append((i, float(score), entry, np.asarray(euler).copy(), result))
            winners = {}
            for outcome in outcomes:
                i, score, entry, euler, result = outcome
                if i not in winners or score > winners[i][1]:
                    winners[i] = outcome
            done = np.array(sorted(winners), dtype=np.int64)
            if residual:
                self._invalidate_overlap_mixture_cache(done)
            else:
                self._invalidate_residual_cache(done)
            for i, score, entry, euler, result in winners.values():
                if residual:
                    result.secondary_phase_key = entry.key
                    primary = self._entry_for_phase_id(int(self.current_phases[i]))
                    result.primary_phase_key = primary.key if primary else None
                    self.residual_eulers_rad[i] = euler
                    self.residual_phases[i] = entry.output_id
                    self.last_residual_scores_map.reshape(-1)[i] = score
                    self._store_residual_result(result)
                else:
                    self.current_eulers_rad[i] = euler
                    self.current_phases[i] = entry.output_id
                    self.last_scores_map.reshape(-1)[i] = score
                    self.indexed_mask[i] = True
                    alternatives = sorted(o[1] for o in outcomes if o[0] == i)
                    if self.phase_score_gap is not None:
                        self.phase_score_gap[i] = alternatives[-1] - alternatives[-2] if len(alternatives) > 1 else np.nan
            self._invalidate_orientation_cache()
            self._invalidate_residual_color_cache()
            if progress_callback:
                progress_callback(100 * (start + len(batch)) / len(selected), f"Refined {start + len(batch)}/{len(selected)} points")
        return f"Refined phase candidates for {len(selected)} points."

    def _phase_checkpoint_arrays(self):
        arrays = {}
        for prefix, stores in (("primary", self.phase_candidates), ("secondary", self.residual_phase_candidates)):
            for key, records in stores.items():
                if not records:
                    continue
                indices = sorted(records)
                arrays[f"{prefix}_phase_{key}_indices"] = np.asarray(indices, dtype=np.int64)
                k = max(len(records[i][0]) for i in indices)
                arrays[f"{prefix}_phase_{key}_eulers"] = np.stack([np.pad(records[i][0], ((0,k-len(records[i][0])),(0,0)), mode="edge") for i in indices])
                arrays[f"{prefix}_phase_{key}_scores"] = np.stack([np.pad(records[i][1], (0,k-len(records[i][1])), mode="edge") for i in indices])
        if self.phase_score_gap is not None:
            arrays["phase_score_gap"] = self.phase_score_gap
        arrays["phase_search_revision"] = np.asarray(self.phase_search_revision if self.phase_search_revision is not None else -1)
        arrays["residual_phase_search_revision"] = np.asarray(self.residual_phase_search_revision if self.residual_phase_search_revision is not None else -1)
        arrays["keep_imported_phase_assignments"] = np.asarray(self.keep_imported_phase_assignments)
        return arrays

    def _restore_imported_phase_assets(self):
        if not any(entry.master_path for entry in self.phase_registry.entries):
            return ""
        notes = []
        self._restore_phase_checkpoint({"phase_registry_json": np.asarray(self.phase_registry.to_json())}, notes)
        return " ".join(notes)

    def _restore_phase_checkpoint(self, state, notes):
        if "phase_registry_json" not in state:
            return
        registry = PhaseRegistry.from_json(str(state["phase_registry_json"].item()))
        self.phase_registry = registry
        for entry in registry.entries:
            if not entry.master_path:
                continue
            try:
                # Reject changed files before attaching, preserving saved identity.
                from .phases import file_fingerprint
                if not Path(entry.master_path).is_file():
                    notes.append(f"{entry.name}: master missing; relink it in tab 1.")
                    continue
                if file_fingerprint(entry.master_path) != entry.master_sha256:
                    notes.append(f"{entry.name}: master content changed; relink it in tab 1.")
                    continue
                asset = entry.active_dictionary
                energy_options = {}
                if asset is not None and asset.provenance is not None:
                    settings = json.loads(asset.provenance.settings_json)
                    energy_options = dict(energy_kv=settings.get("energy_kv"), energy_mode=settings.get("energy_mode", "highest"))
                    if energy_options["energy_mode"] == "global_weighted":
                        energy_options.update(energy_values_kv=settings.get("energy_values"), energy_weights=settings.get("energy_weights"),
                                              energy_reference_pc_bruker=settings.get("energy_reference_pc"))
                self.attach_phase_master(entry.master_path, phase_key=entry.key, **energy_options)
                if asset is not None and asset.persistent and Path(asset.path).is_file():
                    try:
                        self.load_phase_dictionary(entry.key, asset.path, allow_legacy=asset.legacy_linked,
                                                   accepted_pc_bruker=asset.accepted_pc_bruker,
                                                   accepted_geometry=asset.accepted_geometry)
                        # Loading is transactional and may add an asset: retain saved version identity.
                        entry.dictionaries.pop()
                        entry.active_dictionary_key = asset.key
                    except Exception as exc:
                        notes.append(f"{entry.name}: dictionary unavailable ({exc}).")
                elif asset is not None:
                    notes.append(f"{entry.name}: dictionary missing; regenerate or relink it.")
            except Exception as exc:
                notes.append(f"{entry.name}: assets not restored ({exc}).")
        registry.search_revision = PhaseRegistry.from_json(str(state["phase_registry_json"].item())).search_revision
        for prefix, stores in (("primary", self.phase_candidates), ("secondary", self.residual_phase_candidates)):
            for entry in registry.entries:
                base = f"{prefix}_phase_{entry.key}_"
                if base + "indices" not in state:
                    continue
                indices = np.asarray(state[base + "indices"], dtype=np.int64)
                eulers, scores = state[base + "eulers"], state[base + "scores"]
                if eulers.ndim != 3 or eulers.shape[0] != len(indices) or eulers.shape[2] != 3 or scores.shape != eulers.shape[:2]:
                    raise ValueError("Invalid saved phase candidate arrays.")
                if np.any(indices < 0) or np.any(indices >= self.data.count):
                    raise ValueError("Saved phase candidate index is outside the input map.")
                stores[entry.key] = {int(i): (e.copy(), score.copy()) for i, e, score in zip(indices, eulers, scores)}
        self.phase_score_gap = np.asarray(state["phase_score_gap"], dtype=np.float32).copy() if "phase_score_gap" in state else None
        if self.phase_score_gap is not None and self.phase_score_gap.shape != (self.data.count,):
            raise ValueError("Saved phase score-gap map has the wrong shape.")
        revision = int(state["phase_search_revision"]) if "phase_search_revision" in state else -1
        self.phase_search_revision = revision if revision >= 0 else None
        revision = int(state["residual_phase_search_revision"]) if "residual_phase_search_revision" in state else -1
        self.residual_phase_search_revision = revision if revision >= 0 else None
        self.keep_imported_phase_assignments = bool(state["keep_imported_phase_assignments"]) if "keep_imported_phase_assignments" in state else False

    def phase_map_layers(self):
        return ["Primary phase", "Residual phase", "Phase pair", "Dominant phase", "Phase score gap", "Overlap acceptance", "Mixture NCC improvement",
                *[f"Contribution: {e.name} [{e.output_id}]" for e in self.phase_registry.entries]]

    def phase_layer_map(self, label):
        shape = (self.data.rows, self.data.cols)
        values = np.full(self.data.count, np.nan, dtype=np.float32)
        if label == "Primary phase":
            values = self.current_phases.astype(np.float32).copy()
            values[values <= 0] = np.nan
        elif label == "Residual phase":
            if self.residual_phases is not None and self.residual_eulers_rad is not None:
                valid = np.all(np.isfinite(self.residual_eulers_rad), axis=1)
                values[valid] = self.residual_phases[valid]
        elif label == "Phase score gap":
            if self.phase_score_gap is not None:
                values = self.phase_score_gap.copy()
        elif label in ("Overlap acceptance", "Mixture NCC improvement"):
            for index, result in self.overlap_mixture_results.items():
                if label == "Mixture NCC improvement":
                    values[index] = result.overlap_ncc_improvement if result.overlap_ncc_improvement is not None else np.nan
                elif result.overlap_accepted is not None:
                    values[index] = 2 if result.phase_pair_ambiguous else int(result.overlap_accepted)
        elif label in ("Phase pair", "Dominant phase") or label.startswith("Contribution: "):
            entries = sorted(self.phase_registry.entries, key=lambda e: e.output_id)
            pairs = {(a.key,b.key): n+1 for n,(a,b) in enumerate((a,b) for i,a in enumerate(entries) for b in entries[i:])}
            order = {e.key:i for i,e in enumerate(entries)}
            for index, result in self.overlap_mixture_results.items():
                a, b = result.primary_phase_key, result.secondary_phase_key
                if a not in order or b not in order:
                    continue
                if label == "Phase pair":
                    values[index] = pairs[tuple(sorted((a,b), key=order.get))]
                elif label == "Dominant phase":
                    key = a if result.primary_fraction >= result.secondary_fraction else b
                    values[index] = self.phase_registry.by_key(key).output_id
                else:
                    phase_id = int(label.rsplit('[',1)[1].rstrip(']'))
                    key = self.phase_registry.by_output_id(phase_id).key
                    values[index] = (result.primary_fraction if a == key else 0) + (result.secondary_fraction if b == key else 0)
        else:
            raise KeyError(label)
        return values.reshape(shape)

    def phase_map_legend(self, label):
        entries = sorted(self.phase_registry.entries, key=lambda e:e.output_id)
        if label in ("Phase", "Primary phase", "Residual phase", "Dominant phase"):
            return [(e.output_id, f"{e.name} [{e.output_id}]", e.color) for e in entries]
        if label == "Overlap acceptance":
            return [(0,"Not accepted", "#bbbbbb"), (1,"Accepted", "#228833"), (2,"Ambiguous", "#eeaa33")]
        if label == "Phase pair":
            from matplotlib.colors import to_hex, to_rgb
            return [(n+1, f"{a.name} [{a.output_id}] + {b.name} [{b.output_id}]",
                     to_hex((np.asarray(to_rgb(a.color)) + np.asarray(to_rgb(b.color)))/2))
                    for n,(a,b) in enumerate((a,b) for i,a in enumerate(entries) for b in entries[i:])]
        return []

    def _compare_phase_pair_fits(self, index, initial, *, fit_maxiter, fit_popsize, fit_bounds, fit_method):
        """Bounded candidate search, with explicit diagnostic acceptance criteria.

        These conservative thresholds are fit diagnostics, not calibrated phase
        probabilities or an experimental detection limit.
        """
        from .core import _overlap_mixture_result_from_raw_patterns, _overlap_point_result_from_raw_patterns
        idx = int(index)
        primary = self._entry_for_phase_id(int(self.current_phases[idx]))
        secondary = self._entry_for_phase_id(int(self.residual_phases[idx]))
        initial.primary_phase_key, initial.secondary_phase_key = primary.key, secondary.key
        def shortlist(stores, chosen, euler):
            candidates = [(chosen.key, np.asarray(euler).copy())]
            ranked = []
            for entry in self.phase_registry.entries:
                if not entry.enabled or entry.key not in self.phase_masters or entry.key == chosen.key:
                    continue
                record = stores.get(entry.key, {}).get(idx)
                if record is None:
                    continue
                eulers, scores = record
                valid = np.flatnonzero(np.isfinite(scores))
                if valid.size:
                    n = valid[np.argmax(scores[valid])]
                    ranked.append((float(scores[n]), entry.output_id, entry.key, eulers[n]))
            # Keep the chosen component and at most one competing phase each.
            for _, _, key, angle in sorted(ranked, key=lambda x:(-x[0],x[1]))[:1]:
                candidates.append((key, angle.copy()))
            return candidates
        a = shortlist(self.phase_candidates, primary, initial.primary_euler_rad)
        b = shortlist(self.residual_phase_candidates, secondary, initial.secondary_euler_rad)
        experimental = self._processed_pattern_at(idx)
        weights = self._overlap_weights()
        fits = [initial]
        simulations = {}
        def simulate(key, euler):
            token = (key, tuple(np.asarray(euler).reshape(3)))
            if token not in simulations:
                simulations[token] = self._phase_context(key, private_arrays=False)._simulate_pattern_for_euler(idx,euler)
            return simulations[token]
        options = dict(fit_maxiter=fit_maxiter, fit_popsize=fit_popsize, fit_bounds=fit_bounds, fit_method=fit_method)
        for ai, (ak, ae) in enumerate(a):
            for bi, (bk, be) in enumerate(b):
                if ai == bi == 0:
                    continue
                fitted = _overlap_mixture_result_from_raw_patterns(idx, initial.row, initial.col,
                    experimental, simulate(ak,ae), simulate(bk,be), weights,
                    primary_euler_rad=ae, secondary_euler_rad=be,
                    old_primary_ncc=initial.old_primary_ncc, old_secondary_ncc=initial.old_secondary_ncc, **options)
                fitted.primary_phase_key, fitted.secondary_phase_key = ak,bk
                fits.append(fitted)
        valid = [fit for fit in fits if np.isfinite(fit.ncc_mixture)]
        chosen = max(valid, key=lambda fit:fit.ncc_mixture) if valid else initial
        # Compare against each proposed component with the same nuisance model.
        baselines = []
        for key, euler in ((chosen.primary_phase_key, chosen.primary_euler_rad),
                           (chosen.secondary_phase_key, chosen.secondary_euler_rad)):
            single = _overlap_point_result_from_raw_patterns(idx, initial.row, initial.col,
                experimental, simulate(key,euler), weights, fit_blur_gain=True, **options)
            baselines.append(single.ncc_es)
        chosen.single_component_ncc = float(max(baselines))
        chosen.overlap_ncc_improvement = float(chosen.ncc_mixture - chosen.single_component_ncc)
        reasons = []
        if not np.isfinite(chosen.ncc_mixture) or not np.isfinite(chosen.single_component_ncc):
            reasons.append("non-finite fit score")
        if not chosen.fit_success:
            reasons.append("optimizer did not converge")
        if chosen.overlap_ncc_improvement < .01:
            reasons.append("NCC improvement below 0.01")
        if min(chosen.primary_fraction, chosen.secondary_fraction) < .05:
            reasons.append("component contribution below 5%")
        if abs(chosen.component_correlation) > .98:
            reasons.append("component simulations are nearly indistinguishable")
        ranked = sorted((fit.ncc_mixture for fit in valid), reverse=True)
        chosen.phase_pair_ambiguous = len(ranked) > 1 and ranked[0] - ranked[1] < .005
        if chosen.phase_pair_ambiguous:
            reasons.append("competing phase pair within 0.005 NCC")
        chosen.overlap_accepted = not reasons
        chosen.overlap_acceptance_note = "; ".join(reasons) if reasons else "Passes provisional fit-improvement, contribution and distinguishability checks"
        return chosen

    def _migrate_legacy_phase_assets(self):
        """Preserve a saved one-master association as explicitly unverified.

        A historical dictionary can continue alone, but is never admitted into
        a cross-phase comparison until regenerated with provenance.
        """
        if self.phase_masters or self.master is None or not Path(self.master.path).is_file():
            return
        pid = int(self.dictionary_cache.phase_id if self.dictionary_cache is not None else self.master.phase_id)
        entry = self._entry_for_phase_id(pid)
        if entry is None:
            entry = self.phase_registry.add(str(getattr(self.master.phase, "name", "") or f"Phase {pid}"),
                                             output_id=pid if pid > 0 else None, input_ids=[pid])
        self.phase_registry.bind_master(entry.key, self.master.path)
        entry.structure["master"] = phase_structure(self.master.phase)
        self.master.phase_id = entry.output_id
        self.phase_masters[entry.key] = self.master
        for other in self.phase_registry.entries:
            other.enabled = other.key == entry.key
        if self.dictionary_cache is not None:
            self.dictionary_cache.phase_id = entry.output_id
            self.phase_dictionaries[entry.key] = self.dictionary_cache
            asset = DictionaryAsset(path=self.dictionary_cache.storage_path or "", legacy_linked=True,
                                    persistent=not self.dictionary_cache.owns_storage)
            entry.dictionaries.append(asset)
            entry.active_dictionary_key = asset.key
