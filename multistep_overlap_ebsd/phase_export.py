"""Standard component phase catalogs (no synthetic overlap phases)."""
import json
import re

import h5py
import numpy as np

LAUE = {"-1": 1, "2/m": 2, "mmm": 3, "-3": 4, "-3m": 5, "4/m": 6,
        "4/mmm": 7, "6/m": 8, "6/mmm": 9, "m-3": 10, "m-3m": 11}
ANG_SYMMETRY = {1: "1", 2: "2", 3: "22", 4: "3", 5: "32", 6: "4", 7: "42", 8: "6", 9: "62", 10: "23", 11: "43"}


def phase_metadata(entry, master=None):
    metadata = dict(entry.structure.get("h5oina", {}))
    structure = entry.structure.get("master", entry.structure)
    lattice = structure.get("lattice_angstrom_degrees")
    if not metadata and lattice is not None:
        metadata.update({"Lattice Dimensions": lattice[:3], "Lattice Angles": np.deg2rad(lattice[3:]).tolist()})
    if "Laue Group" not in metadata:
        group = getattr(getattr(master, "phase", None), "point_group", None)
        laue = getattr(getattr(group, "laue", None), "name", structure.get("point_group"))
        if laue in LAUE:
            metadata["Laue Group"] = LAUE[laue]
    if "Space Group" not in metadata and structure.get("space_group") is not None:
        metadata["Space Group"] = structure["space_group"]
    required = ("Lattice Dimensions", "Lattice Angles", "Laue Group")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"{entry.name}: cannot export incomplete crystal metadata ({', '.join(missing)}).")
    dimensions = np.asarray(metadata["Lattice Dimensions"], dtype=float).reshape(3)
    angles = np.asarray(metadata["Lattice Angles"], dtype=float).reshape(3)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0) or not np.all(np.isfinite(angles)) or np.any(angles <= 0) or np.any(angles >= np.pi):
        raise ValueError(f"{entry.name}: invalid crystal lattice metadata.")
    metadata["Phase Name"] = entry.name
    metadata.setdefault("Reference", "Master pattern metadata")
    return metadata


def write_h5_phase_catalog(h5, roots, registry, masters):
    catalogs = [(entry, phase_metadata(entry, masters.get(entry.key))) for entry in registry.entries]
    for root in roots:
        for header in ("EBSD/Header", "Data Processing/Header"):
            parent = h5.require_group(f"{root}/{header}/Phases".strip("/"))
            for entry, metadata in catalogs:
                group = parent.require_group(str(entry.output_id))
                for key, value in metadata.items():
                    if key in group:
                        del group[key]
                    if isinstance(value, str) or isinstance(value, list) and value and isinstance(value[0], str):
                        group.create_dataset(key, data=value, dtype=h5py.string_dtype("utf-8"))
                    else:
                        dataset = group.create_dataset(key, data=value)
                        if key == "Lattice Dimensions":
                            dataset.attrs["Unit"] = "angstrom"
                        elif key == "Lattice Angles":
                            dataset.attrs["Unit"] = "rad"
        h5.require_group(f"{root}/EBSD/Header".strip("/")).attrs["Overlap Phase Registry"] = registry.to_json()


def ang_phase_header(header, registry, masters):
    # Replace phase declarations while preserving acquisition/scan metadata.
    phase_fields = re.compile(r"^#\s*(Phase\s+\d|MaterialName\b|Formula\b|Info\b|Symmetry\b|LatticeConstants\b|NumberFamilies\b|hklFamilies\b|ElasticConstants\b|Categories\b)", re.I)
    kept = [line for line in header if not phase_fields.match(line)]
    phase_lines = []
    for entry in registry.entries:
        m = phase_metadata(entry, masters.get(entry.key))
        laue = int(np.asarray(m["Laue Group"]).reshape(-1)[0])
        if laue not in ANG_SYMMETRY:
            raise ValueError(f"Unsupported Laue group {laue} for ANG export.")
        values = [*np.asarray(m["Lattice Dimensions"]).reshape(3), *np.rad2deg(np.asarray(m["Lattice Angles"]).reshape(3))]
        name = entry.name.replace("\n", " ").replace("\r", " ")
        phase_lines.extend([f"# Phase {entry.output_id}", f"# MaterialName {name}", "# Formula", "# Info",
                            f"# Symmetry {ANG_SYMMETRY[laue]}", "# LatticeConstants " + " ".join(f"{v:.9g}" for v in values),
                            "# NumberFamilies 0", "# Categories0 0 0 0 0"])
    return [*phase_lines, "# CI column contains pattern-matching NCC, not EDAX confidence index.", *kept]
