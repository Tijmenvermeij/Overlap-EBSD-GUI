"""Metadata-only H5OINA loading without allocating a fictitious pattern stack."""
import h5py
import numpy as np


def load_map_only_signal(path):
    import dask.array as da
    import kikuchipy as kp
    from .core import _h5oina_analysis_roots, _is_h5oina_pattern_dataset
    with h5py.File(path, "r") as h5:
        roots = _h5oina_analysis_roots(h5)
        for root in roots:
            if f"{root}/EBSD/Data/Euler" not in h5:
                continue
            header = h5.get(f"{root}/EBSD/Header")
            if header is None:
                continue
            data = h5[f"{root}/EBSD/Data"]
            if any(_is_h5oina_pattern_dataset(f"{root}/EBSD/Data/{name}") for name in data):
                # A corrupt pattern-bearing file must not silently become map-only.
                raise ValueError("Pattern data exists but could not be loaded.")
            def scalar(name):
                if name not in header:
                    raise ValueError(f"Map-only file is missing {name}.")
                return int(np.asarray(header[name][()]).reshape(-1)[0])
            rows, cols = scalar("Y Cells"), scalar("X Cells")
            h, w = scalar("Pattern Height"), scalar("Pattern Width")
            if min(rows, cols, h, w) <= 0:
                raise ValueError("Invalid map or detector dimensions.")
            # Only a shape proxy is needed by the existing geometry loader.
            # Pattern access is blocked by LoadedInputData.patterns_available.
            signal = kp.signals.LazyEBSD(da.empty((rows, cols, h, w), chunks=(1, 1, h, w), dtype=np.uint8))
            pc = []
            for name in ("Pattern Center X", "Pattern Center Y", "Detector Distance"):
                if name in data:
                    pc.append(np.asarray(data[name][()]).reshape(-1))
                elif name in header:
                    pc.append(np.full(rows * cols, float(np.asarray(header[name][()]).reshape(-1)[0])))
                else:
                    raise ValueError(f"Map-only file is missing {name}.")
            pcs = np.stack(pc, axis=1)
            tilt = np.rad2deg(float(np.asarray(header['Tilt Angle'][()]).reshape(-1)[0])) if 'Tilt Angle' in header else 70.
            signal.detector = kp.detectors.EBSDDetector(shape=(h,w),pc=pcs.reshape(rows,cols,3),convention='oxford',sample_tilt=tilt)
            for axis, label in zip(signal.axes_manager.navigation_axes, ("X Step", "Y Step")):
                if label in header:
                    axis.scale=float(np.asarray(header[label][()]).reshape(-1)[0])
                axis.units="um"
            return signal
    raise ValueError("No supported map-only H5OINA analysis was found.")
