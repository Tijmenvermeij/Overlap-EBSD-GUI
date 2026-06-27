from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from transforms3d.euler import euler2mat

EPS = 1e-8


def normalize_zmuv(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32, copy=False)
    mean_val = float(arr.mean())
    std_val = float(arr.std())
    if std_val < EPS:
        return arr - mean_val
    return (arr - mean_val) / std_val


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize_zmuv(a)
    bb = normalize_zmuv(b)
    return float(np.mean(aa * bb))


def make_gnomonic_coordinates(h: int, w: int, pc: tuple[float, float, float]) -> np.ndarray:
    px, py, dd = pc
    x_min = -px / dd
    x_max = (1.0 - px) / dd
    y_min = -py / dd
    y_max = (h - py * w) / (dd * w)
    xs = np.linspace(x_min, x_max, w, dtype=np.float32)
    ys = np.linspace(y_max, y_min, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.vstack((xx.ravel(), yy.ravel(), np.ones(h * w, dtype=np.float32)))


def homogeneous_to_stereographic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(xyz, axis=0)
    norms = np.where(norms < EPS, 1.0, norms)
    xyz_n = xyz / norms
    xu = xyz_n[0] / (1.0 + xyz_n[2] + EPS)
    yu = xyz_n[1] / (1.0 + xyz_n[2] + EPS)
    xl = xyz_n[0] / (1.0 - xyz_n[2] + EPS)
    yl = xyz_n[1] / (1.0 - xyz_n[2] + EPS)
    return np.vstack((yu, xu)), np.vstack((yl, xl))


class XProjector:
    def __init__(self, hemi_pair: tuple[np.ndarray, np.ndarray] | dict, h: int, w: int):
        if isinstance(hemi_pair, dict):
            up = hemi_pair["up"]
            lo = hemi_pair["lo"]
            self.projection = str(hemi_pair.get("projection", "stereographic")).lower()
        else:
            up, lo = hemi_pair
            self.projection = "stereographic"
        self.up = up.astype(np.float32, copy=False)
        self.lo = lo.astype(np.float32, copy=False)
        self.size = int(self.up.shape[0])
        self.h = int(h)
        self.w = int(w)

    def project(
        self,
        euler_rad: np.ndarray,
        pc: tuple[float, float, float],
        rot_sd: np.ndarray,
        direction_cosines: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.projection == "lambert":
            return self._project_lambert(euler_rad, pc, rot_sd, direction_cosines=direction_cosines)
        return self._project_stereographic(euler_rad, pc, rot_sd, direction_cosines=direction_cosines)

    def _project_stereographic(
        self,
        euler_rad: np.ndarray,
        pc: tuple[float, float, float],
        rot_sd: np.ndarray,
        direction_cosines: np.ndarray | None = None,
    ) -> np.ndarray:
        rot_o = euler2mat(float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2]), axes="rzxz")
        if direction_cosines is None:
            xyz = (rot_o.T @ rot_sd) @ make_gnomonic_coordinates(self.h, self.w, pc)
        else:
            xyz = rot_o.T @ direction_cosines
        up_st, lo_st = homogeneous_to_stereographic(xyz)
        idx = lambda arr: (arr + 1.0) * (self.size - 1) / 2.0
        cval_sentinel = -1.0e30
        vu = map_coordinates(
            self.up,
            idx(up_st),
            order=1,
            cval=cval_sentinel,
            mode="constant",
            prefilter=False,
        )
        vl = map_coordinates(
            self.lo,
            idx(lo_st),
            order=1,
            cval=cval_sentinel,
            mode="constant",
            prefilter=False,
        )
        vu_img = vu.reshape(self.h, self.w)
        vl_img = vl.reshape(self.h, self.w)
        ru2 = up_st[0] * up_st[0] + up_st[1] * up_st[1]
        rl2 = lo_st[0] * lo_st[0] + lo_st[1] * lo_st[1]
        coord_valid_u = ru2.reshape(self.h, self.w) <= 1.0 + 1e-6
        coord_valid_l = rl2.reshape(self.h, self.w) <= 1.0 + 1e-6

        valid_u = (vu_img > (cval_sentinel * 0.5)) & coord_valid_u
        valid_l = (vl_img > (cval_sentinel * 0.5)) & coord_valid_l
        out = np.where(valid_u, vu_img, vl_img)
        invalid = ~(valid_u | valid_l)
        if np.any(invalid):
            out[invalid] = 0.0
        return out.astype(np.float32)

    @staticmethod
    def _vector_to_lambert_xy(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = v[0]
        y = v[1]
        z = v[2]
        abs_z = np.abs(z)
        sqrt_z = np.sqrt(np.clip(2.0 * (1.0 - abs_z), 0.0, None))

        lam_x = np.zeros_like(x, dtype=np.float64)
        lam_y = np.zeros_like(y, dtype=np.float64)
        not_pole = abs_z < (1.0 - 1e-15)
        branch_x = not_pole & (np.abs(y) <= np.abs(x))
        if np.any(branch_x):
            sx = np.sign(x[branch_x])
            lam_x[branch_x] = sx * sqrt_z[branch_x] * (np.sqrt(np.pi) / 2.0)
            lam_y[branch_x] = sx * sqrt_z[branch_x] * (2.0 / np.sqrt(np.pi)) * np.arctan(y[branch_x] / x[branch_x])
        branch_y = not_pole & (~branch_x)
        if np.any(branch_y):
            sy = np.sign(y[branch_y])
            lam_x[branch_y] = sy * sqrt_z[branch_y] * (2.0 / np.sqrt(np.pi)) * np.arctan(x[branch_y] / y[branch_y])
            lam_y[branch_y] = sy * sqrt_z[branch_y] * (np.sqrt(np.pi) / 2.0)

        return lam_x, lam_y

    def _project_lambert(
        self,
        euler_rad: np.ndarray,
        pc: tuple[float, float, float],
        rot_sd: np.ndarray,
        direction_cosines: np.ndarray | None = None,
    ) -> np.ndarray:
        rot_o = euler2mat(float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2]), axes="rzxz")
        if direction_cosines is None:
            xyz = (rot_o.T @ rot_sd) @ make_gnomonic_coordinates(self.h, self.w, pc)
        else:
            xyz = rot_o.T @ direction_cosines
        norms = np.linalg.norm(xyz, axis=0)
        norms = np.where(norms < EPS, 1.0, norms)
        v = xyz / norms

        lam_x, lam_y = self._vector_to_lambert_xy(v)
        scale = (self.size - 1) / 2.0
        sqrt_pi_half = np.sqrt(np.pi / 2.0)
        jj = scale * (lam_x / sqrt_pi_half) + scale
        ii = scale * (lam_y / sqrt_pi_half) + scale

        i0 = np.floor(ii).astype(np.int32)
        j0 = np.floor(jj).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, self.size - 1)
        j1 = np.clip(j0 + 1, 0, self.size - 1)
        i0 = np.clip(i0, 0, self.size - 1)
        j0 = np.clip(j0, 0, self.size - 1)
        di = ii - i0
        dj = jj - j0
        dim = 1.0 - di
        djm = 1.0 - dj

        upper = self.up
        lower = self.lo
        val_u = (
            upper[i0, j0] * dim * djm
            + upper[i1, j0] * di * djm
            + upper[i0, j1] * dim * dj
            + upper[i1, j1] * di * dj
        )
        val_l = (
            lower[i0, j0] * dim * djm
            + lower[i1, j0] * di * djm
            + lower[i0, j1] * dim * dj
            + lower[i1, j1] * di * dj
        )
        out = np.where(v[2] >= 0.0, val_u, val_l)
        return out.reshape(self.h, self.w).astype(np.float32)


def _select_emsoft_energy_index(
    f: h5py.File,
    num_energies: int,
    mode: str,
    target_beam_kv: float | None,
) -> tuple[int | None, np.ndarray | None]:
    mode_norm = mode.strip().lower()
    ekevs = None
    if "EMData/EBSDmaster/EkeVs" in f:
        ekevs = np.asarray(f["EMData/EBSDmaster/EkeVs"][()]).astype(np.float32).ravel()
        if ekevs.size != num_energies:
            ekevs = None

    if num_energies <= 1:
        return 0, np.array([1.0], dtype=np.float32)

    if mode_norm == "mc_weighted":
        weights_raw = None
        if "EMData/MCOpenCL/accum_e" in f:
            acc_e = np.asarray(f["EMData/MCOpenCL/accum_e"][()])
            if acc_e.ndim >= 1 and acc_e.shape[-1] == num_energies:
                axes = tuple(range(acc_e.ndim - 1))
                weights_raw = acc_e.sum(axis=axes).astype(np.float64)
        if weights_raw is None and "EMData/MCOpenCL/accumSP" in f:
            acc_sp = np.asarray(f["EMData/MCOpenCL/accumSP"][()])
            if acc_sp.ndim >= 1 and acc_sp.shape[-1] == num_energies:
                axes = tuple(range(acc_sp.ndim - 1))
                weights_raw = acc_sp.sum(axis=axes).astype(np.float64)
        if weights_raw is not None:
            weights_raw = np.clip(weights_raw, 0.0, None)
            sum_w = float(weights_raw.sum())
            if sum_w > EPS:
                weights = (weights_raw / sum_w).astype(np.float32)
                return None, weights

    if ekevs is not None:
        if target_beam_kv is not None and np.isfinite(target_beam_kv):
            idx = int(np.argmin(np.abs(ekevs - float(target_beam_kv))))
            return idx, None
        idx = int(np.argmax(ekevs))
        return idx, None
    return num_energies - 1, None


def _prepare_emsoft_energy_stack(
    arr: np.ndarray,
    dataset_name: str,
    expected_energies: int | None,
) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim < 2:
        raise ValueError(f"{dataset_name}: expected at least 2 dimensions, got shape {data.shape}")
    if data.ndim == 2:
        return data[np.newaxis, ...].astype(np.float32)
    if data.ndim == 3:
        return data.astype(np.float32)

    lead_shape = data.shape[:-2]
    energy_axis = None
    if expected_energies is not None:
        matches = [ax for ax, size in enumerate(lead_shape) if int(size) == int(expected_energies)]
        if len(matches) == 1:
            energy_axis = matches[0]
    if energy_axis is None:
        energy_axis = int(np.argmax(lead_shape))

    data = np.moveaxis(data, energy_axis, 0)
    if data.ndim > 3:
        collapse_axes = tuple(range(1, data.ndim - 2))
        data = data.sum(axis=collapse_axes)
    if data.ndim != 3:
        raise ValueError(f"{dataset_name}: could not reshape to (energy, H, W), got {data.shape}")
    return data.astype(np.float32)


def _extract_emsoft_hemis(
    f: h5py.File,
    up_path: str,
    lo_path: str,
    mode: str,
    target_beam_kv: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if up_path not in f or lo_path not in f:
        raise ValueError(f"EMsoft file missing required datasets: '{up_path}' and/or '{lo_path}'.")

    up_raw = np.asarray(f[up_path][()])
    lo_raw = np.asarray(f[lo_path][()])
    if up_raw.shape != lo_raw.shape:
        raise ValueError(f"EMsoft hemispheres have different shapes: {up_raw.shape} vs {lo_raw.shape}")

    expected_energies = None
    if "EMData/EBSDmaster/EkeVs" in f:
        expected_energies = int(np.asarray(f["EMData/EBSDmaster/EkeVs"][()]).size)
    up_stack = _prepare_emsoft_energy_stack(up_raw, up_path, expected_energies)
    lo_stack = _prepare_emsoft_energy_stack(lo_raw, lo_path, expected_energies)
    if up_stack.shape != lo_stack.shape:
        raise ValueError(f"Prepared EMsoft hemisphere stacks have different shapes: {up_stack.shape} vs {lo_stack.shape}")

    num_energies = int(up_stack.shape[0])
    energy_idx, weights = _select_emsoft_energy_index(f, num_energies, mode, target_beam_kv)
    if weights is not None:
        up = np.tensordot(weights, up_stack, axes=(0, 0))
        lo = np.tensordot(weights, lo_stack, axes=(0, 0))
    else:
        idx = int(energy_idx if energy_idx is not None else (num_energies - 1))
        idx = max(0, min(num_energies - 1, idx))
        up = up_stack[idx]
        lo = lo_stack[idx]
    return up.astype(np.float32), lo.astype(np.float32)


def load_master_hemis(
    master_files: list[str],
    emsoft_energy_mode: str = "highest",
    emsoft_projection_mode: str = "auto",
    target_beam_kv: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray] | dict]:
    hemis: list[tuple[np.ndarray, np.ndarray] | dict] = []
    for fn in master_files:
        with h5py.File(fn, "r") as f:
            if "hemi1" in f and "hemi2" in f:
                up = f["hemi1"][()]
                lo = f["hemi2"][()]
                hemis.append((up.astype(np.float32), lo.astype(np.float32)))
                continue
            if "Data/Master/Dynamical/Upper" in f and "Data/Master/Dynamical/Lower" in f:
                up = f["Data/Master/Dynamical/Upper"][()]
                lo = f["Data/Master/Dynamical/Lower"][()]
                hemis.append((up.astype(np.float32), lo.astype(np.float32)))
                continue
            if "EMData/EBSDmaster" in f:
                proj_mode = emsoft_projection_mode.strip().lower()
                has_sp = "EMData/EBSDmaster/masterSPNH" in f and "EMData/EBSDmaster/masterSPSH" in f
                has_lp = "EMData/EBSDmaster/mLPNH" in f and "EMData/EBSDmaster/mLPSH" in f
                if proj_mode == "auto":
                    proj_mode = "lambert" if has_lp else "stereographic"
                if proj_mode == "stereographic":
                    up, lo = _extract_emsoft_hemis(
                        f,
                        up_path="EMData/EBSDmaster/masterSPNH",
                        lo_path="EMData/EBSDmaster/masterSPSH",
                        mode=emsoft_energy_mode,
                        target_beam_kv=target_beam_kv,
                    )
                    hemis.append({"up": up, "lo": lo, "projection": "stereographic"})
                    continue
                if proj_mode == "lambert":
                    up, lo = _extract_emsoft_hemis(
                        f,
                        up_path="EMData/EBSDmaster/mLPNH",
                        lo_path="EMData/EBSDmaster/mLPSH",
                        mode=emsoft_energy_mode,
                        target_beam_kv=target_beam_kv,
                    )
                    hemis.append({"up": up, "lo": lo, "projection": "lambert"})
                    continue
            raise ValueError(
                f"Unsupported master pattern file format: {fn}. "
                "Expected Oxford/legacy hemispheres or EMsoft datasets."
            )
    return hemis


def read_edax_up_patterns(path: str) -> tuple[np.memmap, int, int, int]:
    ext = Path(path).suffix.lower()
    if ext not in (".up1", ".up2"):
        raise ValueError(f"Unsupported pattern file extension '{ext}'. Expected .up1 or .up2.")

    dtype = np.uint8 if ext == ".up1" else np.uint16
    file_size = os.path.getsize(path)
    if file_size < 16:
        raise ValueError(f"Pattern file is too small or empty: {path}")

    with open(path, "rb") as f:
        version_arr = np.fromfile(f, dtype=np.uint32, count=1)
        if version_arr.size != 1:
            raise ValueError(f"Could not read UP header version from {path}")
        version = int(version_arr[0])
        if version == 2:
            raise ValueError("UP file version 2 is not supported (expected version 1 or >= 3).")

        f.seek(4)
        hdr = np.fromfile(f, dtype=np.uint32, count=3)
        if hdr.size != 3:
            raise ValueError(f"Could not read UP pattern dimensions from {path}")
        sx, sy, pattern_offset = [int(v) for v in hdr]
        if sx <= 0 or sy <= 0:
            raise ValueError(f"Invalid UP pattern shape ({sy}, {sx}) in {path}")
        if pattern_offset <= 0 or pattern_offset >= file_size:
            raise ValueError(f"Invalid UP pattern offset {pattern_offset} in {path}")
        bytes_per_pattern = int(sx * sy * np.dtype(dtype).itemsize)
        n_patterns = int((file_size - pattern_offset) // bytes_per_pattern)
        if n_patterns <= 0:
            raise ValueError(f"No patterns found in {path}")

    data = np.memmap(path, dtype=dtype, mode="r", offset=pattern_offset, shape=(n_patterns, sy, sx))
    return data, int(n_patterns), int(sy), int(sx)
