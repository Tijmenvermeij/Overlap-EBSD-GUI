from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Literal

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution, minimize

from .legacy_projector import XProjector, load_master_hemis, ncc, normalize_zmuv

MAP_LAYER_CANDIDATES: dict[str, list[str]] = {
    "BC": ["1/EBSD/Data/Band Contrast"],
    "BS": ["1/EBSD/Data/Band Slope"],
    "IQ": ["1/EBSD/Data/Pattern Quality", "1/EBSD/Data/Image Quality"],
    "CI": ["1/EBSD/Data/Confidence Index", "1/EBSD/Data/CI"],
    "NCC": ["1/Data Processing/Pattern Matching/Data/Cross Correlation Coefficient"],
    "MAD": ["1/EBSD/Data/Mean Angular Deviation", "1/Data Processing/Data/Mean Angular Deviation"],
    "Phase": ["1/EBSD/Data/Phase", "1/Data Processing/Data/Phase"],
}
ORIENTATION_LAYER_LABEL = "IPF-Z (Euler)"


@dataclass
class GeometryConfig:
    pc_convention: str = "edax"
    sample_tilt_deg: float = 70.0
    detector_tilt_deg: float = 0.0
    azimuthal_deg: float = 0.0
    twist_deg: float = 0.0
    phi1_offset_deg: float = 0.0


@dataclass(frozen=True)
class PatternMaskConfig:
    kind: Literal["none", "circle"] = "circle"
    diameter_px: int = 0

    @property
    def option(self) -> int:
        if self.kind == "none":
            return -1
        return int(self.diameter_px)


@dataclass(frozen=True)
class DynamicBackgroundConfig:
    enabled: bool = False
    std_px: float = 0.0
    truncate: float = 4.0


@dataclass
class UPPatternReader:
    path: str
    dtype: np.dtype
    pattern_offset: int
    n_patterns: int
    h: int
    w: int

    @property
    def bytes_per_pattern(self) -> int:
        return int(self.h * self.w * np.dtype(self.dtype).itemsize)

    def read_pattern(self, index: int) -> np.ndarray:
        idx = int(index)
        if idx < 0 or idx >= int(self.n_patterns):
            raise IndexError(f"UP pattern index out of bounds: {idx} (N={self.n_patterns}).")
        n_px = int(self.h * self.w)
        offset = int(self.pattern_offset + idx * self.bytes_per_pattern)
        with open(self.path, "rb") as f:
            f.seek(offset)
            arr = np.fromfile(f, dtype=self.dtype, count=n_px)
        if arr.size != n_px:
            raise IOError(
                f"Failed to read full pattern #{idx} from {self.path}: expected {n_px} values, got {arr.size}."
            )
        return arr.reshape(self.h, self.w).astype(np.float32, copy=False)

    def read_patterns(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64).ravel()
        if idx.size == 0:
            raise ValueError("No pattern indices requested.")
        out = np.empty((idx.size, self.h, self.w), dtype=np.float32)
        for i, pidx in enumerate(idx.tolist()):
            out[i] = self.read_pattern(int(pidx))
        return out


@dataclass
class LoadedInputData:
    source_type: Literal["h5oina", "up_ang"]
    pattern_path: str
    orientation_path: str | None
    rows: int
    cols: int
    h: int
    w: int
    signal: object
    eulers_rad: np.ndarray
    phases: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    map_layers: dict[str, np.ndarray]
    beam_kv: float | None
    sample_tilt_deg: float
    detector_tilt_deg: float
    azimuthal_deg: float
    twist_deg: float
    rot_sd: np.ndarray
    direction_cosines: np.ndarray | None
    step_x: float
    step_y: float
    scan_unit: str
    pc_output_convention: str
    phase_symmetries: dict[int, object] = field(default_factory=dict)
    detector_px_size: float | None = None
    detector_binning: float | None = None
    ang_header_lines: list[str] | None = None
    ang_numeric: np.ndarray | None = None
    ang_angles_were_degrees: bool = False
    up_pattern_reader: UPPatternReader | None = None

    @property
    def count(self) -> int:
        return int(self.rows * self.cols)


@dataclass
class MasterPatternModel:
    kind: Literal["kikuchipy", "legacy"]
    path: str
    mp_signal: object | None
    projector: XProjector | None
    phase: object | None
    energy_kv: float | None = None
    phase_id: int = 1


@dataclass
class OverlapPointResult:
    index: int
    row: int
    col: int
    ncc_es: float
    scale: float
    ncc_residual_sim: float
    experimental: np.ndarray | None
    simulated: np.ndarray | None
    residual: np.ndarray | None
    simulated_unfitted: np.ndarray | None = None
    blurred_simulated: np.ndarray | None = None
    gain_map: np.ndarray | None = None
    fitted_sigma: float = 0.0
    gain_params: tuple[float, ...] = ()
    ellipse_params: tuple[float, ...] = ()
    ncc_unfitted: float | None = None
    fit_success: bool = True
    fit_message: str = ""
    secondary_dictionary_ncc_kp: float | None = None
    secondary_ncc_kp: float | None = None
    secondary_ncc_full: float | None = None
    secondary_euler_rad: np.ndarray | None = None
    secondary_simulated: np.ndarray | None = None
    secondary_refined: bool = False
    secondary_refinement_note: str = ""


@dataclass
class OverlapMixtureResult:
    index: int
    row: int
    col: int
    primary_fraction: float
    secondary_fraction: float
    primary_coefficient: float
    secondary_coefficient: float
    ncc_mixture: float
    residual_rms: float
    old_primary_ncc: float | None
    old_secondary_ncc: float | None
    experimental: np.ndarray | None
    primary_simulated: np.ndarray | None
    secondary_simulated: np.ndarray | None
    combined_simulated: np.ndarray | None
    residual: np.ndarray | None
    gain_map: np.ndarray | None = None
    fitted_sigma: float = 0.0
    gain_params: tuple[float, ...] = ()
    ellipse_params: tuple[float, ...] = ()
    component_correlation: float = 0.0
    primary_euler_rad: np.ndarray | None = None
    secondary_euler_rad: np.ndarray | None = None
    fit_success: bool = True
    fit_message: str = ""
    orientation_refined: bool = False
    orientation_refinement_note: str = ""
    initial_mixture_ncc: float | None = None
    primary_euler_delta_deg: tuple[float, ...] = ()
    secondary_euler_delta_deg: tuple[float, ...] = ()


@dataclass
class DictionaryCache:
    phase_id: int
    resolution_deg: float
    pc_bruker: np.ndarray
    software_binning: int
    crop_extent: tuple[int, int, int, int]
    pattern_shape: tuple[int, int]
    signal: object
    rotation_count: int


@dataclass
class ResidualBatchPayload:
    indices: np.ndarray
    experimental: np.ndarray
    eulers_rad: np.ndarray
    pc_bruker: np.ndarray
    pc_custom: np.ndarray


@dataclass
class OverlapMixtureBatchPayload:
    indices: np.ndarray
    experimental: np.ndarray
    primary_eulers_rad: np.ndarray
    secondary_eulers_rad: np.ndarray
    pc_bruker: np.ndarray
    pc_custom: np.ndarray
    old_primary_ncc: np.ndarray
    old_secondary_ncc: np.ndarray


@dataclass
class PrimaryPatternFit:
    sigma: float
    gain_params: tuple[float, float, float]
    ellipse_params: tuple[float, float, float, float]
    experimental: np.ndarray
    simulated_unfitted: np.ndarray
    blurred_simulated: np.ndarray
    gain_map: np.ndarray
    processed_simulated: np.ndarray
    ncc_unfitted: float
    ncc_fitted: float
    scale: float
    residual: np.ndarray
    ncc_residual: float
    success: bool
    message: str


@dataclass
class OverlapMixtureFit:
    sigma: float
    gain_params: tuple[float, float, float]
    ellipse_params: tuple[float, float, float, float]
    primary_coefficient: float
    secondary_coefficient: float
    primary_fraction: float
    secondary_fraction: float
    experimental: np.ndarray
    primary_processed: np.ndarray
    secondary_processed: np.ndarray
    combined_simulated: np.ndarray
    residual: np.ndarray
    gain_map: np.ndarray
    ncc_mixture: float
    residual_rms: float
    component_correlation: float
    success: bool
    message: str


def _pattern_mask_config_from_option(option: int) -> PatternMaskConfig:
    value = int(option)
    if value == -1:
        return PatternMaskConfig(kind="none", diameter_px=-1)
    if value < -1:
        raise ValueError("Pattern mask option must be -1 (none), 0 (largest circle), or a positive diameter.")
    return PatternMaskConfig(kind="circle", diameter_px=value)


def _describe_pattern_mask(config: PatternMaskConfig, shape: tuple[int, int] | None = None) -> str:
    if config.kind == "none":
        return "no pattern mask"
    if config.diameter_px == 0:
        if shape is None:
            return "circular mask, diameter=min(pattern height, width)"
        return f"circular mask, diameter={min(map(int, shape))} px (largest fitting circle)"
    return f"circular mask, diameter={int(config.diameter_px)} px"


def _valid_circular_pixels(shape: tuple[int, int], diameter_px: float) -> np.ndarray:
    # Kikuchipy consumes boolean signal masks, but Window("circular") is not
    # the requested min-dimension circle for rectangular patterns.
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid pattern shape {shape}.")
    diameter = float(diameter_px)
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Circular pattern mask diameter must be positive.")
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    radius = max(0.0, (diameter - 1.0) / 2.0)
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
    valid = dist2 <= (radius * radius + 1e-9)
    if not np.any(valid):
        valid = dist2 <= (float(np.min(dist2)) + 1e-9)
    return valid


def _signal_mask_from_pattern_mask(
    shape: tuple[int, int],
    config: PatternMaskConfig,
    *,
    software_binning: int = 1,
) -> np.ndarray | None:
    if config.kind == "none":
        return None
    h, w = map(int, shape)
    if config.diameter_px == 0:
        diameter = float(min(h, w))
    else:
        factor = max(1, int(software_binning))
        diameter = float(config.diameter_px) / factor
    valid = _valid_circular_pixels((h, w), diameter)
    return ~valid


def _weights_from_pattern_mask(
    shape: tuple[int, int],
    config: PatternMaskConfig,
    *,
    software_binning: int = 1,
) -> np.ndarray:
    signal_mask = _signal_mask_from_pattern_mask(
        shape,
        config,
        software_binning=software_binning,
    )
    if signal_mask is None:
        weights = np.ones(tuple(map(int, shape)), dtype=np.float32)
    else:
        weights = (~signal_mask).astype(np.float32)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"Pattern mask excludes all pixels for shape {shape}.")
    return weights / total


def _dynamic_bg_config(enabled: bool, std_px: float = 0.0, truncate: float = 4.0) -> DynamicBackgroundConfig:
    enabled = bool(enabled)
    std = float(std_px)
    if enabled and (not np.isfinite(std) or std < 0.0):
        raise ValueError("Dynamic background std must be 0 (auto) or a positive pixel value.")
    trunc = float(truncate)
    if not np.isfinite(trunc) or trunc <= 0.0:
        raise ValueError("Dynamic background Gaussian truncate must be positive.")
    if not enabled:
        std = 0.0
    return DynamicBackgroundConfig(enabled=enabled, std_px=std, truncate=trunc)


def _dynamic_bg_sigma_for_shape(shape: tuple[int, int], config: DynamicBackgroundConfig) -> float:
    if config.std_px > 0.0:
        return float(config.std_px)
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid pattern shape {shape}.")
    return max(1e-6, float(w) / 8.0)


def _describe_dynamic_background(config: DynamicBackgroundConfig, shape: tuple[int, int] | None = None) -> str:
    if not config.enabled:
        return "off"
    if config.std_px <= 0.0:
        if shape is None:
            return "subtract, std=auto (pattern width/8)"
        return f"subtract, std={_dynamic_bg_sigma_for_shape(shape, config):g} px (auto)"
    return f"subtract, std={float(config.std_px):g} px"


def _dynamic_bg_std_arg(config: DynamicBackgroundConfig) -> float | None:
    return None if float(config.std_px) <= 0.0 else float(config.std_px)


def _subtract_dynamic_background_pattern(
    pattern: np.ndarray,
    config: DynamicBackgroundConfig,
    *,
    signal_mask: np.ndarray | None = None,
) -> np.ndarray:
    return _subtract_dynamic_background_patterns(pattern, config, signal_mask=signal_mask)


def _subtract_dynamic_background_patterns(
    patterns: np.ndarray,
    config: DynamicBackgroundConfig,
    *,
    signal_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(patterns, dtype=np.float32)
    if arr.ndim == 2:
        work = arr[np.newaxis, ...]
        squeeze = True
    elif arr.ndim == 3:
        work = arr
        squeeze = False
    else:
        raise ValueError(f"Dynamic background subtraction expects 2D/3D patterns, got shape {arr.shape}.")
    sigma = _dynamic_bg_sigma_for_shape(work.shape[-2:], config)
    truncate = float(config.truncate)
    spatial_sigma = (0.0, sigma, sigma)

    if signal_mask is None:
        background = gaussian_filter(work, sigma=spatial_sigma, truncate=truncate)
        corrected = (work - background).astype(np.float32, copy=False)
        return corrected[0] if squeeze else corrected

    mask = np.asarray(signal_mask, dtype=bool)
    if mask.shape != work.shape[-2:]:
        raise ValueError(f"Signal mask shape {mask.shape} does not match pattern shape {work.shape[-2:]}.")
    valid = ~mask
    if not np.any(valid):
        raise ValueError(f"Pattern mask excludes all pixels for shape {work.shape[-2:]}.")
    if np.all(valid):
        background = gaussian_filter(work, sigma=spatial_sigma, truncate=truncate)
        corrected = (work - background).astype(np.float32, copy=False)
        return corrected[0] if squeeze else corrected

    valid_f = valid.astype(np.float32)
    values = np.where(valid[np.newaxis, :, :], work, 0.0).astype(np.float32, copy=False)
    numerator = gaussian_filter(values, sigma=spatial_sigma, truncate=truncate)
    denominator = gaussian_filter(valid_f, sigma=sigma, truncate=truncate)
    fallback = np.mean(work[:, valid], axis=1, dtype=np.float64).astype(np.float32).reshape(-1, 1, 1)
    background = np.empty_like(numerator, dtype=np.float32)
    background[...] = fallback
    background = np.divide(
        numerator,
        denominator,
        out=background,
        where=denominator > 1e-6,
    )
    corrected = (work - background).astype(np.float32, copy=False)
    corrected[:, mask] = 0.0
    return corrected[0] if squeeze else corrected


def _zero_unweighted_pixels(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(weights, dtype=np.float32) > 0.0
    if valid.shape != arr.shape or np.all(valid):
        return arr
    out = arr.copy()
    out[~valid] = 0.0
    return out


def _normalize_weighted(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    total = float(w.sum())
    if total <= 1e-12:
        return normalize_zmuv(arr)
    wn = w / total
    mean = float(np.sum(wn * arr))
    centered = arr - mean
    variance = float(np.sum(wn * centered * centered))
    if variance <= 1e-12:
        return centered
    return centered / np.sqrt(variance)


def _weighted_ncc(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    aa = _normalize_weighted(a, weights)
    bb = _normalize_weighted(b, weights)
    w = np.asarray(weights, dtype=np.float32)
    total = float(w.sum())
    if total <= 1e-12:
        return ncc(aa, bb)
    return float(np.sum((w / total) * aa * bb))


def _power_gain_map(
    shape: tuple[int, int],
    gain_params: tuple[float, float, float] | list[float],
    ellipse_params: tuple[float, float, float, float] | list[float],
) -> np.ndarray:
    h, w = map(int, shape)
    g_min, g_max, power = (float(v) for v in gain_params)
    a_scale, b_scale, center_y_offset, center_x_offset = (float(v) for v in ellipse_params)
    a = max(0.1, a_scale) * h / 2.0
    b = max(0.1, b_scale) * w / 2.0
    center_y = h / 2.0 + center_y_offset * h
    center_x = w / 2.0 + center_x_offset * w
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(((yy - center_y) / a) ** 2 + ((xx - center_x) / b) ** 2)
    radius = np.clip(radius, 0.0, 1.0)
    return (g_min + (1.0 - radius) ** power * (g_max - g_min)).astype(np.float32)


def _normalize_overlap_pattern(pattern: np.ndarray, circular_mask: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(pattern, dtype=np.float32)
    if circular_mask is None:
        return normalize_zmuv(arr)
    mask = np.asarray(circular_mask, dtype=bool)
    values = arr[mask]
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std


def _fit_overlap_primary_pattern(
    experimental_raw: np.ndarray,
    simulated_raw: np.ndarray,
    weights: np.ndarray,
    *,
    maxiter: int,
    popsize: int,
    seed: int,
    fit_bounds: list[tuple[float, float]] | None = None,
) -> PrimaryPatternFit:
    experimental = _normalize_weighted(experimental_raw, weights)
    simulated_unfitted = _normalize_weighted(simulated_raw, weights)
    ncc_unfitted = _weighted_ncc(experimental, simulated_unfitted, weights)

    def evaluate(params: np.ndarray):
        sigma = float(params[0])
        gain_params = tuple(float(v) for v in params[1:4])
        ellipse_params = tuple(float(v) for v in params[4:8])
        blurred = _normalize_weighted(gaussian_filter(simulated_raw, sigma=sigma), weights)
        gain_map = _power_gain_map(experimental.shape, gain_params, ellipse_params)
        processed = _normalize_weighted(blurred * gain_map, weights)
        fitted_ncc = _weighted_ncc(experimental, processed, weights)
        residual = experimental - fitted_ncc * processed
        ssr = float(np.sum(weights * residual * residual))
        return ssr, blurred, gain_map, processed, fitted_ncc, residual

    default_bounds = [
        (0.1, 5.0),   # Gaussian sigma
        (-1.5, 4.5),  # g_min
        (0.0, 12.5),  # g_max
        (0.1, 10.0),  # power p
        (0.6, 1.4),   # ellipse a scale
        (0.6, 1.4),   # ellipse b scale
        (-0.15, 0.15),
        (-0.15, 0.15),
    ]
    bounds = default_bounds if fit_bounds is None else [tuple(map(float, pair)) for pair in fit_bounds]
    if len(bounds) != 8:
        raise ValueError("Primary fit bounds must contain eight (low, high) pairs.")
    for low, high in bounds:
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Primary fit bounds must be finite and strictly increasing.")

    def objective(params: np.ndarray) -> float:
        return evaluate(params)[0]

    optimization = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max(1, int(maxiter)),
        popsize=max(4, int(popsize)),
        polish=True,
        seed=int(seed),
        disp=False,
        updating="deferred",
        workers=1,
    )
    params = np.asarray(optimization.x, dtype=np.float64)
    _ssr, blurred, gain_map, processed, fitted_ncc, residual = evaluate(params)
    residual_ncc = _weighted_ncc(residual, processed, weights)
    return PrimaryPatternFit(
        sigma=float(params[0]),
        gain_params=tuple(float(v) for v in params[1:4]),
        ellipse_params=tuple(float(v) for v in params[4:8]),
        experimental=experimental,
        simulated_unfitted=simulated_unfitted,
        blurred_simulated=blurred,
        gain_map=gain_map,
        processed_simulated=processed,
        ncc_unfitted=ncc_unfitted,
        ncc_fitted=float(fitted_ncc),
        scale=float(fitted_ncc),
        residual=np.asarray(residual, dtype=np.float32),
        ncc_residual=float(residual_ncc),
        success=bool(optimization.success),
        message=str(optimization.message),
    )


def _fit_nonnegative_two_component_coefficients(
    experimental: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    y = np.asarray(experimental, dtype=np.float32)
    s1 = np.asarray(primary, dtype=np.float32)
    s2 = np.asarray(secondary, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    g11 = float(np.sum(w * s1 * s1))
    g22 = float(np.sum(w * s2 * s2))
    g12 = float(np.sum(w * s1 * s2))
    b1 = float(np.sum(w * s1 * y))
    b2 = float(np.sum(w * s2 * y))

    candidates: list[tuple[float, float]] = [(0.0, 0.0)]
    if g11 > 1e-12:
        candidates.append((max(0.0, b1 / g11), 0.0))
    if g22 > 1e-12:
        candidates.append((0.0, max(0.0, b2 / g22)))

    det = g11 * g22 - g12 * g12
    if det > 1e-12:
        a1 = (b1 * g22 - b2 * g12) / det
        a2 = (b2 * g11 - b1 * g12) / det
        if a1 >= 0.0 and a2 >= 0.0:
            candidates.append((float(a1), float(a2)))

    best = (0.0, 0.0)
    best_ssr = float("inf")
    for a1, a2 in candidates:
        residual = y - float(a1) * s1 - float(a2) * s2
        ssr = float(np.sum(w * residual * residual))
        if ssr < best_ssr:
            best_ssr = ssr
            best = (float(a1), float(a2))
    return best


def _fractions_from_coefficients(a1: float, a2: float) -> tuple[float, float]:
    total = float(a1) + float(a2)
    if not np.isfinite(total) or total <= 1e-12:
        return float("nan"), float("nan")
    return float(a1) / total, float(a2) / total


def _evaluate_overlap_mixture_pattern(
    experimental_raw: np.ndarray,
    primary_raw: np.ndarray,
    secondary_raw: np.ndarray,
    weights: np.ndarray,
    params: np.ndarray,
) -> OverlapMixtureFit:
    experimental = _normalize_weighted(experimental_raw, weights)
    sigma = float(params[0])
    gain_params = tuple(float(v) for v in params[1:4])
    ellipse_params = tuple(float(v) for v in params[4:8])
    gain_map = _power_gain_map(experimental.shape, gain_params, ellipse_params)

    primary_blurred = gaussian_filter(primary_raw, sigma=sigma)
    secondary_blurred = gaussian_filter(secondary_raw, sigma=sigma)
    primary_processed = _normalize_weighted(primary_blurred * gain_map, weights)
    secondary_processed = _normalize_weighted(secondary_blurred * gain_map, weights)
    a1, a2 = _fit_nonnegative_two_component_coefficients(
        experimental,
        primary_processed,
        secondary_processed,
        weights,
    )
    combined = float(a1) * primary_processed + float(a2) * secondary_processed
    residual = experimental - combined
    primary_fraction, secondary_fraction = _fractions_from_coefficients(a1, a2)
    ncc_mixture = _weighted_ncc(experimental, combined, weights) if (a1 + a2) > 1e-12 else float("nan")
    residual_rms = float(np.sqrt(np.sum(np.asarray(weights, dtype=np.float32) * residual * residual)))
    component_correlation = _weighted_ncc(primary_processed, secondary_processed, weights)
    return OverlapMixtureFit(
        sigma=float(sigma),
        gain_params=tuple(float(v) for v in gain_params),
        ellipse_params=tuple(float(v) for v in ellipse_params),
        primary_coefficient=float(a1),
        secondary_coefficient=float(a2),
        primary_fraction=float(primary_fraction),
        secondary_fraction=float(secondary_fraction),
        experimental=np.asarray(experimental, dtype=np.float32),
        primary_processed=np.asarray(primary_processed, dtype=np.float32),
        secondary_processed=np.asarray(secondary_processed, dtype=np.float32),
        combined_simulated=np.asarray(combined, dtype=np.float32),
        residual=np.asarray(residual, dtype=np.float32),
        gain_map=np.asarray(gain_map, dtype=np.float32),
        ncc_mixture=float(ncc_mixture),
        residual_rms=float(residual_rms),
        component_correlation=float(component_correlation),
        success=True,
        message="",
    )


def _fit_overlap_mixture_pattern(
    experimental_raw: np.ndarray,
    primary_raw: np.ndarray,
    secondary_raw: np.ndarray,
    weights: np.ndarray,
    *,
    maxiter: int,
    popsize: int,
    seed: int,
    fit_bounds: list[tuple[float, float]] | None = None,
) -> OverlapMixtureFit:
    default_bounds = [
        (0.1, 5.0),
        (-1.5, 4.5),
        (0.0, 12.5),
        (0.1, 10.0),
        (0.6, 1.4),
        (0.6, 1.4),
        (-0.15, 0.15),
        (-0.15, 0.15),
    ]
    bounds = default_bounds if fit_bounds is None else [tuple(map(float, pair)) for pair in fit_bounds]
    if len(bounds) != 8:
        raise ValueError("Mixture fit bounds must contain eight (low, high) pairs.")
    for low, high in bounds:
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Mixture fit bounds must be finite and strictly increasing.")

    def objective(params: np.ndarray) -> float:
        fit = _evaluate_overlap_mixture_pattern(
            experimental_raw,
            primary_raw,
            secondary_raw,
            weights,
            np.asarray(params, dtype=np.float64),
        )
        return float(np.sum(np.asarray(weights, dtype=np.float32) * fit.residual * fit.residual))

    optimization = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max(1, int(maxiter)),
        popsize=max(4, int(popsize)),
        polish=True,
        seed=int(seed),
        disp=False,
        updating="deferred",
        workers=1,
    )
    fit = _evaluate_overlap_mixture_pattern(
        experimental_raw,
        primary_raw,
        secondary_raw,
        weights,
        np.asarray(optimization.x, dtype=np.float64),
    )
    fit.success = bool(optimization.success)
    fit.message = str(optimization.message)
    return fit


def _overlap_mixture_result_from_fit(
    index: int,
    row: int,
    col: int,
    fit: OverlapMixtureFit,
    weights: np.ndarray,
    *,
    primary_euler_rad: np.ndarray,
    secondary_euler_rad: np.ndarray,
    old_primary_ncc: float | None,
    old_secondary_ncc: float | None,
    orientation_refined: bool = False,
    orientation_refinement_note: str = "",
    initial_mixture_ncc: float | None = None,
    primary_euler_delta_deg: tuple[float, ...] = (),
    secondary_euler_delta_deg: tuple[float, ...] = (),
) -> OverlapMixtureResult:
    return OverlapMixtureResult(
        index=int(index),
        row=int(row),
        col=int(col),
        primary_fraction=float(fit.primary_fraction),
        secondary_fraction=float(fit.secondary_fraction),
        primary_coefficient=float(fit.primary_coefficient),
        secondary_coefficient=float(fit.secondary_coefficient),
        ncc_mixture=float(fit.ncc_mixture),
        residual_rms=float(fit.residual_rms),
        old_primary_ncc=None if old_primary_ncc is None else float(old_primary_ncc),
        old_secondary_ncc=None if old_secondary_ncc is None else float(old_secondary_ncc),
        experimental=_zero_unweighted_pixels(fit.experimental, weights),
        primary_simulated=_zero_unweighted_pixels(fit.primary_processed, weights),
        secondary_simulated=_zero_unweighted_pixels(fit.secondary_processed, weights),
        combined_simulated=_zero_unweighted_pixels(fit.combined_simulated, weights),
        residual=_zero_unweighted_pixels(fit.residual, weights),
        gain_map=_zero_unweighted_pixels(fit.gain_map, weights),
        fitted_sigma=fit.sigma,
        gain_params=fit.gain_params,
        ellipse_params=fit.ellipse_params,
        component_correlation=float(fit.component_correlation),
        primary_euler_rad=np.asarray(primary_euler_rad, dtype=np.float64).reshape(3).copy(),
        secondary_euler_rad=np.asarray(secondary_euler_rad, dtype=np.float64).reshape(3).copy(),
        fit_success=fit.success,
        fit_message=fit.message,
        orientation_refined=bool(orientation_refined),
        orientation_refinement_note=str(orientation_refinement_note),
        initial_mixture_ncc=None if initial_mixture_ncc is None else float(initial_mixture_ncc),
        primary_euler_delta_deg=tuple(float(v) for v in primary_euler_delta_deg),
        secondary_euler_delta_deg=tuple(float(v) for v in secondary_euler_delta_deg),
    )


def _overlap_mixture_result_from_raw_patterns(
    index: int,
    row: int,
    col: int,
    experimental_raw: np.ndarray,
    primary_raw: np.ndarray,
    secondary_raw: np.ndarray,
    weights: np.ndarray,
    *,
    primary_euler_rad: np.ndarray,
    secondary_euler_rad: np.ndarray,
    old_primary_ncc: float | None,
    old_secondary_ncc: float | None,
    fit_maxiter: int,
    fit_popsize: int,
    fit_bounds: list[tuple[float, float]] | None,
) -> OverlapMixtureResult:
    fit = _fit_overlap_mixture_pattern(
        experimental_raw,
        primary_raw,
        secondary_raw,
        weights,
        maxiter=int(fit_maxiter),
        popsize=int(fit_popsize),
        seed=int(index) + 7919,
        fit_bounds=fit_bounds,
    )
    return _overlap_mixture_result_from_fit(
        index,
        row,
        col,
        fit,
        weights,
        primary_euler_rad=primary_euler_rad,
        secondary_euler_rad=secondary_euler_rad,
        old_primary_ncc=old_primary_ncc,
        old_secondary_ncc=old_secondary_ncc,
    )


def _overlap_point_result_from_raw_patterns(
    index: int,
    row: int,
    col: int,
    experimental_raw: np.ndarray,
    simulated_raw: np.ndarray,
    weights: np.ndarray,
    *,
    fit_blur_gain: bool,
    fit_maxiter: int,
    fit_popsize: int,
    fit_bounds: list[tuple[float, float]] | None,
    blur_sigma: float = 0.0,
) -> OverlapPointResult:
    if fit_blur_gain:
        fit = _fit_overlap_primary_pattern(
            experimental_raw,
            simulated_raw,
            weights,
            maxiter=int(fit_maxiter),
            popsize=int(fit_popsize),
            seed=int(index),
            fit_bounds=fit_bounds,
        )
    else:
        experimental = _normalize_weighted(experimental_raw, weights)
        simulated_unfitted = _normalize_weighted(simulated_raw, weights)
        blurred = _normalize_weighted(
            gaussian_filter(simulated_raw, sigma=float(blur_sigma)) if blur_sigma > 0 else simulated_raw,
            weights,
        )
        gain_map = np.ones_like(blurred, dtype=np.float32)
        processed = _normalize_weighted(blurred, weights)
        fitted_ncc = _weighted_ncc(experimental, processed, weights)
        residual = experimental - fitted_ncc * processed
        fit = PrimaryPatternFit(
            sigma=max(0.0, float(blur_sigma)),
            gain_params=(1.0, 1.0, 1.0),
            ellipse_params=(1.0, 1.0, 0.0, 0.0),
            experimental=experimental,
            simulated_unfitted=simulated_unfitted,
            blurred_simulated=blurred,
            gain_map=gain_map,
            processed_simulated=processed,
            ncc_unfitted=_weighted_ncc(experimental, simulated_unfitted, weights),
            ncc_fitted=fitted_ncc,
            scale=fitted_ncc,
            residual=np.asarray(residual, dtype=np.float32),
            ncc_residual=_weighted_ncc(residual, processed, weights),
            success=True,
            message="Manual blur; gain fitting disabled.",
        )

    return OverlapPointResult(
        index=int(index),
        row=int(row),
        col=int(col),
        ncc_es=fit.ncc_fitted,
        scale=fit.scale,
        ncc_residual_sim=fit.ncc_residual,
        experimental=_zero_unweighted_pixels(fit.experimental, weights),
        simulated=_zero_unweighted_pixels(fit.processed_simulated, weights),
        residual=_zero_unweighted_pixels(fit.residual, weights),
        simulated_unfitted=_zero_unweighted_pixels(fit.simulated_unfitted, weights),
        blurred_simulated=_zero_unweighted_pixels(fit.blurred_simulated, weights),
        gain_map=_zero_unweighted_pixels(fit.gain_map, weights),
        fitted_sigma=fit.sigma,
        gain_params=fit.gain_params,
        ellipse_params=fit.ellipse_params,
        ncc_unfitted=fit.ncc_unfitted,
        fit_success=fit.success,
        fit_message=fit.message,
    )


def _residual_to_uint8(residual_z: np.ndarray, zlim: float = 3.0, *, hist_norm: bool = False) -> np.ndarray:
    arr = np.asarray(residual_z, dtype=np.float32)
    zlim = max(float(zlim), 0.1)
    clipped = np.clip(arr, -zlim, zlim)
    scaled = np.round((clipped + zlim) * (255.0 / (2.0 * zlim))).astype(np.uint8)
    if not hist_norm:
        return scaled
    hist = np.bincount(scaled.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = np.flatnonzero(hist)
    if nonzero.size <= 1:
        return scaled
    cdf_min = int(cdf[nonzero[0]])
    cdf_max = int(cdf[nonzero[-1]])
    if cdf_max <= cdf_min:
        return scaled
    lut = np.round((cdf - cdf_min) * 255.0 / (cdf_max - cdf_min)).astype(np.int32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[scaled]


def _write_up_pattern_at(
    out_path: str,
    reader: UPPatternReader,
    index: int,
    pattern_u8: np.ndarray,
    *,
    file_obj=None,
) -> None:
    idx = int(index)
    arr_u8 = np.asarray(pattern_u8, dtype=np.uint8)
    if arr_u8.shape != (reader.h, reader.w):
        raise ValueError(f"UP write shape mismatch for idx={idx}: got {arr_u8.shape}, expected {(reader.h, reader.w)}.")
    if reader.dtype == np.dtype(np.uint16):
        arr_write = (arr_u8.astype(np.uint16) * 257).astype(np.uint16, copy=False)
    else:
        arr_write = arr_u8.astype(np.uint8, copy=False)

    if file_obj is None:
        with open(out_path, "r+b") as f:
            f.seek(reader.pattern_offset + idx * reader.bytes_per_pattern)
            arr_write.tofile(f)
    else:
        file_obj.seek(reader.pattern_offset + idx * reader.bytes_per_pattern)
        arr_write.tofile(file_obj)


_RESIDUAL_ROI_WORKER_STATE: dict[str, object] | None = None


def _init_residual_roi_worker(
    master_kind: str,
    master_path: str,
    master_energy_kv: float | None,
    h: int,
    w: int,
    cols: int,
    rot_sd: np.ndarray,
    direction_cosines: np.ndarray | None,
    sample_tilt_deg: float,
    detector_tilt_deg: float,
    azimuthal_deg: float,
    twist_deg: float,
    kikuchipy_frame_active: bool,
    weights: np.ndarray,
    fit_blur_gain: bool,
    fit_maxiter: int,
    fit_popsize: int,
    fit_bounds: list[tuple[float, float]] | None,
) -> None:
    import kikuchipy as kp

    global _RESIDUAL_ROI_WORKER_STATE

    state: dict[str, object] = {
        "master_kind": str(master_kind),
        "master_energy_kv": None if master_energy_kv is None else float(master_energy_kv),
        "h": int(h),
        "w": int(w),
        "cols": int(cols),
        "rot_sd": np.asarray(rot_sd, dtype=np.float64),
        "direction_cosines": None if direction_cosines is None else np.asarray(direction_cosines, dtype=np.float64),
        "sample_tilt_deg": float(sample_tilt_deg),
        "detector_tilt_deg": float(detector_tilt_deg),
        "azimuthal_deg": float(azimuthal_deg),
        "twist_deg": float(twist_deg),
        "kikuchipy_frame_active": bool(kikuchipy_frame_active),
        "weights": np.asarray(weights, dtype=np.float32),
        "fit_blur_gain": bool(fit_blur_gain),
        "fit_maxiter": int(fit_maxiter),
        "fit_popsize": int(fit_popsize),
        "fit_bounds": None
        if fit_bounds is None
        else [tuple(map(float, pair)) for pair in fit_bounds],
        "mp_signal": None,
        "projector": None,
    }

    if str(master_kind) == "kikuchipy":
        state["mp_signal"] = kp.load(str(master_path), projection="lambert", hemisphere="both", lazy=True)
    else:
        hemis = load_master_hemis([str(master_path)])
        state["projector"] = XProjector(hemis[0], int(h), int(w))

    _RESIDUAL_ROI_WORKER_STATE = state


def _residual_roi_worker_simulated_pattern(euler_rad: np.ndarray, pc: np.ndarray) -> np.ndarray:
    from orix.quaternion import Rotation

    if _RESIDUAL_ROI_WORKER_STATE is None:
        raise RuntimeError("Residual ROI worker is not initialized.")
    state = _RESIDUAL_ROI_WORKER_STATE
    h = int(state["h"])
    w = int(state["w"])
    master_kind = str(state["master_kind"])

    if master_kind == "kikuchipy":
        import kikuchipy as kp

        mp_signal = state["mp_signal"]
        if mp_signal is None:
            raise RuntimeError("Kikuchipy master pattern was not loaded in the residual ROI worker.")
        rot = np.asarray(euler_rad, dtype=np.float64).reshape(1, 3)
        if bool(state["kikuchipy_frame_active"]):
            rot = _left_multiply_eulers_zxz(rot, angle_rad=np.deg2rad(90.0))
        rotation = Rotation.from_euler(rot, degrees=False)
        detector = kp.detectors.EBSDDetector(
            shape=(h, w),
            pc=np.asarray(pc, dtype=np.float64).reshape(1, 3),
            convention="bruker",
            sample_tilt=float(state["sample_tilt_deg"]),
            tilt=float(state["detector_tilt_deg"]),
            azimuthal=float(state["azimuthal_deg"]),
            twist=float(state["twist_deg"]),
        )
        signal = mp_signal.get_patterns(
            rotations=rotation,
            detector=detector,
            energy=float(state["master_energy_kv"]) if state["master_energy_kv"] is not None else 20.0,
            compute=True,
            show_progressbar=False,
        )
        return np.asarray(signal.data[0], dtype=np.float32)

    projector = state["projector"]
    if projector is None:
        raise RuntimeError("Legacy master projector was not loaded in the residual ROI worker.")
    return projector.project(
        np.asarray(euler_rad, dtype=np.float64).reshape(3),
        tuple(float(v) for v in np.asarray(pc, dtype=np.float64).reshape(3)),
        np.asarray(state["rot_sd"], dtype=np.float64),
        direction_cosines=state["direction_cosines"],
    )


def _compute_residual_roi_batch(payload: ResidualBatchPayload) -> list[OverlapPointResult]:
    if _RESIDUAL_ROI_WORKER_STATE is None:
        raise RuntimeError("Residual ROI worker is not initialized.")

    state = _RESIDUAL_ROI_WORKER_STATE
    weights = np.asarray(state["weights"], dtype=np.float32)
    fit_blur_gain = bool(state["fit_blur_gain"])
    fit_maxiter = int(state["fit_maxiter"])
    fit_popsize = int(state["fit_popsize"])
    fit_bounds = state["fit_bounds"]
    cols = int(state["cols"])

    indices = np.asarray(payload.indices, dtype=np.int64).ravel()
    experimental = np.asarray(payload.experimental, dtype=np.float32)
    eulers = np.asarray(payload.eulers_rad, dtype=np.float64)
    pc_bruker = np.asarray(payload.pc_bruker, dtype=np.float64)
    pc_custom = np.asarray(payload.pc_custom, dtype=np.float64)

    if experimental.ndim == 2:
        experimental = experimental[np.newaxis, ...]
    if eulers.ndim == 1:
        eulers = eulers.reshape(1, 3)
    if pc_bruker.ndim == 1:
        pc_bruker = pc_bruker.reshape(1, 3)
    if pc_custom.ndim == 1:
        pc_custom = pc_custom.reshape(1, 3)

    results: list[OverlapPointResult] = []
    master_kind = str(state["master_kind"])
    for i, idx in enumerate(indices.tolist()):
        row, col = divmod(int(idx), cols)
        pc_use = pc_bruker[i] if master_kind == "kikuchipy" else pc_custom[i]
        sim_raw = _residual_roi_worker_simulated_pattern(eulers[i], pc_use)
        result = _overlap_point_result_from_raw_patterns(
            int(idx),
            row,
            col,
            experimental[i],
            sim_raw,
            weights,
            fit_blur_gain=fit_blur_gain,
            fit_maxiter=fit_maxiter,
            fit_popsize=fit_popsize,
            fit_bounds=fit_bounds,
        )
        results.append(result)

    return results


def _compute_overlap_mixture_roi_batch(payload: OverlapMixtureBatchPayload) -> list[OverlapMixtureResult]:
    if _RESIDUAL_ROI_WORKER_STATE is None:
        raise RuntimeError("Overlap mixture ROI worker is not initialized.")

    state = _RESIDUAL_ROI_WORKER_STATE
    weights = np.asarray(state["weights"], dtype=np.float32)
    fit_maxiter = int(state["fit_maxiter"])
    fit_popsize = int(state["fit_popsize"])
    fit_bounds = state["fit_bounds"]
    cols = int(state["cols"])

    indices = np.asarray(payload.indices, dtype=np.int64).ravel()
    experimental = np.asarray(payload.experimental, dtype=np.float32)
    primary_eulers = np.asarray(payload.primary_eulers_rad, dtype=np.float64)
    secondary_eulers = np.asarray(payload.secondary_eulers_rad, dtype=np.float64)
    pc_bruker = np.asarray(payload.pc_bruker, dtype=np.float64)
    pc_custom = np.asarray(payload.pc_custom, dtype=np.float64)
    old_primary_ncc = np.asarray(payload.old_primary_ncc, dtype=np.float64).reshape(-1)
    old_secondary_ncc = np.asarray(payload.old_secondary_ncc, dtype=np.float64).reshape(-1)

    if experimental.ndim == 2:
        experimental = experimental[np.newaxis, ...]
    if primary_eulers.ndim == 1:
        primary_eulers = primary_eulers.reshape(1, 3)
    if secondary_eulers.ndim == 1:
        secondary_eulers = secondary_eulers.reshape(1, 3)
    if pc_bruker.ndim == 1:
        pc_bruker = pc_bruker.reshape(1, 3)
    if pc_custom.ndim == 1:
        pc_custom = pc_custom.reshape(1, 3)

    results: list[OverlapMixtureResult] = []
    master_kind = str(state["master_kind"])
    for i, idx in enumerate(indices.tolist()):
        row, col = divmod(int(idx), cols)
        pc_use = pc_bruker[i] if master_kind == "kikuchipy" else pc_custom[i]
        primary_raw = _residual_roi_worker_simulated_pattern(primary_eulers[i], pc_use)
        secondary_raw = _residual_roi_worker_simulated_pattern(secondary_eulers[i], pc_use)
        result = _overlap_mixture_result_from_raw_patterns(
            int(idx),
            row,
            col,
            experimental[i],
            primary_raw,
            secondary_raw,
            weights,
            primary_euler_rad=primary_eulers[i],
            secondary_euler_rad=secondary_eulers[i],
            old_primary_ncc=None if not np.isfinite(old_primary_ncc[i]) else float(old_primary_ncc[i]),
            old_secondary_ncc=None if not np.isfinite(old_secondary_ncc[i]) else float(old_secondary_ncc[i]),
            fit_maxiter=fit_maxiter,
            fit_popsize=fit_popsize,
            fit_bounds=fit_bounds,
        )
        results.append(result)

    return results


@dataclass
class ResidualPatternWriter:
    output_path: Path
    source_type: Literal["h5oina", "up_ang"]
    dtype: np.dtype
    pattern_offset: int | None = None
    h5_file: h5py.File | None = None
    patterns_ds: h5py.Dataset | None = None
    up_reader: UPPatternReader | None = None
    up_file: object | None = None

    def __enter__(self) -> "ResidualPatternWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def create(cls, source_data: LoadedInputData, output_path: str) -> "ResidualPatternWriter":
        src = Path(source_data.pattern_path).expanduser().resolve()
        out = Path(output_path).expanduser().resolve()
        if out == src:
            raise ValueError("Residual output path must be different from the source pattern file.")

        if source_data.source_type == "h5oina":
            if out.suffix.lower() != ".h5oina":
                out = out.with_suffix(".h5oina")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            h5 = h5py.File(out, "r+")
            patterns_ds = h5["1/EBSD/Data/Processed Patterns"]
            return cls(
                output_path=out,
                source_type="h5oina",
                dtype=np.dtype(patterns_ds.dtype),
                h5_file=h5,
                patterns_ds=patterns_ds,
            )

        if source_data.source_type == "up_ang":
            reader = source_data.up_pattern_reader
            if reader is None:
                raise RuntimeError("UP pattern writer requires a loaded UP pattern reader.")
            if src.suffix.lower() not in {".up1", ".up2"}:
                raise ValueError(f"Unsupported UP output source extension '{src.suffix}'.")
            if out.suffix.lower() != src.suffix.lower():
                out = out.with_suffix(src.suffix.lower())
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            up_file = open(out, "r+b")
            return cls(
                output_path=out,
                source_type="up_ang",
                dtype=np.dtype(reader.dtype),
                pattern_offset=int(reader.pattern_offset),
                up_reader=reader,
                up_file=up_file,
            )

        raise ValueError(f"Unsupported source type '{source_data.source_type}' for residual pattern writing.")

    def write(self, index: int, residual_pattern: np.ndarray) -> None:
        if self.source_type == "h5oina":
            if self.patterns_ds is None:
                raise RuntimeError("H5OINA writer is not initialized.")
            arr_u8 = _residual_to_uint8(residual_pattern)
            if self.dtype == np.dtype(np.uint16):
                arr_write = (arr_u8.astype(np.uint16) * 257).astype(np.uint16, copy=False)
            elif self.dtype == np.dtype(np.uint8):
                arr_write = arr_u8
            else:
                arr_write = arr_u8.astype(self.dtype, copy=False)
            self.patterns_ds[int(index)] = arr_write
            return
        if self.source_type == "up_ang":
            if self.up_reader is None or self.up_file is None or self.pattern_offset is None:
                raise RuntimeError("UP writer is not initialized.")
            _write_up_pattern_at(
                str(self.output_path),
                self.up_reader,
                int(index),
                _residual_to_uint8(residual_pattern),
                file_obj=self.up_file,
            )
            return
        raise RuntimeError(f"Unsupported writer type '{self.source_type}'.")

    def close(self) -> None:
        try:
            if self.h5_file is not None:
                self.h5_file.close()
        finally:
            self.h5_file = None
            self.patterns_ds = None
            if self.up_file is not None:
                try:
                    self.up_file.close()
                finally:
                    self.up_file = None


def _h5_phase_symmetries(h5_file: h5py.File) -> dict[int, object]:
    """Read phase symmetry from indexed H5OINA metadata, not the master pattern."""
    from orix.quaternion.symmetry import get_point_group

    out: dict[int, object] = {}
    root_path = "1/Data Processing/Header/Phases"
    if root_path not in h5_file:
        root_path = "1/EBSD/Header/Phases"
    if root_path not in h5_file:
        return out
    for key, group in h5_file[root_path].items():
        try:
            phase_id = int(key)
        except (TypeError, ValueError):
            continue
        if "Space Group" not in group:
            continue
        try:
            space_group = int(np.ravel(group["Space Group"][()])[0])
            out[phase_id] = get_point_group(space_group, proper=False)
        except Exception:
            continue
    return out


def _symmetry_from_ang_name(name: str):
    """Return an orix Laue symmetry for EDAX ANG point-group aliases."""
    from orix.quaternion import symmetry

    alias = {
        "1": "-1",
        "-1": "-1",
        "2": "2/m",
        "20": "2/m",
        "22": "mmm",
        "222": "mmm",
        "42": "4/mmm",
        "422": "4/mmm",
        "32": "-3m",
        "321": "-3m",
        "62": "6/mmm",
        "622": "6/mmm",
        "43": "m-3m",
        "432": "m-3m",
        "m3m": "m-3m",
    }
    target = alias.get(str(name).strip().lower(), str(name).strip().lower())
    for attr in dir(symmetry):
        candidate = getattr(symmetry, attr)
        if getattr(candidate, "name", "").lower() == target:
            return getattr(candidate, "laue", candidate)
    return symmetry.C1


def _ang_phase_symmetries(header_lines: list[str]) -> dict[int, object]:
    out: dict[int, object] = {}
    phase_id: int | None = None
    for line in header_lines:
        txt = line.lstrip("#").strip()
        low = txt.lower()
        if low.startswith("phase"):
            parts = txt.replace(":", " ").split()
            try:
                phase_id = int(parts[-1])
            except (ValueError, IndexError):
                phase_id = None
        elif low.startswith("symmetry") and phase_id is not None:
            parts = txt.replace(":", " ").split()
            if len(parts) >= 2:
                out[phase_id] = _symmetry_from_ang_name(parts[-1])
    return out


def _energy_axis_values_kv_from_master_signal(mp_signal) -> np.ndarray | None:
    try:
        nav_axes = list(mp_signal.axes_manager.navigation_axes)
    except Exception:
        return None
    for ax in nav_axes:
        try:
            name = str(getattr(ax, "name", "")).lower()
            if "energy" not in name:
                continue
            vals = np.asarray(ax.axis, dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                return np.sort(finite.astype(np.float64, copy=False))
        except Exception:
            continue
    return None


def _highest_energy_kv_from_master_signal(mp_signal) -> float | None:
    vals = _energy_axis_values_kv_from_master_signal(mp_signal)
    if vals is None or vals.size == 0:
        return None
    return float(np.max(vals))


def _parse_ang_header_with_lines(path: str) -> tuple[dict[str, str], list[str]]:
    out: dict[str, str] = {}
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
            txt = line[1:].strip()
            if not txt:
                continue
            if ":" in txt:
                key, val = txt.split(":", 1)
                out[key.strip().lower()] = val.strip()
                continue
            parts = txt.split(None, 1)
            if len(parts) == 2:
                out[parts[0].strip().lower()] = parts[1].strip()
    return out, lines


def _parse_header_int(meta: dict[str, str], key: str, default: int | None = None) -> int | None:
    if key not in meta:
        return default
    try:
        return int(float(meta[key].split()[0]))
    except Exception:
        return default


def _parse_header_float(meta: dict[str, str], key: str, default: float | None = None) -> float | None:
    if key not in meta:
        return default
    try:
        return float(meta[key].split()[0])
    except Exception:
        return default


def _open_edax_up_reader(path: str) -> UPPatternReader:
    ext = Path(path).suffix.lower()
    if ext not in (".up1", ".up2"):
        raise ValueError(f"Unsupported pattern file extension '{ext}'. Expected .up1 or .up2.")

    dtype = np.uint8 if ext == ".up1" else np.uint16
    file_size = int(os.path.getsize(path))
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
    available_bytes = int(file_size - pattern_offset)
    n_patterns = int(available_bytes // bytes_per_pattern)
    if n_patterns <= 0:
        raise ValueError(f"No patterns found in {path}")

    return UPPatternReader(
        path=str(path),
        dtype=np.dtype(dtype),
        pattern_offset=int(pattern_offset),
        n_patterns=int(n_patterns),
        h=int(sy),
        w=int(sx),
    )


def _left_multiply_eulers_zxz(eulers_rad: np.ndarray, angle_rad: float) -> np.ndarray:
    """Apply a fixed left multiplication Rz(angle) to Bunge ZXZ Euler angles."""
    from transforms3d.euler import euler2mat, mat2euler

    arr = np.asarray(eulers_rad, dtype=np.float64)
    flat = arr.reshape(-1, 3)
    rz = euler2mat(float(angle_rad), 0.0, 0.0, axes="rzxz")
    out = np.empty_like(flat, dtype=np.float64)
    for i in range(flat.shape[0]):
        r = euler2mat(float(flat[i, 0]), float(flat[i, 1]), float(flat[i, 2]), axes="rzxz")
        rp = rz @ r
        out[i, :] = np.array(mat2euler(rp, axes="rzxz"), dtype=np.float64)
    return out.reshape(arr.shape)


def _load_ang_columns(
    path: str, expected_rows: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    # Stream parse to avoid large temporary allocations from np.loadtxt on huge ANG files.
    if expected_rows is None or int(expected_rows) <= 0:
        n_rows = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                n_rows += 1
    else:
        n_rows = int(expected_rows)

    eulers = np.empty((n_rows, 3), dtype=np.float64)
    x = np.empty(n_rows, dtype=np.float64)
    y = np.empty(n_rows, dtype=np.float64)
    quality1 = np.empty(n_rows, dtype=np.float64)
    quality2 = np.empty(n_rows, dtype=np.float64)
    phase = np.empty(n_rows, dtype=np.int32)

    i = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = np.fromstring(line, dtype=np.float64, sep=" ")
            if vals.size < 8:
                raise ValueError(f"{path}: expected at least 8 columns in .ang data, found {vals.size} on row {i + 1}.")
            if i >= n_rows:
                # Fallback if row count was underestimated.
                grow = max(1024, n_rows // 8 if n_rows > 0 else 1024)
                eulers = np.vstack((eulers, np.empty((grow, 3), dtype=np.float64)))
                x = np.concatenate((x, np.empty(grow, dtype=np.float64)))
                y = np.concatenate((y, np.empty(grow, dtype=np.float64)))
                quality1 = np.concatenate((quality1, np.empty(grow, dtype=np.float64)))
                quality2 = np.concatenate((quality2, np.empty(grow, dtype=np.float64)))
                phase = np.concatenate((phase, np.empty(grow, dtype=np.int32)))
                n_rows += grow
            eulers[i, :] = vals[:3]
            x[i] = vals[3]
            y[i] = vals[4]
            quality1[i] = vals[5]
            quality2[i] = vals[6]
            phase[i] = int(np.rint(vals[7]))
            i += 1

    if i != n_rows:
        eulers = eulers[:i]
        x = x[:i]
        y = y[:i]
        quality1 = quality1[:i]
        quality2 = quality2[:i]
        phase = phase[:i]

    was_deg = False
    max_abs = float(np.nanmax(np.abs(eulers))) if eulers.size else 0.0
    if np.isfinite(max_abs) and max_abs > (2.0 * np.pi + 0.5):
        eulers = np.deg2rad(eulers)
        was_deg = True
    return eulers, x, y, quality1, quality2, phase, was_deg


def _compute_direction_cosines_kikuchipy(
    nrows: int,
    ncols: int,
    pc: tuple[float, float, float],
    pc_convention: str,
    sample_tilt_deg: float,
    detector_tilt_deg: float,
    azimuthal_deg: float,
    twist_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    import kikuchipy as kp
    from kikuchipy.signals.util._master_pattern import _get_direction_cosines_for_fixed_pc

    det = kp.detectors.EBSDDetector(
        shape=(int(nrows), int(ncols)),
        pc=[float(pc[0]), float(pc[1]), float(pc[2])],
        convention=str(pc_convention).lower(),
        sample_tilt=float(sample_tilt_deg),
        tilt=float(detector_tilt_deg),
        azimuthal=float(azimuthal_deg),
        twist=float(twist_deg),
    )
    gnomonic_bounds = np.asarray(det.gnomonic_bounds, dtype=np.float64).reshape(-1)
    pcz = float(np.asarray(det.pc, dtype=np.float64).reshape(-1, 3)[0, 2])
    om_detector_to_sample = (~det.sample_to_detector).to_matrix().squeeze().astype(np.float64)
    signal_mask = np.ones(int(nrows * ncols), dtype=bool)
    dc = _get_direction_cosines_for_fixed_pc(
        gnomonic_bounds,
        pcz,
        int(nrows),
        int(ncols),
        om_detector_to_sample,
        signal_mask,
    )
    return dc.T.astype(np.float64), np.asarray(det.pc, dtype=np.float64).reshape(-1, 3)[0]


def _convert_pc_map(
    pc_map: np.ndarray,
    src_convention: str,
    dst_convention: str,
    shape: tuple[int, int],
    sample_tilt_deg: float,
    detector_tilt_deg: float,
    azimuthal_deg: float,
    twist_deg: float,
) -> np.ndarray:
    if src_convention.lower() == dst_convention.lower():
        return pc_map.astype(np.float64, copy=True)
    import kikuchipy as kp

    det = kp.detectors.EBSDDetector(
        shape=shape,
        pc=pc_map.astype(np.float64, copy=False),
        convention=src_convention.lower(),
        sample_tilt=float(sample_tilt_deg),
        tilt=float(detector_tilt_deg),
        azimuthal=float(azimuthal_deg),
        twist=float(twist_deg),
    )
    dst = dst_convention.lower()
    if dst in {"bruker"}:
        return np.asarray(det.pc_bruker(), dtype=np.float64)
    if dst in {"edax", "tsl", "amatek"}:
        return np.asarray(det.pc_tsl(), dtype=np.float64)
    if dst in {"oxford", "aztec"}:
        return np.asarray(det.pc_oxford(), dtype=np.float64)
    if dst in {"emsoft", "emsoft4", "emsoft5"}:
        return np.asarray(det.pc_emsoft(), dtype=np.float64)
    raise ValueError(f"Unsupported destination PC convention '{dst_convention}'.")


class WorkflowSession:
    def __init__(self) -> None:
        self.data: LoadedInputData | None = None
        self.master: MasterPatternModel | None = None

        self.current_eulers_rad: np.ndarray | None = None
        self.initial_eulers_rad: np.ndarray | None = None
        self.current_phases: np.ndarray | None = None
        self.current_pc_bruker: np.ndarray | None = None
        self.current_pc_custom: np.ndarray | None = None

        self.last_scores_map: np.ndarray | None = None
        self.last_action_note: str = ""
        self._orientation_color_cache: dict[str, np.ndarray] = {}
        self.calibration_indices: list[int] = []
        self.calibrated_center_pc_bruker: np.ndarray | None = None
        self.calibrated_center_pc_custom: np.ndarray | None = None
        self.dictionary_cache: DictionaryCache | None = None
        self.dictionary_settings: dict[str, object] | None = None
        self.last_indexed_indices: np.ndarray | None = None
        self.indexed_candidate_eulers_rad: np.ndarray | None = None
        self.residual_eulers_rad: np.ndarray | None = None
        self.residual_phases: np.ndarray | None = None
        self.last_residual_scores_map: np.ndarray | None = None
        self.last_residual_indexed_indices: np.ndarray | None = None
        self.residual_candidate_eulers_rad: np.ndarray | None = None
        self.residual_point_results: dict[int, OverlapPointResult] = {}
        self.last_overlap: OverlapPointResult | None = None
        self.residual_pattern_output_path: str | None = None
        self._residual_pattern_source_cache: tuple[str, object] | None = None
        self.overlap_mixture_results: dict[int, OverlapMixtureResult] = {}
        self.last_overlap_mixture: OverlapMixtureResult | None = None
        self.overlap_primary_fraction_map: np.ndarray | None = None
        self.overlap_secondary_fraction_map: np.ndarray | None = None
        self.overlap_mixture_ncc_map: np.ndarray | None = None
        self.pattern_mask_config = _pattern_mask_config_from_option(-1)
        self.dynamic_bg_config = DynamicBackgroundConfig()

    # -------------------- Index mapping helpers -------------------- #

    @property
    def pattern_mask_option(self) -> int:
        return self.pattern_mask_config.option

    def pattern_mask_description(self, shape: tuple[int, int] | None = None) -> str:
        if shape is None and self.data is not None:
            shape = (self.data.h, self.data.w)
        return _describe_pattern_mask(self.pattern_mask_config, shape)

    def _invalidate_pattern_conditioning_results(self) -> None:
        self._invalidate_residual_cache()
        self.last_overlap = None
        self.last_indexed_indices = None
        if self.last_scores_map is not None:
            self.last_scores_map[:] = np.nan

    def set_pattern_mask_option(self, option: int) -> str:
        config = _pattern_mask_config_from_option(int(option))
        if config == self.pattern_mask_config:
            return f"Pattern mask unchanged: {self.pattern_mask_description()}."
        self.pattern_mask_config = config
        self._invalidate_pattern_conditioning_results()
        self.last_action_note = "Existing dictionary patterns remain usable; rerun indexing/refinement to use the new pattern conditioning."
        return f"Pattern mask set to {self.pattern_mask_description()}."

    def dynamic_background_description(self, shape: tuple[int, int] | None = None) -> str:
        if shape is None and self.data is not None:
            shape = (self.data.h, self.data.w)
        return _describe_dynamic_background(self.dynamic_bg_config, shape)

    def set_dynamic_background(
        self,
        enabled: bool,
        *,
        std_px: float = 0.0,
        truncate: float = 4.0,
    ) -> str:
        config = _dynamic_bg_config(bool(enabled), std_px=std_px, truncate=truncate)
        if config == self.dynamic_bg_config:
            return f"Dynamic background subtraction unchanged: {self.dynamic_background_description()}."
        self.dynamic_bg_config = config
        self._invalidate_pattern_conditioning_results()
        self.last_action_note = (
            "Existing dictionary patterns remain usable; rerun indexing/refinement to use the new pattern conditioning."
        )
        return f"Dynamic background subtraction set to {self.dynamic_background_description()}."

    def _signal_mask_for_shape(
        self,
        shape: tuple[int, int],
        *,
        software_binning: int = 1,
    ) -> np.ndarray | None:
        return _signal_mask_from_pattern_mask(
            shape,
            self.pattern_mask_config,
            software_binning=int(software_binning),
        )

    def _signal_mask_for_full_pattern(self) -> np.ndarray | None:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        return self._signal_mask_for_shape((self.data.h, self.data.w))

    def _signal_mask_for_dictionary_cache(self, cache: DictionaryCache) -> np.ndarray | None:
        return self._signal_mask_for_shape(
            cache.pattern_shape,
            software_binning=int(cache.software_binning),
        )

    def row_col_from_index(self, index: int) -> tuple[int, int]:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        return divmod(int(index), int(self.data.cols))

    def index_from_row_col(self, row: int, col: int) -> int:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        if row < 0 or col < 0 or row >= self.data.rows or col >= self.data.cols:
            raise ValueError(f"row/col out of bounds: row={row}, col={col}")
        return int(row * self.data.cols + col)

    def roi_indices(self, r0: int, c0: int, nrows: int, ncols: int) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        rr0 = max(0, min(int(r0), self.data.rows - 1))
        cc0 = max(0, min(int(c0), self.data.cols - 1))
        nr = max(1, int(nrows))
        nc = max(1, int(ncols))
        rr1 = min(self.data.rows, rr0 + nr)
        cc1 = min(self.data.cols, cc0 + nc)
        rows = np.arange(rr0, rr1, dtype=np.int64)
        cols = np.arange(cc0, cc1, dtype=np.int64)
        grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
        return (grid_r * self.data.cols + grid_c).ravel().astype(np.int64)

    def _invalidate_orientation_cache(self) -> None:
        self._orientation_color_cache.clear()

    def _invalidate_residual_cache(self) -> None:
        self.residual_eulers_rad = None
        self.residual_phases = None
        self.last_residual_scores_map = None
        self.last_residual_indexed_indices = None
        self.residual_point_results.clear()
        self.last_overlap = None
        self.residual_pattern_output_path = None
        self._clear_residual_pattern_source_cache()
        self._invalidate_residual_color_cache()
        self._invalidate_overlap_mixture_cache()
        self.residual_candidate_eulers_rad = None

    def _reset_indexed_candidate_cache(self, keep_n: int | None = None) -> None:
        if self.data is None:
            self.indexed_candidate_eulers_rad = None
            return
        if keep_n is None or int(keep_n) <= 1:
            self.indexed_candidate_eulers_rad = None
            return
        self.indexed_candidate_eulers_rad = np.full(
            (self.data.count, int(keep_n), 3),
            np.nan,
            dtype=np.float64,
        )

    def _reset_residual_candidate_cache(self, keep_n: int | None = None) -> None:
        if self.data is None:
            self.residual_candidate_eulers_rad = None
            return
        if keep_n is None or int(keep_n) <= 1:
            self.residual_candidate_eulers_rad = None
            return
        self.residual_candidate_eulers_rad = np.full(
            (self.data.count, int(keep_n), 3),
            np.nan,
            dtype=np.float64,
        )

    def _clear_indexed_candidate_rows(self, indices: np.ndarray | None = None) -> None:
        if self.indexed_candidate_eulers_rad is None:
            return
        if indices is None:
            self.indexed_candidate_eulers_rad = None
            return
        idx = np.asarray(indices, dtype=np.int64).ravel()
        if idx.size == 0:
            return
        self.indexed_candidate_eulers_rad[idx] = np.nan

    def _clear_residual_candidate_rows(self, indices: np.ndarray | None = None) -> None:
        if self.residual_candidate_eulers_rad is None:
            return
        if indices is None:
            self.residual_candidate_eulers_rad = None
            return
        idx = np.asarray(indices, dtype=np.int64).ravel()
        if idx.size == 0:
            return
        self.residual_candidate_eulers_rad[idx] = np.nan

    def _invalidate_residual_color_cache(self) -> None:
        for key in list(self._orientation_color_cache.keys()):
            if key.startswith("RES-IPF-"):
                self._orientation_color_cache.pop(key, None)

    def _invalidate_overlap_mixture_cache(self) -> None:
        self.overlap_mixture_results.clear()
        self.last_overlap_mixture = None
        self.overlap_primary_fraction_map = None
        self.overlap_secondary_fraction_map = None
        self.overlap_mixture_ncc_map = None

    def _clear_residual_pattern_source_cache(self) -> None:
        self._residual_pattern_source_cache = None

    def _kikuchipy_frame_active(self) -> bool:
        return self.data is not None and self.data.source_type == "h5oina"

    def _eulers_to_kikuchipy_frame(self, eulers_rad: np.ndarray) -> np.ndarray:
        arr = np.asarray(eulers_rad, dtype=np.float64)
        if not self._kikuchipy_frame_active():
            return arr.copy()
        return _left_multiply_eulers_zxz(arr, angle_rad=np.deg2rad(90.0))

    def _eulers_from_kikuchipy_frame(self, eulers_rad: np.ndarray) -> np.ndarray:
        arr = np.asarray(eulers_rad, dtype=np.float64)
        if not self._kikuchipy_frame_active():
            return arr.copy()
        return _left_multiply_eulers_zxz(arr, angle_rad=np.deg2rad(-90.0))

    def available_layers(self) -> list[str]:
        if self.data is None:
            return []
        layers = list(self.data.map_layers.keys())
        if not layers:
            layers = ["Phase"]
        if ORIENTATION_LAYER_LABEL not in layers:
            layers.append(ORIENTATION_LAYER_LABEL)
        return layers

    def get_layer_map(self, label: str) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        key = str(label).strip()
        if key.upper() in {ORIENTATION_LAYER_LABEL.upper(), "IPF", "IPF-Z"}:
            return self.get_ipf_color_map(direction="z")
        if key in self.data.map_layers:
            return self.data.map_layers[key]
        if key.lower() == "phase" and self.current_phases is not None:
            return self.current_phases.reshape(self.data.rows, self.data.cols).astype(np.float32, copy=False)
        if self.data.map_layers:
            return self.data.map_layers[next(iter(self.data.map_layers.keys()))]
        return np.zeros((self.data.rows, self.data.cols), dtype=np.float32)

    def _ipf_color_map_from_arrays(
        self,
        eulers_rad: np.ndarray,
        phases: np.ndarray | None,
        *,
        direction: Literal["x", "y", "z"] = "z",
        cache_key_prefix: str = "IPF",
    ) -> np.ndarray:
        from orix.plot import IPFColorKeyTSL
        from orix.quaternion import Orientation, symmetry
        from orix.vector import Vector3d

        if self.data is None:
            raise RuntimeError("Load input data first.")

        dir_key = str(direction).strip().lower()
        if dir_key not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported direction '{direction}', expected x/y/z.")
        cache_key = f"{cache_key_prefix}-{dir_key.upper()}"
        cached = self._orientation_color_cache.get(cache_key)
        if cached is not None and cached.shape == (self.data.rows, self.data.cols, 3):
            return cached

        if dir_key == "x":
            v = Vector3d.xvector()
        elif dir_key == "y":
            v = Vector3d.yvector()
        else:
            v = Vector3d.zvector()

        phase_ids = np.asarray(phases, dtype=np.int32).reshape(-1) if phases is not None else np.zeros(self.data.count, dtype=np.int32)
        euler_arr = np.asarray(eulers_rad, dtype=np.float64).reshape(-1, 3)
        valid_euler = np.all(np.isfinite(euler_arr), axis=1)
        rgb_flat = np.ones((self.data.count, 3), dtype=np.float32)
        metadata_symmetries = self.data.phase_symmetries
        fallback_sym = next(iter(metadata_symmetries.values())) if len(metadata_symmetries) == 1 else symmetry.C1
        for phase_id in np.unique(phase_ids).tolist():
            if phase_id < 0:
                continue
            mask = (phase_ids == int(phase_id)) & valid_euler
            if not np.any(mask):
                continue
            sym = metadata_symmetries.get(int(phase_id), fallback_sym)
            sym = getattr(sym, "laue", sym)
            orientations = Orientation.from_euler(euler_arr[mask], symmetry=sym, degrees=False)
            ckey = IPFColorKeyTSL(symmetry=sym, direction=v)
            rgb_flat[mask] = np.asarray(ckey.orientation2color(orientations), dtype=np.float32).reshape(-1, 3)
        rgb = np.clip(rgb_flat.reshape(self.data.rows, self.data.cols, 3), 0.0, 1.0)
        self._orientation_color_cache[cache_key] = rgb
        return rgb

    def get_ipf_color_map(self, direction: Literal["x", "y", "z"] = "z") -> np.ndarray:
        if self.current_eulers_rad is None:
            raise RuntimeError("Load input data first.")
        return self._ipf_color_map_from_arrays(
            self.current_eulers_rad,
            self.current_phases,
            direction=direction,
            cache_key_prefix="IPF",
        )

    def get_preliminary_ipf_color_map(self, direction: Literal["x", "y", "z"] = "z") -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        eulers = self.initial_eulers_rad if self.initial_eulers_rad is not None else self.current_eulers_rad
        if eulers is None:
            raise RuntimeError("Preliminary orientations are not available.")
        phases = self.data.phases if self.data.phases is not None else self.current_phases
        return self._ipf_color_map_from_arrays(
            eulers,
            phases,
            direction=direction,
            cache_key_prefix="PRE-IPF",
        )

    def get_residual_ipf_color_map(self, direction: Literal["x", "y", "z"] = "z") -> np.ndarray:
        if self.residual_eulers_rad is None:
            raise RuntimeError("Run residual ROI indexing first.")
        return self._ipf_color_map_from_arrays(
            self.residual_eulers_rad,
            self.residual_phases if self.residual_phases is not None else self.current_phases,
            direction=direction,
            cache_key_prefix="RES-IPF",
        )

    def get_residual_point_result(self, index: int) -> OverlapPointResult | None:
        result = self.residual_point_results.get(int(index))
        if result is None:
            return None
        if (
            result.residual is None
            or result.experimental is None
            or result.simulated is None
            or (result.secondary_euler_rad is not None and result.secondary_simulated is None)
        ):
            result = self._materialize_residual_point_result(result)
            self.residual_point_results[int(index)] = result
        return result

    def get_primary_index_ncc(self, index: int) -> float | None:
        if self.data is None or self.last_scores_map is None:
            return None
        row, col = self.row_col_from_index(int(index))
        score = float(self.last_scores_map[row, col])
        if not np.isfinite(score):
            return None
        return score

    def _ensure_residual_state(self) -> None:
        if self.data is None or self.current_eulers_rad is None or self.current_phases is None:
            raise RuntimeError("Load input data first.")
        if self.residual_eulers_rad is None or self.residual_eulers_rad.shape != self.current_eulers_rad.shape:
            self.residual_eulers_rad = np.full_like(self.current_eulers_rad, np.nan, dtype=np.float64)
        if self.residual_phases is None or self.residual_phases.shape != self.current_phases.shape:
            self.residual_phases = self.current_phases.copy()
        if self.last_residual_scores_map is None or self.last_residual_scores_map.shape != (self.data.rows, self.data.cols):
            self.last_residual_scores_map = np.full((self.data.rows, self.data.cols), np.nan, dtype=np.float32)

    def _ensure_overlap_mixture_state(self) -> None:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        shape = (self.data.rows, self.data.cols)
        if self.overlap_primary_fraction_map is None or self.overlap_primary_fraction_map.shape != shape:
            self.overlap_primary_fraction_map = np.full(shape, np.nan, dtype=np.float32)
        if self.overlap_secondary_fraction_map is None or self.overlap_secondary_fraction_map.shape != shape:
            self.overlap_secondary_fraction_map = np.full(shape, np.nan, dtype=np.float32)
        if self.overlap_mixture_ncc_map is None or self.overlap_mixture_ncc_map.shape != shape:
            self.overlap_mixture_ncc_map = np.full(shape, np.nan, dtype=np.float32)

    def _strip_residual_point_result(self, result: OverlapPointResult) -> OverlapPointResult:
        secondary_euler = None if result.secondary_euler_rad is None else np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3)
        return OverlapPointResult(
            index=int(result.index),
            row=int(result.row),
            col=int(result.col),
            ncc_es=float(result.ncc_es),
            scale=float(result.scale),
            ncc_residual_sim=float(result.ncc_residual_sim),
            experimental=None,
            simulated=None,
            residual=None,
            simulated_unfitted=None,
            blurred_simulated=None,
            gain_map=None,
            fitted_sigma=float(result.fitted_sigma),
            gain_params=tuple(float(v) for v in result.gain_params),
            ellipse_params=tuple(float(v) for v in result.ellipse_params),
            ncc_unfitted=None if result.ncc_unfitted is None else float(result.ncc_unfitted),
            fit_success=bool(result.fit_success),
            fit_message=str(result.fit_message),
            secondary_dictionary_ncc_kp=None
            if result.secondary_dictionary_ncc_kp is None
            else float(result.secondary_dictionary_ncc_kp),
            secondary_ncc_kp=None if result.secondary_ncc_kp is None else float(result.secondary_ncc_kp),
            secondary_ncc_full=None if result.secondary_ncc_full is None else float(result.secondary_ncc_full),
            secondary_euler_rad=secondary_euler,
            secondary_simulated=None,
            secondary_refined=bool(result.secondary_refined),
            secondary_refinement_note=str(result.secondary_refinement_note),
        )

    def _strip_overlap_mixture_result(self, result: OverlapMixtureResult) -> OverlapMixtureResult:
        return OverlapMixtureResult(
            index=int(result.index),
            row=int(result.row),
            col=int(result.col),
            primary_fraction=float(result.primary_fraction),
            secondary_fraction=float(result.secondary_fraction),
            primary_coefficient=float(result.primary_coefficient),
            secondary_coefficient=float(result.secondary_coefficient),
            ncc_mixture=float(result.ncc_mixture),
            residual_rms=float(result.residual_rms),
            old_primary_ncc=None if result.old_primary_ncc is None else float(result.old_primary_ncc),
            old_secondary_ncc=None if result.old_secondary_ncc is None else float(result.old_secondary_ncc),
            experimental=None,
            primary_simulated=None,
            secondary_simulated=None,
            combined_simulated=None,
            residual=None,
            gain_map=None,
            fitted_sigma=float(result.fitted_sigma),
            gain_params=tuple(float(v) for v in result.gain_params),
            ellipse_params=tuple(float(v) for v in result.ellipse_params),
            component_correlation=float(result.component_correlation),
            primary_euler_rad=None
            if result.primary_euler_rad is None
            else np.asarray(result.primary_euler_rad, dtype=np.float64).reshape(3).copy(),
            secondary_euler_rad=None
            if result.secondary_euler_rad is None
            else np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3).copy(),
            fit_success=bool(result.fit_success),
            fit_message=str(result.fit_message),
            orientation_refined=bool(result.orientation_refined),
            orientation_refinement_note=str(result.orientation_refinement_note),
            initial_mixture_ncc=None
            if result.initial_mixture_ncc is None
            else float(result.initial_mixture_ncc),
            primary_euler_delta_deg=tuple(float(v) for v in result.primary_euler_delta_deg),
            secondary_euler_delta_deg=tuple(float(v) for v in result.secondary_euler_delta_deg),
        )

    def _materialize_residual_point_result(self, result: OverlapPointResult) -> OverlapPointResult:
        if self.data is None or self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(result.index)
        weights = self._overlap_weights()
        experimental_raw = self._processed_pattern_at(idx)
        simulated_raw = self._simulate_pattern_for_euler(idx, self.current_eulers_rad[idx])
        experimental = _normalize_weighted(experimental_raw, weights)
        simulated_unfitted = _normalize_weighted(simulated_raw, weights)
        blurred = _normalize_weighted(gaussian_filter(simulated_raw, sigma=float(result.fitted_sigma)), weights)
        gain_params = tuple(float(v) for v in result.gain_params) if len(result.gain_params) >= 3 else (1.0, 1.0, 1.0)
        ellipse_params = tuple(float(v) for v in result.ellipse_params) if len(result.ellipse_params) >= 4 else (1.0, 1.0, 0.0, 0.0)
        gain_map = _power_gain_map(experimental.shape, gain_params, ellipse_params)
        processed = _normalize_weighted(blurred * gain_map, weights)
        scale = float(result.scale)
        fitted_ncc = float(result.ncc_es)
        residual = experimental - scale * processed
        ncc_unfitted = result.ncc_unfitted
        if ncc_unfitted is None:
            ncc_unfitted = _weighted_ncc(experimental, simulated_unfitted, weights)
        ncc_residual = result.ncc_residual_sim
        if ncc_residual is None:
            ncc_residual = _weighted_ncc(residual, processed, weights)
        secondary_sim = None
        secondary_ncc_full = None if result.secondary_ncc_full is None else float(result.secondary_ncc_full)
        if result.secondary_euler_rad is not None:
            secondary_sim = self._normalize_pattern_for_overlap(self._simulate_pattern_for_euler(idx, result.secondary_euler_rad))
            if secondary_ncc_full is None:
                secondary_ncc_full = self._pattern_ncc_for_overlap(residual, secondary_sim)
        return OverlapPointResult(
            index=int(result.index),
            row=int(result.row),
            col=int(result.col),
            ncc_es=fitted_ncc,
            scale=scale,
            ncc_residual_sim=float(ncc_residual),
            experimental=_zero_unweighted_pixels(experimental, weights),
            simulated=_zero_unweighted_pixels(processed, weights),
            residual=_zero_unweighted_pixels(residual, weights),
            simulated_unfitted=_zero_unweighted_pixels(simulated_unfitted, weights),
            blurred_simulated=_zero_unweighted_pixels(blurred, weights),
            gain_map=_zero_unweighted_pixels(gain_map, weights),
            fitted_sigma=float(result.fitted_sigma),
            gain_params=tuple(float(v) for v in gain_params),
            ellipse_params=tuple(float(v) for v in ellipse_params),
            ncc_unfitted=None if ncc_unfitted is None else float(ncc_unfitted),
            fit_success=bool(result.fit_success),
            fit_message=str(result.fit_message),
            secondary_dictionary_ncc_kp=None
            if result.secondary_dictionary_ncc_kp is None
            else float(result.secondary_dictionary_ncc_kp),
            secondary_ncc_kp=None if result.secondary_ncc_kp is None else float(result.secondary_ncc_kp),
            secondary_ncc_full=secondary_ncc_full,
            secondary_euler_rad=None
            if result.secondary_euler_rad is None
            else np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3),
            secondary_simulated=secondary_sim,
            secondary_refined=bool(result.secondary_refined),
            secondary_refinement_note=str(result.secondary_refinement_note),
        )

    def _secondary_overlap_orientation_for_index(
        self,
        index: int,
        residual_result: OverlapPointResult | None = None,
    ) -> tuple[np.ndarray | None, float | None, float | None]:
        idx = int(index)
        result = residual_result if residual_result is not None and int(residual_result.index) == idx else None
        if result is None:
            result = self.residual_point_results.get(idx)

        old_primary_ncc: float | None = None
        old_secondary_ncc: float | None = None
        secondary_euler: np.ndarray | None = None

        if result is not None:
            old_primary_ncc = float(result.ncc_es)
            if result.secondary_ncc_full is not None:
                old_secondary_ncc = float(result.secondary_ncc_full)
            elif result.secondary_ncc_kp is not None:
                old_secondary_ncc = float(result.secondary_ncc_kp)
            if result.secondary_euler_rad is not None:
                secondary_euler = np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3)

        if old_primary_ncc is None:
            old_primary_ncc = self.get_primary_index_ncc(idx)

        if secondary_euler is None and result is None and self.residual_eulers_rad is not None:
            candidate = np.asarray(self.residual_eulers_rad[idx], dtype=np.float64).reshape(3)
            if np.all(np.isfinite(candidate)):
                secondary_euler = candidate
                if self.last_residual_scores_map is not None and self.data is not None:
                    row, col = self.row_col_from_index(idx)
                    score = float(self.last_residual_scores_map[row, col])
                    if np.isfinite(score):
                        old_secondary_ncc = score

        return secondary_euler, old_primary_ncc, old_secondary_ncc

    def _store_overlap_mixture_result(self, result: OverlapMixtureResult, *, keep_patterns: bool) -> None:
        self._ensure_overlap_mixture_state()
        idx = int(result.index)
        stored = result if keep_patterns else self._strip_overlap_mixture_result(result)
        self.overlap_mixture_results[idx] = stored
        if keep_patterns:
            self.last_overlap_mixture = result
        if self.overlap_primary_fraction_map is not None:
            self.overlap_primary_fraction_map[int(result.row), int(result.col)] = float(result.primary_fraction)
        if self.overlap_secondary_fraction_map is not None:
            self.overlap_secondary_fraction_map[int(result.row), int(result.col)] = float(result.secondary_fraction)
        if self.overlap_mixture_ncc_map is not None:
            self.overlap_mixture_ncc_map[int(result.row), int(result.col)] = float(result.ncc_mixture)

    def _materialize_overlap_mixture_result(self, result: OverlapMixtureResult) -> OverlapMixtureResult:
        if self.data is None or self.current_eulers_rad is None:
            raise RuntimeError("Session state is not initialized.")
        if result.secondary_euler_rad is None:
            raise RuntimeError("Overlap mixture result has no secondary orientation.")

        idx = int(result.index)
        weights = self._overlap_weights()
        experimental_raw = self._processed_pattern_at(idx)
        primary_euler = (
            np.asarray(result.primary_euler_rad, dtype=np.float64).reshape(3)
            if result.primary_euler_rad is not None
            else np.asarray(self.current_eulers_rad[idx], dtype=np.float64).reshape(3)
        )
        secondary_euler = np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3)
        primary_raw = self._simulate_pattern_for_euler(idx, primary_euler)
        secondary_raw = self._simulate_pattern_for_euler(idx, secondary_euler)
        params = np.asarray(
            [
                float(result.fitted_sigma),
                *tuple(float(v) for v in result.gain_params[:3]),
                *tuple(float(v) for v in result.ellipse_params[:4]),
            ],
            dtype=np.float64,
        )
        if params.size != 8:
            raise ValueError("Stored overlap mixture parameters are incomplete.")
        fit = _evaluate_overlap_mixture_pattern(
            experimental_raw,
            primary_raw,
            secondary_raw,
            weights,
            params,
        )
        materialized = OverlapMixtureResult(
            index=int(result.index),
            row=int(result.row),
            col=int(result.col),
            primary_fraction=float(fit.primary_fraction),
            secondary_fraction=float(fit.secondary_fraction),
            primary_coefficient=float(fit.primary_coefficient),
            secondary_coefficient=float(fit.secondary_coefficient),
            ncc_mixture=float(fit.ncc_mixture),
            residual_rms=float(fit.residual_rms),
            old_primary_ncc=None if result.old_primary_ncc is None else float(result.old_primary_ncc),
            old_secondary_ncc=None if result.old_secondary_ncc is None else float(result.old_secondary_ncc),
            experimental=_zero_unweighted_pixels(fit.experimental, weights),
            primary_simulated=_zero_unweighted_pixels(fit.primary_processed, weights),
            secondary_simulated=_zero_unweighted_pixels(fit.secondary_processed, weights),
            combined_simulated=_zero_unweighted_pixels(fit.combined_simulated, weights),
            residual=_zero_unweighted_pixels(fit.residual, weights),
            gain_map=_zero_unweighted_pixels(fit.gain_map, weights),
            fitted_sigma=fit.sigma,
            gain_params=fit.gain_params,
            ellipse_params=fit.ellipse_params,
            component_correlation=float(fit.component_correlation),
            primary_euler_rad=primary_euler.copy(),
            secondary_euler_rad=secondary_euler.copy(),
            fit_success=bool(result.fit_success),
            fit_message=str(result.fit_message),
            orientation_refined=bool(result.orientation_refined),
            orientation_refinement_note=str(result.orientation_refinement_note),
            initial_mixture_ncc=None
            if result.initial_mixture_ncc is None
            else float(result.initial_mixture_ncc),
            primary_euler_delta_deg=tuple(float(v) for v in result.primary_euler_delta_deg),
            secondary_euler_delta_deg=tuple(float(v) for v in result.secondary_euler_delta_deg),
        )
        self.overlap_mixture_results[idx] = materialized
        self.last_overlap_mixture = materialized
        return materialized

    def get_overlap_mixture_result(self, index: int) -> OverlapMixtureResult | None:
        result = self.overlap_mixture_results.get(int(index))
        if result is None:
            return None
        if (
            result.experimental is None
            or result.primary_simulated is None
            or result.secondary_simulated is None
            or result.residual is None
        ):
            result = self._materialize_overlap_mixture_result(result)
        return result

    # ------------------------- Data loading ------------------------ #

    def load_input(self, pattern_path: str, orientation_path: str | None, geom: GeometryConfig) -> str:
        import kikuchipy as kp
        from transforms3d.euler import euler2mat

        p = str(Path(pattern_path).expanduser().resolve())
        if not Path(p).exists():
            raise FileNotFoundError(p)
        ext = Path(p).suffix.lower()

        if ext == ".h5oina":
            s = kp.load(p, lazy=True)
            rows, cols, h, w = map(int, s.data.shape)
            with h5py.File(p, "r") as f:
                eulers = f["1/Data Processing/Data/Euler"][()].reshape(-1, 3).astype(np.float64)
                phases = f["1/Data Processing/Data/Phase"][()].ravel().astype(np.int32)
                x = f["1/EBSD/Data/X"][()].ravel().astype(np.float64) if "1/EBSD/Data/X" in f else np.tile(np.arange(cols), rows).astype(np.float64)
                y = f["1/EBSD/Data/Y"][()].ravel().astype(np.float64) if "1/EBSD/Data/Y" in f else np.repeat(np.arange(rows), cols).astype(np.float64)
                beam_kv = float(f["1/EBSD/Header/Beam Voltage"][()][0]) if "1/EBSD/Header/Beam Voltage" in f else None
                tilt = float(f["1/EBSD/Header/Tilt Angle"][()][0]) if "1/EBSD/Header/Tilt Angle" in f else 0.0
                det_euler = f["1/EBSD/Header/Detector Orientation Euler"][()][0] if "1/EBSD/Header/Detector Orientation Euler" in f else np.zeros(3)
                det_rot = euler2mat(float(det_euler[0]), float(det_euler[1]), float(det_euler[2]), axes="rzxz")
                stage_rot = euler2mat(0.0, tilt, 0.0, axes="rzxz")
                rot_sd = stage_rot.T @ det_rot

                map_layers: dict[str, np.ndarray] = {}
                for label, candidates in MAP_LAYER_CANDIDATES.items():
                    for ds_path in candidates:
                        if ds_path in f:
                            arr = np.ravel(f[ds_path][()])
                            if arr.size >= rows * cols:
                                map_layers[label] = arr[: rows * cols].reshape(rows, cols).astype(np.float32)
                                break
                phase_symmetries = _h5_phase_symmetries(f)

            det = s.detector
            pc_bruker = np.asarray(det.pc_bruker(), dtype=np.float64).reshape(rows, cols, 3)
            pc_custom = np.asarray(det.pc_oxford(), dtype=np.float64).reshape(rows, cols, 3)
            self.data = LoadedInputData(
                source_type="h5oina",
                pattern_path=p,
                orientation_path=None,
                rows=rows,
                cols=cols,
                h=h,
                w=w,
                signal=s,
                eulers_rad=eulers,
                phases=phases,
                x_coords=x,
                y_coords=y,
                map_layers=map_layers,
                beam_kv=beam_kv,
                sample_tilt_deg=float(det.sample_tilt),
                detector_tilt_deg=float(det.tilt),
                azimuthal_deg=float(det.azimuthal),
                twist_deg=float(det.twist),
                rot_sd=rot_sd.astype(np.float64),
                direction_cosines=None,
                step_x=float(s.axes_manager.navigation_axes[0].scale) if len(s.axes_manager.navigation_axes) >= 1 else 1.0,
                step_y=float(s.axes_manager.navigation_axes[1].scale) if len(s.axes_manager.navigation_axes) >= 2 else 1.0,
                scan_unit=str(s.axes_manager.navigation_axes[0].units) if len(s.axes_manager.navigation_axes) >= 1 else "px",
                pc_output_convention="oxford",
                phase_symmetries=phase_symmetries,
                detector_px_size=float(det.px_size) if det.px_size is not None else None,
                detector_binning=float(det.binning) if det.binning is not None else None,
                up_pattern_reader=None,
            )
            self.initial_eulers_rad = eulers.copy()
            self.current_eulers_rad = self.data.eulers_rad
            self.current_phases = self.data.phases
            self.current_pc_bruker = pc_bruker.reshape(-1, 3).copy()
            self.current_pc_custom = pc_custom.reshape(-1, 3).copy()
            self._invalidate_orientation_cache()
            self._invalidate_residual_cache()
            self.last_scores_map = np.full((rows, cols), np.nan, dtype=np.float32)
            self.calibration_indices = []
            self.calibrated_center_pc_bruker = None
            self.calibrated_center_pc_custom = None
            self.dictionary_cache = None
            self.dictionary_settings = None
            self.last_indexed_indices = None
            self.indexed_candidate_eulers_rad = None
            self.residual_candidate_eulers_rad = None
            unique_ph = np.unique(phases)
            self.last_action_note = f"Phases in data: {unique_ph.tolist()}"
            return f"Loaded H5OINA: map={rows}x{cols}, pattern={h}x{w}, N={rows * cols}."

        if ext in (".up1", ".up2"):
            if not orientation_path:
                raise ValueError("For .up1/.up2 input, provide an orientation/indexing .ang file.")
            ang = str(Path(orientation_path).expanduser().resolve())
            if not Path(ang).exists():
                raise FileNotFoundError(ang)
            header, header_lines = _parse_ang_header_with_lines(ang)
            nrows = _parse_header_int(header, "nrows", None)
            ncols_odd = _parse_header_int(header, "ncols_odd", None)
            ncols_even = _parse_header_int(header, "ncols_even", ncols_odd)
            if nrows is None or ncols_odd is None:
                raise ValueError(f"{ang}: missing NROWS/NCOLS_ODD in header.")
            if ncols_even is not None and ncols_even != ncols_odd:
                raise ValueError(f"{ang}: hex grid not supported (NCOLS_ODD != NCOLS_EVEN).")

            eulers, x, y, q1, q2, phase, eulers_were_deg = _load_ang_columns(ang, expected_rows=int(nrows) * int(ncols_odd))
            rows = int(nrows)
            cols = int(ncols_odd)
            expected = rows * cols
            if eulers.shape[0] != expected:
                raise ValueError(f"{ang}: data rows ({eulers.shape[0]}) do not match NROWS*NCOLS ({expected}).")

            up_reader = _open_edax_up_reader(p)
            h = int(up_reader.h)
            w = int(up_reader.w)
            if int(up_reader.n_patterns) != expected:
                raise ValueError(f"{p}: pattern count {up_reader.n_patterns} does not match ANG grid {expected}.")

            # Do not build a full LazyEBSD map for huge UP files at load time.
            # Patterns are read from disk on demand in _pattern_at/_signal_from_indices.
            s = None

            if abs(float(geom.phi1_offset_deg)) > 1e-9:
                eulers[:, 0] = eulers[:, 0] + np.deg2rad(float(geom.phi1_offset_deg))

            x_star = _parse_header_float(header, "x-star", None)
            y_star = _parse_header_float(header, "y-star", None)
            z_star = _parse_header_float(header, "z-star", None)
            if x_star is None or y_star is None or z_star is None:
                raise ValueError(f"{ang}: missing x-star/y-star/z-star in header.")
            # Temporary user-requested behavior: interpret ANG x/y/z-star with Oxford scaling.
            # This is exact for Oxford patterns and typically close for square EDAX patterns.
            ang_pc_convention = "oxford"

            det = kp.detectors.EBSDDetector(
                shape=(h, w),
                pc=[float(x_star), float(y_star), float(z_star)],
                convention=ang_pc_convention,
                sample_tilt=float(geom.sample_tilt_deg),
                tilt=float(geom.detector_tilt_deg),
                azimuthal=float(geom.azimuthal_deg),
                twist=float(geom.twist_deg),
            )
            pc_bruker_1 = np.asarray(det.pc_bruker(), dtype=np.float64).reshape(3)
            pc_oxford_1 = np.asarray(det.pc_oxford(), dtype=np.float64).reshape(3)
            pc_bruker = np.tile(pc_bruker_1, (rows * cols, 1))
            pc_custom = np.tile(pc_oxford_1, (rows * cols, 1))

            dc, _pc_internal = _compute_direction_cosines_kikuchipy(
                h,
                w,
                pc=(float(x_star), float(y_star), float(z_star)),
                pc_convention=ang_pc_convention,
                sample_tilt_deg=float(geom.sample_tilt_deg),
                detector_tilt_deg=float(geom.detector_tilt_deg),
                azimuthal_deg=float(geom.azimuthal_deg),
                twist_deg=float(geom.twist_deg),
            )

            map_layers = {
                "IQ": q1.reshape(rows, cols).astype(np.float32),
                "CI": q2.reshape(rows, cols).astype(np.float32),
                "DP": q2.reshape(rows, cols).astype(np.float32),
                "NCC": q2.reshape(rows, cols).astype(np.float32),
                "Phase": phase.reshape(rows, cols).astype(np.float32),
                "X": x.reshape(rows, cols).astype(np.float32),
                "Y": y.reshape(rows, cols).astype(np.float32),
            }
            self.data = LoadedInputData(
                source_type="up_ang",
                pattern_path=p,
                orientation_path=ang,
                rows=rows,
                cols=cols,
                h=h,
                w=w,
                signal=s,
                eulers_rad=eulers.astype(np.float64),
                phases=phase.astype(np.int32),
                x_coords=x.astype(np.float64),
                y_coords=y.astype(np.float64),
                map_layers=map_layers,
                beam_kv=None,
                sample_tilt_deg=float(geom.sample_tilt_deg),
                detector_tilt_deg=float(geom.detector_tilt_deg),
                azimuthal_deg=float(geom.azimuthal_deg),
                twist_deg=float(geom.twist_deg),
                rot_sd=np.eye(3, dtype=np.float64),
                direction_cosines=dc,
                step_x=float(_parse_header_float(header, "xstep", 1.0) or 1.0),
                step_y=float(_parse_header_float(header, "ystep", 1.0) or 1.0),
                scan_unit="um",
                pc_output_convention=ang_pc_convention,
                phase_symmetries=_ang_phase_symmetries(header_lines),
                detector_px_size=None,
                detector_binning=None,
                ang_header_lines=header_lines,
                ang_numeric=None,
                ang_angles_were_degrees=eulers_were_deg,
                up_pattern_reader=up_reader,
            )
            self.initial_eulers_rad = eulers.copy()
            self.current_eulers_rad = self.data.eulers_rad
            self.current_phases = self.data.phases
            self.current_pc_bruker = pc_bruker.copy()
            self.current_pc_custom = pc_custom.copy()
            self._invalidate_orientation_cache()
            self._invalidate_residual_cache()
            self.last_scores_map = np.full((rows, cols), np.nan, dtype=np.float32)
            self.calibration_indices = []
            self.calibrated_center_pc_bruker = None
            self.calibrated_center_pc_custom = None
            self.dictionary_cache = None
            self.dictionary_settings = None
            self.last_indexed_indices = None
            self.indexed_candidate_eulers_rad = None
            self.residual_candidate_eulers_rad = None
            unique_ph = np.unique(phase)
            self.last_action_note = f"Phases in data: {unique_ph.tolist()}"
            return (
                f"Loaded UP+ANG: map={rows}x{cols}, pattern={h}x{w}, N={rows * cols}. "
                "UP patterns are read on demand from disk (bounded batches for map operations). "
                "WARNING: ANG x-star/y-star/z-star are interpreted as OXFORD convention. "
                "PC is only strictly correct for Oxford-acquired patterns or square EDAX patterns."
            )

        raise ValueError(f"Unsupported input file '{p}'. Expected .h5oina or .up1/.up2.")

    # ----------------------- Master loading ------------------------ #

    def load_master(self, master_path: str, energy_kv: float | None = None) -> str:
        import kikuchipy as kp

        if self.data is None:
            raise RuntimeError("Load input data before loading master pattern.")
        p = str(Path(master_path).expanduser().resolve())
        if not Path(p).exists():
            raise FileNotFoundError(p)

        try:
            kwargs = {"projection": "lambert", "hemisphere": "both", "lazy": True}
            mp = kp.load(p, **kwargs)
            energy_vals = _energy_axis_values_kv_from_master_signal(mp)
            selected_energy: float | None = None
            selection_note = ""
            if energy_vals is not None and energy_vals.size > 0:
                if energy_kv is not None and np.isfinite(float(energy_kv)):
                    target = float(energy_kv)
                    idx = int(np.argmin(np.abs(energy_vals - target)))
                    selected_energy = float(energy_vals[idx])
                    selection_note = f"requested {target:.3f} kV"
                elif (
                    self.data.source_type == "h5oina"
                    and self.data.beam_kv is not None
                    and np.isfinite(float(self.data.beam_kv))
                ):
                    target = float(self.data.beam_kv)
                    idx = int(np.argmin(np.abs(energy_vals - target)))
                    selected_energy = float(energy_vals[idx])
                    selection_note = f"nearest to H5OINA beam {target:.3f} kV"
                else:
                    selected_energy = float(np.max(energy_vals))
                    selection_note = "highest available"

            self.master = MasterPatternModel(
                kind="kikuchipy",
                path=p,
                mp_signal=mp,
                projector=None,
                phase=getattr(mp, "phase", None),
                energy_kv=selected_energy,
            )
            self._invalidate_orientation_cache()
            self._invalidate_residual_cache()
            self.indexed_candidate_eulers_rad = None
            if selected_energy is None:
                return (
                    f"Loaded master pattern via Kikuchipy: {Path(p).name}. "
                    "Energy axis unavailable, fallback energy=20 kV will be used."
                )
            return (
                f"Loaded master pattern via Kikuchipy: {Path(p).name}. "
                f"Using energy: {selected_energy:.3f} kV ({selection_note})."
            )
        except Exception:
            hemis = load_master_hemis([p], target_beam_kv=None)
            projector = XProjector(hemis[0], self.data.h, self.data.w)
            self.master = MasterPatternModel(
                kind="legacy",
                path=p,
                mp_signal=None,
                projector=projector,
                phase=None,
                energy_kv=None,
            )
            self._invalidate_orientation_cache()
            self._invalidate_residual_cache()
            self.indexed_candidate_eulers_rad = None
            return (
                f"Loaded master pattern with legacy projector fallback: {Path(p).name}. "
                "Kikuchipy DI/refinement is unavailable for this MP format."
            )

    # ---------------------- Pattern extraction --------------------- #

    def _pattern_at(self, index: int) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        idx = int(index)
        if self.data.source_type == "up_ang" and self.data.up_pattern_reader is not None:
            return self.data.up_pattern_reader.read_pattern(idx)
        row, col = self.row_col_from_index(idx)
        arr = self.data.signal.data[row, col]
        if hasattr(arr, "compute"):
            arr = arr.compute()
        return np.asarray(arr, dtype=np.float32)

    def _apply_dynamic_background_to_patterns(self, patterns: np.ndarray) -> np.ndarray:
        arr = np.asarray(patterns, dtype=np.float32)
        if not self.dynamic_bg_config.enabled:
            return arr
        if arr.ndim == 2:
            work = arr[np.newaxis, ...]
            squeeze = True
        elif arr.ndim == 3:
            work = arr
            squeeze = False
        else:
            raise ValueError(f"Expected 2D or 3D pattern array, got shape {arr.shape}.")

        import kikuchipy as kp

        sig = kp.signals.EBSD(work)
        self._configure_signal_navigation_axis(sig)
        sig = self._apply_dynamic_background_to_signal(sig)
        data = sig.data
        if hasattr(data, "compute"):
            data = data.compute()
        corrected = np.asarray(data, dtype=np.float32)
        return corrected[0] if squeeze else corrected

    def _apply_dynamic_background_to_signal(self, sig):
        if not self.dynamic_bg_config.enabled:
            return sig
        lazy_output = hasattr(sig.data, "chunks")
        return sig.remove_dynamic_background(
            operation="subtract",
            filter_domain="frequency",
            std=_dynamic_bg_std_arg(self.dynamic_bg_config),
            truncate=float(self.dynamic_bg_config.truncate),
            show_progressbar=False,
            inplace=False,
            lazy_output=lazy_output,
        )

    def _configure_signal_navigation_axis(self, sig) -> None:
        if len(sig.axes_manager.navigation_axes) >= 1:
            sig.axes_manager.navigation_axes[0].name = "x"
            sig.axes_manager.navigation_axes[0].scale = 1.0
            sig.axes_manager.navigation_axes[0].units = "px"

    def _materialize_signal_batch(self, sig):
        data = sig.data
        if hasattr(data, "compute"):
            data = data.compute()
        else:
            return sig
        import kikuchipy as kp

        arr = np.asarray(data)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        out = kp.signals.EBSD(arr)
        self._configure_signal_navigation_axis(out)
        return out

    def _processed_pattern_at(self, index: int) -> np.ndarray:
        return self._apply_dynamic_background_to_patterns(self._pattern_at(index))

    def _signal_from_indices(
        self,
        indices: np.ndarray,
        *,
        software_binning: int = 1,
        crop_extent: tuple[int, int, int, int] | None = None,
    ):
        import kikuchipy as kp

        if self.data is None:
            raise RuntimeError("Load input data first.")
        if self.data.source_type == "up_ang" and self.data.up_pattern_reader is not None:
            idx = np.asarray(indices, dtype=np.int64).ravel()
            pats = self.data.up_pattern_reader.read_patterns(idx)
            if pats.ndim == 2:
                pats = pats[np.newaxis, ...]
            sig = kp.signals.EBSD(pats)
            self._configure_signal_navigation_axis(sig)
            sig = self._apply_dynamic_background_to_signal(sig)
            self._configure_signal_navigation_axis(sig)
        else:
            flat = self.data.signal.data.reshape((-1, self.data.h, self.data.w))
            if int(indices.size) == 1:
                sub = flat[int(indices[0])]
            else:
                sub = flat[indices.tolist()]
            if hasattr(sub, "chunks") or hasattr(sub, "compute"):
                sig = kp.signals.LazyEBSD(sub)
            else:
                pats = np.asarray(sub)
                if pats.ndim == 2:
                    pats = pats[np.newaxis, ...]
                sig = kp.signals.EBSD(pats)
            self._configure_signal_navigation_axis(sig)
            sig = self._apply_dynamic_background_to_signal(sig)
            self._configure_signal_navigation_axis(sig)

        sig = self._apply_software_binning_to_signal(
            sig,
            software_binning=software_binning,
            crop_extent=crop_extent,
        )
        if self.dynamic_bg_config.enabled:
            sig = self._materialize_signal_batch(sig)
        return sig

    def _apply_software_binning_to_signal(
        self,
        sig,
        *,
        software_binning: int,
        crop_extent: tuple[int, int, int, int] | None = None,
    ):
        if self.data is None:
            raise RuntimeError("Load input data first.")
        factor = int(software_binning)
        if factor < 1:
            raise ValueError("Software binning must be a positive integer.")
        if crop_extent is None:
            crop_extent = self._binning_crop_extent(factor)
        top, bottom, left, right = crop_extent
        if (top, bottom, left, right) != (0, self.data.h, 0, self.data.w):
            sig.crop_signal(top=top, bottom=bottom, left=left, right=right)
        if factor > 1:
            lazy_output = hasattr(sig.data, "chunks")
            sig = sig.downsample(
                factor=factor,
                dtype_out="float32",
                show_progressbar=False,
                inplace=False,
                lazy_output=lazy_output,
            )
        return sig

    def _load_residual_pattern_source(self):
        """Load and cache the residual pattern source when patterns were written to disk."""
        if self.residual_pattern_output_path is None:
            return None
        if self.data is None:
            raise RuntimeError("Load input data first.")

        path = str(Path(self.residual_pattern_output_path).expanduser().resolve())
        cached = self._residual_pattern_source_cache
        if cached is not None and cached[0] == path:
            return cached[1]

        source: object
        if self.data.source_type == "h5oina":
            import kikuchipy as kp

            source = kp.load(path, lazy=True)
        elif self.data.source_type == "up_ang":
            if self.data.up_pattern_reader is None:
                raise RuntimeError("Residual UP pattern source requires a loaded UP reader.")
            source = UPPatternReader(
                path=path,
                dtype=np.dtype(self.data.up_pattern_reader.dtype),
                pattern_offset=int(self.data.up_pattern_reader.pattern_offset),
                n_patterns=int(self.data.up_pattern_reader.n_patterns),
                h=int(self.data.up_pattern_reader.h),
                w=int(self.data.up_pattern_reader.w),
            )
        else:
            raise ValueError(f"Unsupported residual source type '{self.data.source_type}'.")

        self._residual_pattern_source_cache = (path, source)
        return source

    def _residual_signal_from_indices(
        self,
        indices: np.ndarray,
        *,
        residual_results: dict[int, OverlapPointResult] | None = None,
        apply_dictionary_binning: bool = True,
    ):
        import kikuchipy as kp

        if self.data is None:
            raise RuntimeError("Load input data first.")
        if self.dictionary_cache is None:
            raise RuntimeError("Generate or load a dictionary in tab 2 before indexing residuals.")

        idx = np.asarray(indices, dtype=np.int64).ravel()
        if idx.size == 0:
            raise ValueError("No pattern indices requested.")

        cache = self.dictionary_cache
        residual_results = self.residual_point_results if residual_results is None else residual_results

        can_use_memory = True
        memory_patterns: list[np.ndarray] = []
        for pidx in idx.tolist():
            result = residual_results.get(int(pidx))
            if result is None or result.residual is None:
                can_use_memory = False
                break
            memory_patterns.append(np.asarray(result.residual, dtype=np.float32))

        if can_use_memory:
            patterns = np.stack(memory_patterns, axis=0)
            if patterns.ndim == 2:
                patterns = patterns[np.newaxis, ...]
            sig = kp.signals.EBSD(patterns)
            if len(sig.axes_manager.navigation_axes) >= 1:
                nav = sig.axes_manager.navigation_axes[0]
                nav.name = "x"
                nav.scale = 1.0
                nav.units = "px"
            if not apply_dictionary_binning:
                return sig
            return self._apply_software_binning_to_signal(
                sig,
                software_binning=cache.software_binning,
                crop_extent=cache.crop_extent,
            )

        source = self._load_residual_pattern_source()
        if source is not None:
            if self.data.source_type == "h5oina":
                flat = source.data.reshape((-1, self.data.h, self.data.w))
                sub = flat[idx.tolist()]
                if getattr(sub, "ndim", 0) == 2:
                    sub = sub[np.newaxis, ...]
                sig = kp.signals.LazyEBSD(sub)
            else:
                assert isinstance(source, UPPatternReader)
                patterns = source.read_patterns(idx)
                sig = kp.signals.EBSD(patterns)
            if len(sig.axes_manager.navigation_axes) >= 1:
                nav = sig.axes_manager.navigation_axes[0]
                nav.name = "x"
                nav.scale = 1.0
                nav.units = "px"
            if not apply_dictionary_binning:
                return sig
            return self._apply_software_binning_to_signal(
                sig,
                software_binning=cache.software_binning,
                crop_extent=cache.crop_extent,
            )

        # Fallback: materialize residuals from the primary data source if nothing was cached or written.
        fallback_patterns: list[np.ndarray] = []
        for pidx in idx.tolist():
            result = residual_results.get(int(pidx))
            if result is None:
                result = self.analyze_overlap_point(int(pidx))
                residual_results[int(pidx)] = result
            if result.residual is None:
                result = self._materialize_residual_point_result(result)
                residual_results[int(pidx)] = result
            fallback_patterns.append(np.asarray(result.residual, dtype=np.float32))
        sig = kp.signals.EBSD(np.stack(fallback_patterns, axis=0))
        if len(sig.axes_manager.navigation_axes) >= 1:
            nav = sig.axes_manager.navigation_axes[0]
            nav.name = "x"
            nav.scale = 1.0
            nav.units = "px"
        if not apply_dictionary_binning:
            return sig
        return self._apply_software_binning_to_signal(
            sig,
            software_binning=cache.software_binning,
            crop_extent=cache.crop_extent,
        )

    def _dictionary_index_kikuchipy_signal(
        self,
        signal,
        *,
        cache: DictionaryCache,
        keep_n: int,
        signal_mask: np.ndarray | None,
        n_per_iteration: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xmap = signal.dictionary_indexing(
            dictionary=cache.signal,
            metric="ncc",
            keep_n=max(1, int(keep_n)),
            n_per_iteration=n_per_iteration,
            signal_mask=signal_mask,
            rechunk=False,
        )
        euler_arr = np.asarray(xmap.rotations.to_euler(), dtype=np.float64)
        if euler_arr.ndim == 3:
            candidate_eulers_kp = euler_arr.reshape(euler_arr.shape[0], euler_arr.shape[1], 3)
        elif euler_arr.ndim == 2 and euler_arr.shape[1] == 3:
            candidate_eulers_kp = euler_arr.reshape(euler_arr.shape[0], 1, 3)
        elif euler_arr.ndim == 1 and euler_arr.size == 3:
            candidate_eulers_kp = euler_arr.reshape(1, 1, 3)
        else:
            candidate_eulers_kp = euler_arr.reshape(-1, max(1, int(keep_n)), 3)
        scores_arr = np.asarray(xmap.prop["scores"], dtype=np.float64)
        candidate_scores = scores_arr.reshape(candidate_eulers_kp.shape[0], -1)
        if candidate_scores.shape[1] != candidate_eulers_kp.shape[1]:
            candidate_scores = candidate_scores[:, : candidate_eulers_kp.shape[1]]
        candidate_eulers = self._eulers_from_kikuchipy_frame(candidate_eulers_kp.reshape(-1, 3)).reshape(
            candidate_eulers_kp.shape
        )
        return (
            candidate_eulers[:, 0, :],
            candidate_scores[:, 0],
            candidate_eulers,
            candidate_scores,
        )

    def _dictionary_index_batch_size(self, cache: DictionaryCache, selected_count: int) -> int:
        bytes_per_pattern = max(
            1,
            int(cache.pattern_shape[0] * cache.pattern_shape[1] * np.dtype(np.float32).itemsize),
        )
        by_pattern_memory = int((128 * 1024**2) / bytes_per_pattern)
        return max(
            1,
            min(
                int(selected_count),
                512,
                max(1, by_pattern_memory),
            ),
        )

    def _dictionary_n_per_iteration(
        self,
        cache: DictionaryCache,
        signal_mask: np.ndarray | None,
    ) -> int | None:
        if signal_mask is None:
            return None
        valid_pixels = int(np.count_nonzero(~np.asarray(signal_mask, dtype=bool)))
        if valid_pixels <= 0:
            raise ValueError(f"Pattern mask excludes all pixels for dictionary shape {cache.pattern_shape}.")
        target_bytes = 2 * 1024**3
        n = int(target_bytes / max(1, valid_pixels) / np.dtype(np.float32).itemsize)
        return max(8192, min(int(cache.rotation_count), n))

    def _binning_crop_extent(self, software_binning: int) -> tuple[int, int, int, int]:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        factor = int(software_binning)
        if factor < 1:
            raise ValueError("Software binning must be a positive integer.")
        cropped_h = (self.data.h // factor) * factor
        cropped_w = (self.data.w // factor) * factor
        if cropped_h < factor or cropped_w < factor:
            raise ValueError(
                f"Software binning {factor} is too large for pattern shape {self.data.h}x{self.data.w}."
            )
        top = (self.data.h - cropped_h) // 2
        left = (self.data.w - cropped_w) // 2
        return top, top + cropped_h, left, left + cropped_w

    def _phase_list_for_current_master(self, phase_id: int):
        from orix.crystal_map import PhaseList

        if self.master is None or self.master.phase is None:
            return None
        return PhaseList(phases={int(phase_id): self.master.phase})

    # ----------------------- Refinement step ----------------------- #

    def refine_indices(
        self,
        indices: np.ndarray,
        phase_id: int,
        trust_euler_deg: float = 1.0,
        trust_pc: float = 0.03,
        maxfev: int = 25,
    ) -> str:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")

        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size == 0:
            raise ValueError("No indices selected.")
        valid = self.current_phases[indices] == int(phase_id)
        if not np.any(valid):
            if indices.size == 1:
                actual_phase = int(self.current_phases[int(indices[0])])
                phase_id = actual_phase
                valid = np.array([True], dtype=bool)
                self.last_action_note = (
                    f"Selected point phase ({actual_phase}) differed from requested phase; "
                    f"used phase {actual_phase} for refinement."
                )
            else:
                available = np.unique(self.current_phases[indices]).tolist()
                raise ValueError(
                    f"No selected points match phase ID {phase_id}. "
                    f"Phases present in selection: {available}"
                )
        indices = indices[valid]

        if self.master.kind == "kikuchipy":
            return self._refine_indices_kikuchipy(
                indices=indices,
                phase_id=int(phase_id),
                trust_euler_deg=float(trust_euler_deg),
                trust_pc=float(trust_pc),
                maxfev=int(maxfev),
            )
        return self._refine_indices_legacy(
            indices=indices,
            trust_euler_deg=float(trust_euler_deg),
            trust_pc=float(trust_pc),
            maxfev=int(maxfev),
        )

    def _refine_indices_kikuchipy(
        self,
        indices: np.ndarray,
        phase_id: int,
        trust_euler_deg: float,
        trust_pc: float,
        maxfev: int,
    ) -> str:
        import kikuchipy as kp
        from orix.crystal_map import CrystalMap
        from orix.quaternion import Rotation

        assert self.data is not None
        assert self.master is not None and self.master.mp_signal is not None
        assert self.current_eulers_rad is not None
        assert self.current_pc_bruker is not None

        pc_before = self.current_pc_bruker[indices].copy()
        if indices.size == 1:
            work_indices = np.array([indices[0], indices[0]], dtype=np.int64)
        else:
            work_indices = indices

        sig = self._signal_from_indices(work_indices)
        rot = Rotation.from_euler(self._eulers_to_kikuchipy_frame(self.current_eulers_rad[work_indices]), degrees=False)
        phase_ids = np.full(work_indices.size, int(phase_id), dtype=np.int32)
        phase_list = self._phase_list_for_current_master(phase_id)
        xmap = CrystalMap(
            rotations=rot,
            phase_id=phase_ids,
            x=np.arange(work_indices.size, dtype=np.float64),
            phase_list=phase_list,
            scan_unit="px",
        )
        det = kp.detectors.EBSDDetector(
            shape=(self.data.h, self.data.w),
            pc=self.current_pc_bruker[work_indices],
            convention="bruker",
            sample_tilt=float(self.data.sample_tilt_deg),
            tilt=float(self.data.detector_tilt_deg),
            azimuthal=float(self.data.azimuthal_deg),
            twist=float(self.data.twist_deg),
        )
        sig.detector = det
        signal_mask = self._signal_mask_for_full_pattern()
        trust = [trust_euler_deg, trust_euler_deg, trust_euler_deg, trust_pc, trust_pc, trust_pc]
        xmap_ref, det_ref = sig.refine_orientation_projection_center(
            xmap=xmap,
            detector=det,
            master_pattern=self.master.mp_signal,
            energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
            signal_mask=signal_mask,
            trust_region=trust,
            method="minimize",
            method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=maxfev, disp=False)),
            compute=True,
            rechunk=False,
        )
        metrics_pc, det_pc_ref, _nfev_pc = sig.refine_projection_center(
            xmap=xmap_ref,
            detector=det_ref,
            master_pattern=self.master.mp_signal,
            energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
            signal_mask=signal_mask,
            trust_region=[trust_pc, trust_pc, trust_pc],
            method="minimize",
            method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=maxfev, disp=False)),
            compute=True,
            rechunk=False,
        )
        e_arr = np.asarray(xmap_ref.rotations.to_euler(), dtype=np.float64)
        if e_arr.ndim == 1:
            eulers_new = e_arr.reshape(1, 3)
        elif e_arr.ndim == 2:
            eulers_new = e_arr.reshape(-1, 3)
        else:
            eulers_new = e_arr.reshape(-1, 3)
        pcs_new_bruker = np.asarray(det_pc_ref.pc, dtype=np.float64).reshape(-1, 3)
        eulers_new = eulers_new[: indices.size]
        pcs_new_bruker = pcs_new_bruker[: indices.size]
        self.current_eulers_rad[indices] = self._eulers_from_kikuchipy_frame(eulers_new)
        self.current_pc_bruker[indices] = pcs_new_bruker
        pcs_new_custom = _convert_pc_map(
            pcs_new_bruker.reshape((-1, 1, 3)),
            src_convention="bruker",
            dst_convention=self.data.pc_output_convention,
            shape=(self.data.h, self.data.w),
            sample_tilt_deg=self.data.sample_tilt_deg,
            detector_tilt_deg=self.data.detector_tilt_deg,
            azimuthal_deg=self.data.azimuthal_deg,
            twist_deg=self.data.twist_deg,
        ).reshape(-1, 3)
        self.current_pc_custom[indices] = pcs_new_custom
        self.dictionary_cache = None
        self.dictionary_settings = None
        self._clear_indexed_candidate_rows(indices)
        self._clear_residual_candidate_rows(indices)
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        if self.last_scores_map is not None:
            s_arr = np.asarray(metrics_pc, dtype=np.float64)
            if s_arr.size < work_indices.size:
                s_arr = np.asarray(xmap_ref.prop.get("scores", np.full(work_indices.size, np.nan)), dtype=np.float64)
            scores = s_arr.reshape(-1)
            scores = scores[: indices.size]
            for idx, score in zip(indices.tolist(), scores.tolist()):
                row, col = self.row_col_from_index(idx)
                self.last_scores_map[row, col] = float(score)
        pc_shift = np.linalg.norm(self.current_pc_bruker[indices] - pc_before, axis=1)
        retry_msg = ""
        if float(np.max(pc_shift)) < 1e-6 and indices.size <= 10:
            retry_maxfev = max(200, int(maxfev) * 8)
            metrics_pc_retry, det_pc_retry, _nfev_pc_retry = sig.refine_projection_center(
                xmap=xmap_ref,
                detector=det_pc_ref,
                master_pattern=self.master.mp_signal,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                signal_mask=signal_mask,
                trust_region=[trust_pc, trust_pc, trust_pc],
                method="minimize",
                method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=retry_maxfev, disp=False)),
                compute=True,
                rechunk=False,
            )
            pcs_retry = np.asarray(det_pc_retry.pc, dtype=np.float64).reshape(-1, 3)[: indices.size]
            self.current_pc_bruker[indices] = pcs_retry
            self.current_pc_custom[indices] = _convert_pc_map(
                pcs_retry.reshape((-1, 1, 3)),
                src_convention="bruker",
                dst_convention=self.data.pc_output_convention,
                shape=(self.data.h, self.data.w),
                sample_tilt_deg=self.data.sample_tilt_deg,
                detector_tilt_deg=self.data.detector_tilt_deg,
                azimuthal_deg=self.data.azimuthal_deg,
                twist_deg=self.data.twist_deg,
            ).reshape(-1, 3)
            self.dictionary_cache = None
            self.dictionary_settings = None
            self._invalidate_orientation_cache()
            self._invalidate_residual_cache()
            pc_shift = np.linalg.norm(self.current_pc_bruker[indices] - pc_before, axis=1)
            if self.last_scores_map is not None:
                s_retry = np.asarray(metrics_pc_retry, dtype=np.float64).reshape(-1)
                s_retry = s_retry[: indices.size]
                for idx, score in zip(indices.tolist(), s_retry.tolist()):
                    row, col = self.row_col_from_index(idx)
                    self.last_scores_map[row, col] = float(score)
            retry_msg = f" Auto-retried PC-only with maxfev={retry_maxfev}."
        return (
            f"Refined {indices.size} point(s) with Kikuchipy orientation+PC refinement (Nelder-Mead). "
            f"Mean |dPC|={float(np.mean(pc_shift)):.6f}, max |dPC|={float(np.max(pc_shift)):.6f} (Bruker units)."
            f"{retry_msg}"
        )

    def _refine_indices_legacy(
        self,
        indices: np.ndarray,
        trust_euler_deg: float,
        trust_pc: float,
        maxfev: int,
    ) -> str:
        assert self.data is not None
        assert self.master is not None and self.master.projector is not None
        assert self.current_eulers_rad is not None
        assert self.current_pc_custom is not None

        projector = self.master.projector
        dphi = np.deg2rad(float(trust_euler_deg))
        weights = self._overlap_weights()
        if self.last_scores_map is not None:
            self.last_scores_map[:] = np.nan

        ok = 0
        for idx in indices.tolist():
            exp = self._processed_pattern_at(idx)
            e0 = self.current_eulers_rad[idx].copy()
            p0 = self.current_pc_custom[idx].copy()

            def objective(v: np.ndarray) -> float:
                e = np.array([v[0], v[1], v[2]], dtype=np.float64)
                p = (float(v[3]), float(v[4]), float(v[5]))
                sim = projector.project(e, p, self.data.rot_sd, direction_cosines=self.data.direction_cosines)
                return -_weighted_ncc(exp, sim, weights)

            bounds = [
                (e0[0] - dphi, e0[0] + dphi),
                (e0[1] - dphi, e0[1] + dphi),
                (e0[2] - dphi, e0[2] + dphi),
                (p0[0] - trust_pc, p0[0] + trust_pc),
                (p0[1] - trust_pc, p0[1] + trust_pc),
                (p0[2] - trust_pc, p0[2] + trust_pc),
            ]
            res = minimize(
                objective,
                x0=np.array([e0[0], e0[1], e0[2], p0[0], p0[1], p0[2]], dtype=np.float64),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxfev)},
            )
            e_new = np.array(res.x[:3], dtype=np.float64)
            p_new = np.array(res.x[3:6], dtype=np.float64)
            self.current_eulers_rad[idx] = e_new
            self.current_pc_custom[idx] = p_new
            score = -float(res.fun)
            if self.last_scores_map is not None:
                row, col = self.row_col_from_index(idx)
                self.last_scores_map[row, col] = score
            ok += 1

        # Sync updated custom PCs to Bruker for downstream consistency.
        pcs_b = _convert_pc_map(
            self.current_pc_custom.reshape(self.data.rows, self.data.cols, 3),
            src_convention=self.data.pc_output_convention,
            dst_convention="bruker",
            shape=(self.data.h, self.data.w),
            sample_tilt_deg=self.data.sample_tilt_deg,
            detector_tilt_deg=self.data.detector_tilt_deg,
            azimuthal_deg=self.data.azimuthal_deg,
            twist_deg=self.data.twist_deg,
        ).reshape(-1, 3)
        self.current_pc_bruker[:] = pcs_b
        self.dictionary_cache = None
        self.dictionary_settings = None
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        return f"Refined {ok} point(s) with legacy dynamical projector fallback."

    # --------------------- Point state helpers --------------------- #

    def add_calibration_point(self, index: int) -> str:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        idx = int(index)
        if idx < 0 or idx >= self.data.count:
            raise IndexError(idx)
        if idx not in self.calibration_indices:
            self.calibration_indices.append(idx)
            self.calibration_indices.sort()
        row, col = self.row_col_from_index(idx)
        return f"Added PC calibration point idx={idx} (row={row}, col={col}); total={len(self.calibration_indices)}."

    def remove_calibration_point(self, index: int) -> str:
        idx = int(index)
        if idx in self.calibration_indices:
            self.calibration_indices.remove(idx)
        return f"Removed calibration point idx={idx}; total={len(self.calibration_indices)}."

    def clear_calibration_points(self) -> str:
        self.calibration_indices.clear()
        return "Cleared PC calibration points."

    def calibration_point_summary(self) -> str:
        if not self.calibration_indices:
            return "No calibration points selected."
        labels: list[str] = []
        indices = np.asarray(self.calibration_indices, dtype=np.int64)
        pcs = None
        pc_mean = None
        if self.current_pc_custom is not None:
            pcs = self.current_pc_custom[indices]
            pc_mean = np.mean(pcs, axis=0)
        for i, idx in enumerate(self.calibration_indices):
            row, col = self.row_col_from_index(idx)
            if pcs is None or pc_mean is None:
                labels.append(f"idx={idx} ({row},{col})")
            else:
                deviation = float(np.linalg.norm(pcs[i] - pc_mean))
                labels.append(
                    f"idx={idx} ({row},{col}): PC=({pcs[i, 0]:.6f}, {pcs[i, 1]:.6f}, {pcs[i, 2]:.6f}), "
                    f"|dPC mean|={deviation:.6f}"
                )
        if pcs is None or pc_mean is None:
            return "\n".join(labels)
        ddof = 1 if pcs.shape[0] > 1 else 0
        pc_std = np.std(pcs, axis=0, ddof=ddof)
        pc_range = np.ptp(pcs, axis=0)
        max_deviation = float(np.max(np.linalg.norm(pcs - pc_mean, axis=1)))
        stats = (
            f"Mean PC ({self.data.pc_output_convention if self.data is not None else '-'}): "
            f"({pc_mean[0]:.6f}, {pc_mean[1]:.6f}, {pc_mean[2]:.6f})\n"
            f"PC standard deviation: ({pc_std[0]:.6f}, {pc_std[1]:.6f}, {pc_std[2]:.6f}); "
            f"range: ({pc_range[0]:.6f}, {pc_range[1]:.6f}, {pc_range[2]:.6f}); "
            f"max |dPC mean|={max_deviation:.6f}"
        )
        return "\n".join(labels + [stats])

    def apply_average_calibration_pc(
        self,
        *,
        use_scan_geometry: bool = True,
        detector_px_size: float | None = None,
        detector_binning: float | None = None,
    ) -> str:
        """Average refined calibration PCs and apply a PC map to the scan."""
        import kikuchipy as kp

        if self.data is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Load input data first.")
        if not self.calibration_indices:
            raise ValueError("Select at least one PC calibration point first.")

        indices = np.asarray(self.calibration_indices, dtype=np.int64)
        selected_pc = self.current_pc_bruker[indices].astype(np.float64, copy=True)
        pc_mean = np.mean(selected_pc, axis=0)
        tiled = np.broadcast_to(pc_mean, (self.data.count, 3)).copy()
        pc_map_bruker = tiled.reshape(self.data.rows, self.data.cols, 3)
        correction_note = "scan-position correction disabled"

        if use_scan_geometry:
            px_size = detector_px_size if detector_px_size is not None else self.data.detector_px_size
            binning = detector_binning if detector_binning is not None else self.data.detector_binning
            if px_size is None or not np.isfinite(float(px_size)) or float(px_size) <= 0:
                raise ValueError("A positive unbinned detector pixel size is required for PC extrapolation.")
            if binning is None or not np.isfinite(float(binning)) or float(binning) <= 0:
                raise ValueError("A positive detector binning is required for PC extrapolation.")
            detector = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=selected_pc,
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            point_rc = np.asarray([self.row_col_from_index(int(idx)) for idx in indices], dtype=np.float64)
            extrapolated = detector.extrapolate_pc(
                pc_indices=point_rc,
                navigation_shape=(self.data.rows, self.data.cols),
                step_sizes=(float(self.data.step_y), float(self.data.step_x)),
                px_size=float(px_size),
                binning=float(binning),
            )
            pc_map_bruker = np.asarray(extrapolated.pc, dtype=np.float64).reshape(self.data.rows, self.data.cols, 3)
            max_correction = float(np.max(np.linalg.norm(pc_map_bruker.reshape(-1, 3) - tiled, axis=1)))
            correction_note = f"Kikuchipy scan-position correction, max |dPC|={max_correction:.6f}"

        self.current_pc_bruker[:] = pc_map_bruker.reshape(-1, 3)
        pc_custom = _convert_pc_map(
            pc_map_bruker,
            src_convention="bruker",
            dst_convention=self.data.pc_output_convention,
            shape=(self.data.h, self.data.w),
            sample_tilt_deg=self.data.sample_tilt_deg,
            detector_tilt_deg=self.data.detector_tilt_deg,
            azimuthal_deg=self.data.azimuthal_deg,
            twist_deg=self.data.twist_deg,
        )
        self.current_pc_custom[:] = pc_custom.reshape(-1, 3)
        center_row = int(round((self.data.rows - 1) / 2.0))
        center_col = int(round((self.data.cols - 1) / 2.0))
        center_idx = self.index_from_row_col(center_row, center_col)
        self.calibrated_center_pc_bruker = self.current_pc_bruker[center_idx].copy()
        self.calibrated_center_pc_custom = self.current_pc_custom[center_idx].copy()
        self.dictionary_cache = None
        self.dictionary_settings = None
        self._clear_indexed_candidate_rows()
        self._clear_residual_candidate_rows()
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        center = self.calibrated_center_pc_custom
        span_x = abs(float(self.data.step_x)) * max(0, self.data.cols - 1)
        span_y = abs(float(self.data.step_y)) * max(0, self.data.rows - 1)
        return (
            f"Applied average of {indices.size} refined PC(s); center PC ({self.data.pc_output_convention})="
            f"({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}). {correction_note}. "
            f"Scan span={span_x:.3f} x {span_y:.3f} {self.data.scan_unit}."
        )

    def apply_point_pc_to_full_map(self, index: int) -> str:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        if self.current_pc_custom is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(index)
        row, col = self.row_col_from_index(idx)
        pc_b = self.current_pc_bruker[idx].astype(np.float64, copy=True)
        pc_c = self.current_pc_custom[idx].astype(np.float64, copy=True)
        before = self.current_pc_bruker.copy()
        self.current_pc_bruker[:] = pc_b
        self.current_pc_custom[:] = pc_c
        self.dictionary_cache = None
        self.dictionary_settings = None
        self._clear_indexed_candidate_rows()
        self._clear_residual_candidate_rows()
        dpc = np.linalg.norm(self.current_pc_bruker - before, axis=1)
        return (
            f"Applied selected PC from idx={idx} (row={row}, col={col}) to full map. "
            f"Mean |dPC|={float(np.mean(dpc)):.6f}, max |dPC|={float(np.max(dpc)):.6f} (Bruker units)."
        )

    def get_point_state(self, index: int) -> dict[str, object]:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        if self.current_eulers_rad is None or self.current_pc_custom is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(index)
        row, col = self.row_col_from_index(idx)
        e_rad = self.current_eulers_rad[idx].astype(np.float64, copy=True)
        e_deg = np.rad2deg(e_rad)
        pc_custom = self.current_pc_custom[idx].astype(np.float64, copy=True)
        pc_bruker = self.current_pc_bruker[idx].astype(np.float64, copy=True)
        return {
            "index": idx,
            "row": row,
            "col": col,
            "phase": int(self.current_phases[idx]) if self.current_phases is not None else None,
            "euler_rad": e_rad,
            "euler_deg": e_deg,
            "pc_custom": pc_custom,
            "pc_bruker": pc_bruker,
            "pc_convention": self.data.pc_output_convention,
        }

    def set_point_state(
        self,
        index: int,
        *,
        euler_deg: tuple[float, float, float] | None = None,
        pc_custom: tuple[float, float, float] | None = None,
    ) -> str:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        if self.current_eulers_rad is None or self.current_pc_custom is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(index)
        notes: list[str] = []
        if euler_deg is not None:
            e_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64).reshape(3))
            self.current_eulers_rad[idx] = e_rad
            notes.append("Euler updated")
        if pc_custom is not None:
            pc_c = np.asarray(pc_custom, dtype=np.float64).reshape(3)
            self.current_pc_custom[idx] = pc_c
            pc_b = _convert_pc_map(
                pc_c.reshape(1, 1, 3),
                src_convention=self.data.pc_output_convention,
                dst_convention="bruker",
                shape=(self.data.h, self.data.w),
                sample_tilt_deg=self.data.sample_tilt_deg,
                detector_tilt_deg=self.data.detector_tilt_deg,
                azimuthal_deg=self.data.azimuthal_deg,
                twist_deg=self.data.twist_deg,
            ).reshape(3)
            self.current_pc_bruker[idx] = pc_b
            self.dictionary_cache = None
            self.dictionary_settings = None
            notes.append(f"PC updated ({self.data.pc_output_convention})")
        if not notes:
            return "No changes applied."
        self._clear_indexed_candidate_rows(np.asarray([idx], dtype=np.int64))
        self._clear_residual_candidate_rows(np.asarray([idx], dtype=np.int64))
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        row, col = self.row_col_from_index(idx)
        return f"Updated point idx={idx} (row={row}, col={col}): {', '.join(notes)}."

    # ---------------------- Dictionary indexing -------------------- #

    def generate_dictionary(
        self,
        *,
        phase_id: int,
        resolution_deg: float = 12.0,
        software_binning: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        """Explicitly generate and retain a Kikuchipy pattern dictionary."""
        if self.data is None or self.master is None:
            raise RuntimeError("Load input data and a master pattern first.")
        if self.master.kind != "kikuchipy":
            raise RuntimeError("Explicit dictionary generation requires a Kikuchipy master pattern.")
        if self.current_pc_bruker is None:
            raise RuntimeError("Session PC state is not initialized.")
        factor = int(software_binning)
        if factor < 1 or not np.isclose(float(software_binning), factor):
            raise ValueError("Software binning must be a positive integer.")
        center_row = int(round((self.data.rows - 1) / 2.0))
        center_col = int(round((self.data.cols - 1) / 2.0))
        center_idx = self.index_from_row_col(center_row, center_col)
        pc = (
            self.calibrated_center_pc_bruker.copy()
            if self.calibrated_center_pc_bruker is not None
            else self.current_pc_bruker[center_idx].copy()
        )
        cache = self._get_or_build_kikuchipy_dictionary(
            phase_id=int(phase_id),
            resolution_deg=float(resolution_deg),
            pc_bruker=pc,
            software_binning=factor,
            progress_callback=progress_callback,
        )
        estimated_mb = (
            cache.rotation_count
            * cache.pattern_shape[0]
            * cache.pattern_shape[1]
            * np.dtype(np.float32).itemsize
            / (1024**2)
        )
        return (
            f"Generated Kikuchipy dictionary: {cache.rotation_count} orientations, "
            f"pattern shape={cache.pattern_shape[0]}x{cache.pattern_shape[1]}, "
            f"software binning={cache.software_binning}, estimated data={estimated_mb:.1f} MB."
        )

    def dictionary_index_indices(
        self,
        indices: np.ndarray,
        phase_id: int,
        keep_n: int = 1,
        resolution_deg: float = 12.0,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")

        keep_n = max(1, int(keep_n))

        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size == 0:
            raise ValueError("No indices selected.")
        valid = self.current_phases[indices] == int(phase_id)
        if not np.any(valid):
            if indices.size == 1:
                actual_phase = int(self.current_phases[int(indices[0])])
                phase_id = actual_phase
                valid = np.array([True], dtype=bool)
                self.last_action_note = (
                    f"Selected point phase ({actual_phase}) differed from requested phase; "
                    f"used phase {actual_phase} for indexing."
                )
            else:
                available = np.unique(self.current_phases[indices]).tolist()
                raise ValueError(
                    f"No selected points match phase ID {phase_id}. "
                    f"Phases present in selection: {available}"
                )
        indices = indices[valid]
        if progress_callback is not None:
            progress_callback(0.0, f"Preparing to re-index {indices.size} point(s)...")

        if self.master.kind == "kikuchipy":
            return self._dictionary_index_kikuchipy(
                indices,
                phase_id=phase_id,
                keep_n=keep_n,
                progress_callback=progress_callback,
            )
        return self._dictionary_index_legacy(
            indices,
            keep_n=keep_n,
            resolution_deg=resolution_deg,
            progress_callback=progress_callback,
        )

    def _dictionary_index_kikuchipy(
        self,
        indices: np.ndarray,
        phase_id: int,
        keep_n: int,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        import kikuchipy as kp

        assert self.data is not None
        assert self.master is not None and self.master.mp_signal is not None and self.master.phase is not None
        assert self.current_eulers_rad is not None
        assert self.current_pc_bruker is not None

        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate the dictionary in tab 2 before indexing patterns.")
        if cache.phase_id != int(phase_id):
            raise ValueError(
                f"The generated dictionary is for phase {cache.phase_id}, but phase {phase_id} was requested. "
                "Generate a dictionary for the requested phase."
            )

        keep_n = max(1, int(keep_n))
        if keep_n <= 1:
            self._reset_indexed_candidate_cache(None)
        elif self.indexed_candidate_eulers_rad is None or (
            self.indexed_candidate_eulers_rad is not None
            and self.indexed_candidate_eulers_rad.shape != (self.data.count, keep_n, 3)
        ):
            self._reset_indexed_candidate_cache(keep_n)
        signal_mask = self._signal_mask_for_dictionary_cache(cache)
        n_per_iteration = self._dictionary_n_per_iteration(cache, signal_mask)
        batch_size = self._dictionary_index_batch_size(cache, int(indices.size))
        total_batches = int(np.ceil(indices.size / batch_size))

        for batch_number, start in enumerate(range(0, indices.size, batch_size), start=1):
            batch_indices = indices[start : start + batch_size]
            if batch_indices.size == 1:
                work_indices = np.array([batch_indices[0], batch_indices[0]], dtype=np.int64)
            else:
                work_indices = batch_indices
            if progress_callback is not None:
                label = "BG-corrected " if self.dynamic_bg_config.enabled else ""
                progress_callback(
                    100.0 * start / indices.size,
                    f"Preparing {label}batch {batch_number}/{total_batches} ({batch_indices.size} point(s))...",
                )
            sig = self._signal_from_indices(
                work_indices,
                software_binning=cache.software_binning,
                crop_extent=cache.crop_extent,
            )
            if progress_callback is not None:
                progress_callback(
                    100.0 * start / indices.size,
                    (
                        f"Indexing batch {batch_number}/{total_batches} ({batch_indices.size} point(s)) "
                        f"against {cache.rotation_count} dictionary patterns"
                        f"{'' if n_per_iteration is None else f' ({n_per_iteration} dictionary patterns/iteration)'}..."
                    ),
                )
            eulers_new, scores_arr, candidate_eulers, _candidate_scores = self._dictionary_index_kikuchipy_signal(
                sig,
                cache=cache,
                keep_n=keep_n,
                signal_mask=signal_mask,
                n_per_iteration=n_per_iteration,
            )
            eulers_new = eulers_new[: batch_indices.size]
            scores = scores_arr[: batch_indices.size]
            if self.indexed_candidate_eulers_rad is not None:
                self.indexed_candidate_eulers_rad[batch_indices] = candidate_eulers[: batch_indices.size]
            self.current_eulers_rad[batch_indices] = self._eulers_from_kikuchipy_frame(eulers_new)
            if self.last_scores_map is not None:
                for idx, score in zip(batch_indices.tolist(), scores.tolist()):
                    row, col = self.row_col_from_index(idx)
                    self.last_scores_map[row, col] = float(score)
            if progress_callback is not None:
                completed = min(indices.size, start + batch_indices.size)
                progress_callback(
                    100.0 * completed / indices.size,
                    f"Re-indexed {completed}/{indices.size} point(s)...",
                )

        self.last_indexed_indices = indices.copy()
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        return (
            f"Dictionary indexed {indices.size} point(s) with Kikuchipy NCC "
            f"(keep_n={keep_n}, dictionary size={cache.rotation_count}, batch_size={batch_size})."
        )

    def _get_or_build_kikuchipy_dictionary(
        self,
        *,
        phase_id: int,
        resolution_deg: float,
        pc_bruker: np.ndarray,
        software_binning: int,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> DictionaryCache:
        from orix import sampling
        from orix.crystal_map import CrystalMap
        import kikuchipy as kp

        if self.data is None or self.master is None or self.master.mp_signal is None or self.master.phase is None:
            raise RuntimeError("A Kikuchipy master pattern is required to build a dictionary.")
        pc = np.asarray(pc_bruker, dtype=np.float64).reshape(3)
        factor = int(software_binning)
        crop_extent = self._binning_crop_extent(factor)
        top, bottom, left, right = crop_extent
        pattern_shape = ((bottom - top) // factor, (right - left) // factor)
        existing = self.dictionary_cache
        if (
            existing is not None
            and existing.phase_id == int(phase_id)
            and np.isclose(existing.resolution_deg, float(resolution_deg))
            and np.allclose(existing.pc_bruker, pc, atol=1e-10, rtol=0.0)
            and existing.software_binning == factor
            and existing.crop_extent == crop_extent
        ):
            if progress_callback is not None:
                progress_callback(100.0, "Dictionary already matches these settings.")
            return existing

        if progress_callback is not None:
            progress_callback(5.0, "Sampling dictionary orientations...")
        rots = sampling.get_sample_fundamental(
            resolution=float(resolution_deg),
            point_group=self.master.phase.point_group,
        )
        if rots.size == 0:
            raise RuntimeError("No dictionary rotations were generated.")
        if progress_callback is not None:
            progress_callback(10.0, f"Preparing binned detector for {rots.size} orientations...")
        detector_full = kp.detectors.EBSDDetector(
            shape=(self.data.h, self.data.w),
            pc=pc,
            convention="bruker",
            px_size=float(self.data.detector_px_size or 1.0),
            binning=int(round(float(self.data.detector_binning or 1.0))),
            sample_tilt=float(self.data.sample_tilt_deg),
            tilt=float(self.data.detector_tilt_deg),
            azimuthal=float(self.data.azimuthal_deg),
            twist=float(self.data.twist_deg),
        )
        detector_cropped = detector_full.crop(crop_extent)
        detector = kp.detectors.EBSDDetector(
            shape=pattern_shape,
            pc=np.asarray(detector_cropped.pc, dtype=np.float64).reshape(-1, 3),
            convention="bruker",
            px_size=float(self.data.detector_px_size or 1.0),
            binning=int(round(float(self.data.detector_binning or 1.0))) * factor,
            sample_tilt=float(self.data.sample_tilt_deg),
            tilt=float(self.data.detector_tilt_deg),
            azimuthal=float(self.data.azimuthal_deg),
            twist=float(self.data.twist_deg),
        )
        bytes_per_pattern = max(1, pattern_shape[0] * pattern_shape[1] * np.dtype(np.float32).itemsize)
        chunk_size = max(16, min(64, int((128 * 1024**2) / bytes_per_pattern)))
        dictionary_data = np.empty((int(rots.size), *pattern_shape), dtype=np.float32)
        energy = float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0
        for start in range(0, int(rots.size), chunk_size):
            stop = min(int(rots.size), start + chunk_size)
            chunk_signal = self.master.mp_signal.get_patterns(
                rotations=rots[start:stop],
                detector=detector,
                energy=energy,
                compute=True,
                show_progressbar=False,
            )
            dictionary_data[start:stop] = np.asarray(chunk_signal.data, dtype=np.float32).reshape(
                stop - start,
                *pattern_shape,
            )
            if progress_callback is not None:
                progress_callback(
                    10.0 + 85.0 * stop / int(rots.size),
                    f"Generated {stop}/{int(rots.size)} dictionary patterns...",
                )
        dictionary = kp.signals.EBSD(dictionary_data)
        if len(dictionary.axes_manager.navigation_axes) >= 1:
            dictionary.axes_manager.navigation_axes[0].name = "x"
            dictionary.axes_manager.navigation_axes[0].scale = 1.0
            dictionary.axes_manager.navigation_axes[0].units = "px"
        dictionary.xmap = CrystalMap(
            rotations=rots,
            phase_id=np.full(rots.size, int(phase_id), dtype=np.int32),
            phase_list=self._phase_list_for_current_master(int(phase_id)),
        )
        cache = DictionaryCache(
            phase_id=int(phase_id),
            resolution_deg=float(resolution_deg),
            pc_bruker=pc.copy(),
            software_binning=factor,
            crop_extent=crop_extent,
            pattern_shape=pattern_shape,
            signal=dictionary,
            rotation_count=int(rots.size),
        )
        self.dictionary_cache = cache
        self.dictionary_settings = {
            "phase_id": int(phase_id),
            "resolution_deg": float(resolution_deg),
            "pc_bruker": pc.copy(),
            "software_binning": factor,
            "crop_extent": np.asarray(crop_extent, dtype=np.int64),
        }
        if progress_callback is not None:
            progress_callback(100.0, "Dictionary generation complete.")
        return cache

    def _dictionary_index_legacy(
        self,
        indices: np.ndarray,
        keep_n: int,
        resolution_deg: float,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        from orix import sampling
        from orix.quaternion.symmetry import Oh

        assert self.data is not None
        assert self.master is not None and self.master.projector is not None
        assert self.current_pc_custom is not None
        assert self.current_eulers_rad is not None

        rots = sampling.get_sample_fundamental(resolution=float(resolution_deg), point_group=Oh)
        euler_dict = np.asarray(rots.to_euler(), dtype=np.float64).reshape(-1, 3)
        projector = self.master.projector
        weights = self._overlap_weights()
        sqrt_weights = np.sqrt(weights.ravel()).astype(np.float32, copy=False)

        pc_mean = np.mean(self.current_pc_custom[indices], axis=0)
        dict_vecs = []
        for e in euler_dict:
            sim = projector.project(
                e,
                (float(pc_mean[0]), float(pc_mean[1]), float(pc_mean[2])),
                self.data.rot_sd,
                direction_cosines=self.data.direction_cosines,
            )
            dict_vecs.append(_normalize_weighted(sim, weights).ravel() * sqrt_weights)
        dict_mat = np.asarray(dict_vecs, dtype=np.float32)
        dict_mat /= (np.linalg.norm(dict_mat, axis=1, keepdims=True) + 1e-8)

        k = max(1, int(keep_n))
        for point_number, idx in enumerate(indices.tolist(), start=1):
            exp = _normalize_weighted(self._processed_pattern_at(idx), weights).ravel().astype(np.float32) * sqrt_weights
            exp /= float(np.linalg.norm(exp) + 1e-8)
            scores = dict_mat @ exp
            top = np.argpartition(-scores, kth=min(k - 1, scores.size - 1))[:k]
            best = int(top[np.argmax(scores[top])])
            self.current_eulers_rad[idx] = euler_dict[best]
            if self.last_scores_map is not None:
                r, c = self.row_col_from_index(idx)
                self.last_scores_map[r, c] = float(scores[best])
            if progress_callback is not None:
                progress_callback(
                    100.0 * point_number / indices.size,
                    f"Re-indexed {point_number}/{indices.size} point(s)...",
                )
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        self.last_indexed_indices = indices.copy()

        return (
            f"Dictionary indexed {indices.size} point(s) with legacy NCC fallback "
            f"(keep_n={k}, dictionary size={euler_dict.shape[0]})."
        )

    def refine_orientations_indices(
        self,
        indices: np.ndarray,
        *,
        phase_id: int,
        trust_euler_deg: float = 1.0,
        maxfev: int = 50,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        """Run Kikuchipy orientation-only refinement after dictionary indexing."""
        import kikuchipy as kp
        from orix.crystal_map import CrystalMap
        from orix.quaternion import Rotation

        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Kikuchipy input and master pattern are required for orientation refinement.")
        if self.master.mp_signal is None or self.current_eulers_rad is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")
        selected = np.asarray(indices, dtype=np.int64).ravel()
        if selected.size == 0:
            raise ValueError("No points selected for orientation refinement.")
        valid = self.current_phases[selected] == int(phase_id)
        if not np.any(valid) and selected.size == 1:
            phase_id = int(self.current_phases[selected[0]])
            valid[:] = True
        selected = selected[valid]
        if selected.size == 0:
            raise ValueError(f"No selected points match phase ID {phase_id}.")

        candidate_store = self.indexed_candidate_eulers_rad
        candidate_count = int(candidate_store.shape[1]) if candidate_store is not None else 1
        use_candidates = candidate_store is not None and candidate_count > 1
        if progress_callback is not None:
            progress_callback(0.0, f"Preparing to refine {selected.size} orientation(s)...")

        signal_mask = self._signal_mask_for_full_pattern()
        batch_size = min(256, max(1, int(selected.size)))
        if use_candidates:
            batch_size = max(1, min(batch_size, max(1, 256 // candidate_count)))
        for start in range(0, selected.size, batch_size):
            batch_indices = selected[start : start + batch_size]
            if use_candidates:
                candidate_batch = np.asarray(candidate_store[batch_indices], dtype=np.float64).reshape(
                    batch_indices.size, candidate_count, 3
                )
                fallback = np.repeat(self.current_eulers_rad[batch_indices][:, np.newaxis, :], candidate_count, axis=1)
                valid_candidates = np.all(np.isfinite(candidate_batch), axis=2)
                if not np.all(valid_candidates):
                    candidate_batch = np.where(valid_candidates[..., np.newaxis], candidate_batch, fallback)
                work_indices = np.repeat(batch_indices, candidate_count)
                signal = self._signal_from_indices(work_indices)
                initial = candidate_batch.reshape(-1, 3)
                xmap = CrystalMap(
                    rotations=Rotation.from_euler(self._eulers_to_kikuchipy_frame(initial), degrees=False),
                    phase_id=np.full(initial.shape[0], int(phase_id), dtype=np.int32),
                    x=np.arange(initial.shape[0], dtype=np.float64),
                    phase_list=self._phase_list_for_current_master(int(phase_id)),
                    scan_unit="px",
                )
                detector = kp.detectors.EBSDDetector(
                    shape=(self.data.h, self.data.w),
                    pc=np.repeat(self.current_pc_bruker[batch_indices], candidate_count, axis=0),
                    convention="bruker",
                    sample_tilt=float(self.data.sample_tilt_deg),
                    tilt=float(self.data.detector_tilt_deg),
                    azimuthal=float(self.data.azimuthal_deg),
                    twist=float(self.data.twist_deg),
                )
                refined = signal.refine_orientation(
                    xmap=xmap,
                    detector=detector,
                    master_pattern=self.master.mp_signal,
                    energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                    signal_mask=signal_mask,
                    trust_region=[float(trust_euler_deg)] * 3,
                    method="minimize",
                    method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=int(maxfev), disp=False)),
                    compute=True,
                    rechunk=False,
                )
                refined_eulers = np.asarray(refined.rotations.to_euler(), dtype=np.float64).reshape(
                    batch_indices.size, candidate_count, 3
                )
                scores = np.asarray(
                    refined.prop.get("scores", np.full(refined_eulers.shape[0] * candidate_count, np.nan)),
                    dtype=np.float64,
                ).reshape(batch_indices.size, candidate_count)
                best = np.argmax(scores, axis=1)
                best_rows = np.arange(batch_indices.size)
                eulers = refined_eulers[best_rows, best]
                best_scores = scores[best_rows, best]
                self.current_eulers_rad[batch_indices] = self._eulers_from_kikuchipy_frame(eulers)
                if self.last_scores_map is not None:
                    for idx, score in zip(batch_indices.tolist(), best_scores.tolist()):
                        row, col = self.row_col_from_index(idx)
                        self.last_scores_map[row, col] = float(score)
                self._clear_indexed_candidate_rows(batch_indices)
            else:
                work_indices = (
                    np.array([batch_indices[0], batch_indices[0]], dtype=np.int64)
                    if batch_indices.size == 1
                    else batch_indices
                )
                signal = self._signal_from_indices(work_indices)
                xmap = CrystalMap(
                    rotations=Rotation.from_euler(
                        self._eulers_to_kikuchipy_frame(self.current_eulers_rad[work_indices]),
                        degrees=False,
                    ),
                    phase_id=np.full(work_indices.size, int(phase_id), dtype=np.int32),
                    x=np.arange(work_indices.size, dtype=np.float64),
                    phase_list=self._phase_list_for_current_master(int(phase_id)),
                    scan_unit="px",
                )
                detector = kp.detectors.EBSDDetector(
                    shape=(self.data.h, self.data.w),
                    pc=self.current_pc_bruker[work_indices],
                    convention="bruker",
                    sample_tilt=float(self.data.sample_tilt_deg),
                    tilt=float(self.data.detector_tilt_deg),
                    azimuthal=float(self.data.azimuthal_deg),
                    twist=float(self.data.twist_deg),
                )
                refined = signal.refine_orientation(
                    xmap=xmap,
                    detector=detector,
                    master_pattern=self.master.mp_signal,
                    energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                    signal_mask=signal_mask,
                    trust_region=[float(trust_euler_deg)] * 3,
                    method="minimize",
                    method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=int(maxfev), disp=False)),
                    compute=True,
                    rechunk=False,
                )
                eulers = np.asarray(refined.rotations.to_euler(), dtype=np.float64).reshape(-1, 3)[: batch_indices.size]
                self.current_eulers_rad[batch_indices] = self._eulers_from_kikuchipy_frame(eulers)
                scores = np.asarray(
                    refined.prop.get("scores", np.full(work_indices.size, np.nan)),
                    dtype=np.float64,
                ).reshape(-1)
                if self.last_scores_map is not None:
                    for idx, score in zip(batch_indices.tolist(), scores[: batch_indices.size].tolist()):
                        row, col = self.row_col_from_index(idx)
                        self.last_scores_map[row, col] = float(score)
            if progress_callback is not None:
                completed = min(selected.size, start + batch_indices.size)
                progress_callback(
                    100.0 * completed / selected.size,
                    f"Refined {completed}/{selected.size} orientation(s)...",
                )
        self._invalidate_orientation_cache()
        self._invalidate_residual_cache()
        return (
            f"Refined {selected.size} indexed orientation(s) with Kikuchipy (Nelder-Mead, "
            f"batch_size={batch_size})."
        )

    # ----------------------- Overlap analysis ---------------------- #

    def _overlap_weights(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        return _weights_from_pattern_mask((self.data.h, self.data.w), self.pattern_mask_config)

    def _fit_primary_pattern(
        self,
        experimental_raw: np.ndarray,
        simulated_raw: np.ndarray,
        *,
        maxiter: int,
        popsize: int,
        seed: int,
        fit_bounds: list[tuple[float, float]] | None = None,
    ) -> PrimaryPatternFit:
        """Fit the paper's Gaussian blur and elliptical power-law gain model."""
        return _fit_overlap_primary_pattern(
            experimental_raw,
            simulated_raw,
            self._overlap_weights(),
            maxiter=int(maxiter),
            popsize=int(popsize),
            seed=int(seed),
            fit_bounds=fit_bounds,
        )

    def _kikuchipy_pattern_ncc(self, experimental: np.ndarray, simulated: np.ndarray) -> float:
        """Score one pattern pair with Kikuchipy's NCC metric and the active signal mask."""
        import kikuchipy as kp

        if self.data is None:
            raise RuntimeError("Load input data first.")
        signal_mask = self._signal_mask_for_full_pattern()
        metric = kp.indexing.NormalizedCrossCorrelationMetric(
            n_experimental_patterns=1,
            n_dictionary_patterns=1,
            signal_mask=signal_mask,
            dtype=np.float32,
            rechunk=False,
        )
        exp_prepared = metric.prepare_experimental(
            np.asarray(experimental, dtype=np.float32).reshape(1, self.data.h, self.data.w).copy()
        )
        sim_prepared = metric.prepare_dictionary(
            np.asarray(simulated, dtype=np.float32).reshape(1, -1).copy()
        )
        score = metric.match(exp_prepared, sim_prepared)
        if hasattr(score, "compute"):
            score = score.compute()
        return float(np.asarray(score, dtype=np.float64).reshape(-1)[0])

    def _pattern_ncc_for_overlap(self, experimental: np.ndarray, simulated: np.ndarray) -> float:
        if self.master is not None and self.master.kind == "kikuchipy":
            return self._kikuchipy_pattern_ncc(experimental, simulated)
        return _weighted_ncc(experimental, simulated, self._overlap_weights())

    def _normalize_pattern_for_overlap(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize over the same pixels used by active pattern-mask scoring."""
        if self.data is None:
            return normalize_zmuv(np.asarray(pattern, dtype=np.float32))
        weights = self._overlap_weights()
        return _zero_unweighted_pixels(_normalize_weighted(pattern, weights), weights)

    def _simulate_pattern_for_euler(self, index: int, euler_rad: np.ndarray) -> np.ndarray:
        if self.data is None or self.master is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Load input data and a master pattern first.")
        idx = int(index)
        euler = np.asarray(euler_rad, dtype=np.float64).reshape(3)
        if self.master.kind == "kikuchipy" and self.master.mp_signal is not None:
            import kikuchipy as kp
            from orix.quaternion import Rotation

            rotation = Rotation.from_euler(self._eulers_to_kikuchipy_frame(euler.reshape(1, 3)), degrees=False)
            detector = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=self.current_pc_bruker[idx],
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            signal = self.master.mp_signal.get_patterns(
                rotations=rotation,
                detector=detector,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                compute=True,
                show_progressbar=False,
            )
            return np.asarray(signal.data[0], dtype=np.float32)
        if self.master.projector is None:
            raise RuntimeError("No pattern projector is available.")
        pc = self.current_pc_custom[idx]
        return self.master.projector.project(
            euler,
            (float(pc[0]), float(pc[1]), float(pc[2])),
            self.data.rot_sd,
            direction_cosines=self.data.direction_cosines,
        )

    def analyze_overlap_point(
        self,
        index: int,
        blur_sigma: float = 0.0,
        *,
        fit_blur_gain: bool = True,
        fit_maxiter: int = 40,
        fit_popsize: int = 8,
        fit_bounds: list[tuple[float, float]] | None = None,
        store_result: bool = True,
    ) -> OverlapPointResult:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(index)
        row, col = self.row_col_from_index(idx)
        experimental_raw = self._processed_pattern_at(idx)
        simulated_raw = self._simulate_pattern_for_euler(idx, self.current_eulers_rad[idx])
        weights = self._overlap_weights()
        result = _overlap_point_result_from_raw_patterns(
            idx,
            row,
            col,
            experimental_raw,
            simulated_raw,
            weights,
            fit_blur_gain=bool(fit_blur_gain),
            fit_maxiter=int(fit_maxiter),
            fit_popsize=int(fit_popsize),
            fit_bounds=fit_bounds,
            blur_sigma=float(blur_sigma),
        )
        if store_result:
            self._clear_residual_candidate_rows(np.asarray([idx], dtype=np.int64))
            self.residual_point_results[int(idx)] = result
        self._ensure_residual_state()
        self._invalidate_residual_color_cache()
        return result

    def index_overlap_residual(
        self,
        index: int,
        *,
        blur_sigma: float = 0.0,
        keep_n: int = 1,
        residual_result: OverlapPointResult | None = None,
    ) -> OverlapPointResult:
        """Index one NCC-scaled residual with the dictionary generated in step 2."""
        import kikuchipy as kp

        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Residual dictionary indexing requires a Kikuchipy master pattern.")
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate or load a Kikuchipy dictionary in tab 2 before indexing a residual.")

        result = residual_result
        if result is None:
            result = self.analyze_overlap_point(int(index), blur_sigma=float(blur_sigma))
        if result.index != int(index):
            raise ValueError("The supplied residual result belongs to a different map point.")
        keep_n = max(1, int(keep_n))
        if keep_n <= 1:
            self._reset_residual_candidate_cache(None)
        elif self.residual_candidate_eulers_rad is None or (
            self.residual_candidate_eulers_rad is not None
            and self.residual_candidate_eulers_rad.shape != (self.data.count, keep_n, 3)
        ):
            self._reset_residual_candidate_cache(keep_n)
        self._invalidate_overlap_mixture_cache()
        if result.residual is None or result.simulated is None:
            result = self._materialize_residual_point_result(result)
        work = np.stack((result.residual, result.residual), axis=0).astype(np.float32, copy=False)
        signal = kp.signals.EBSD(work)
        signal = self._apply_software_binning_to_signal(
            signal,
            software_binning=cache.software_binning,
            crop_extent=cache.crop_extent,
        )
        signal_mask = self._signal_mask_for_dictionary_cache(cache)
        xmap = signal.dictionary_indexing(
            dictionary=cache.signal,
            metric="ncc",
            keep_n=keep_n,
            signal_mask=signal_mask,
            rechunk=False,
        )
        euler_arr = np.asarray(xmap.rotations.to_euler(), dtype=np.float64)
        if euler_arr.ndim == 3:
            candidate_eulers_kp = euler_arr.reshape(euler_arr.shape[0], euler_arr.shape[1], 3)
        elif euler_arr.ndim == 2 and euler_arr.shape[1] == 3:
            candidate_eulers_kp = euler_arr.reshape(euler_arr.shape[0], 1, 3)
        elif euler_arr.ndim == 1 and euler_arr.size == 3:
            candidate_eulers_kp = euler_arr.reshape(1, 1, 3)
        else:
            candidate_eulers_kp = euler_arr.reshape(-1, keep_n, 3)
        scores = np.asarray(xmap.prop["scores"], dtype=np.float64).reshape(candidate_eulers_kp.shape[0], -1)
        if scores.shape[1] != candidate_eulers_kp.shape[1]:
            scores = scores[:, : candidate_eulers_kp.shape[1]]
        candidate_eulers = self._eulers_from_kikuchipy_frame(candidate_eulers_kp.reshape(-1, 3)).reshape(
            candidate_eulers_kp.shape
        )
        best_kp_euler = candidate_eulers_kp[0, 0]
        secondary_euler = candidate_eulers[0, 0]
        secondary_sim = self._normalize_pattern_for_overlap(
            self._simulate_pattern_for_euler(int(index), secondary_euler)
        )
        secondary_ncc_kp = float(scores[0, 0])
        result.secondary_ncc_kp = secondary_ncc_kp
        result.secondary_dictionary_ncc_kp = secondary_ncc_kp
        result.secondary_ncc_full = self._pattern_ncc_for_overlap(result.residual, secondary_sim)
        result.secondary_euler_rad = secondary_euler
        result.secondary_simulated = secondary_sim
        if keep_n > 1 and self.residual_candidate_eulers_rad is not None:
            self.residual_candidate_eulers_rad[int(index)] = candidate_eulers[0]
        self.residual_point_results[int(index)] = result
        self._ensure_residual_state()
        self.residual_eulers_rad[int(index)] = np.asarray(secondary_euler, dtype=np.float64).reshape(3)
        self.residual_phases[int(index)] = int(self.current_phases[int(index)])
        row, col = self.row_col_from_index(int(index))
        if self.last_residual_scores_map is not None:
            self.last_residual_scores_map[row, col] = float(secondary_ncc_kp)
        self._invalidate_residual_color_cache()
        return result

    def _batch_index_residual_points(
        self,
        indices: np.ndarray,
        *,
        keep_n: int,
        reset_candidates: bool = True,
        residual_results: dict[int, OverlapPointResult] | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        import kikuchipy as kp

        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Residual dictionary indexing requires a Kikuchipy master pattern.")
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate or load a Kikuchipy dictionary in tab 2 before indexing a residual.")

        selected = np.asarray(indices, dtype=np.int64).ravel()
        if selected.size == 0:
            raise ValueError("No points selected.")

        keep_n = max(1, int(keep_n))
        if reset_candidates or self.residual_candidate_eulers_rad is None or (
            self.residual_candidate_eulers_rad is not None
            and self.residual_candidate_eulers_rad.shape != (self.data.count, keep_n, 3)
        ):
            self._reset_residual_candidate_cache(keep_n)

        residual_results = self.residual_point_results if residual_results is None else residual_results
        signal_mask = self._signal_mask_for_dictionary_cache(cache)
        n_per_iteration = self._dictionary_n_per_iteration(cache, signal_mask)
        batch_size = self._dictionary_index_batch_size(cache, int(selected.size))
        total_batches = int(np.ceil(selected.size / batch_size))

        for batch_number, start in enumerate(range(0, selected.size, batch_size), start=1):
            batch_indices = selected[start : start + batch_size]
            work_indices = (
                np.array([batch_indices[0], batch_indices[0]], dtype=np.int64)
                if batch_indices.size == 1
                else batch_indices
            )
            if progress_callback is not None:
                progress_callback(
                    100.0 * start / selected.size,
                    f"Preparing residual batch {batch_number}/{total_batches} ({batch_indices.size} point(s))...",
                )
            signal = self._residual_signal_from_indices(work_indices, residual_results=residual_results)
            if progress_callback is not None:
                progress_callback(
                    100.0 * start / selected.size,
                    (
                        f"Indexing residual batch {batch_number}/{total_batches} ({batch_indices.size} point(s)) "
                        f"against {cache.rotation_count} dictionary patterns"
                        f"{'' if n_per_iteration is None else f' ({n_per_iteration} dictionary patterns/iteration)'}..."
                    ),
                )
            eulers_new, scores_arr, candidate_eulers, _candidate_scores = self._dictionary_index_kikuchipy_signal(
                signal,
                cache=cache,
                keep_n=keep_n,
                signal_mask=signal_mask,
                n_per_iteration=n_per_iteration,
            )
            eulers_new = eulers_new[: batch_indices.size]
            scores = scores_arr[: batch_indices.size]
            if self.residual_candidate_eulers_rad is not None:
                self.residual_candidate_eulers_rad[batch_indices] = candidate_eulers[: batch_indices.size]

            for idx, kp_euler, score in zip(batch_indices.tolist(), eulers_new.tolist(), scores.tolist()):
                result = residual_results.get(int(idx))
                if result is None:
                    result = self.analyze_overlap_point(int(idx))
                    residual_results[int(idx)] = result
                secondary_euler = self._eulers_from_kikuchipy_frame(np.asarray(kp_euler, dtype=np.float64).reshape(1, 3))[0]
                result.secondary_euler_rad = secondary_euler
                result.secondary_dictionary_ncc_kp = float(score)
                result.secondary_ncc_kp = float(score)
                result.secondary_ncc_full = None
                result.secondary_simulated = None
                result.secondary_refined = False
                result.secondary_refinement_note = ""
                row, col = self.row_col_from_index(int(idx))
                if self.last_residual_scores_map is not None:
                    self.last_residual_scores_map[row, col] = float(score)
                self.residual_eulers_rad[int(idx)] = np.asarray(secondary_euler, dtype=np.float64).reshape(3)
                self.residual_phases[int(idx)] = int(self.current_phases[int(idx)])

            if progress_callback is not None:
                completed = min(selected.size, start + batch_indices.size)
                progress_callback(
                    100.0 * completed / selected.size,
                    f"Residual re-indexed {completed}/{selected.size} point(s)...",
                )

        self._invalidate_residual_color_cache()
        self._invalidate_overlap_mixture_cache()
        if self.residual_candidate_eulers_rad is not None and self.residual_candidate_eulers_rad.shape[1] <= 1:
            self.residual_candidate_eulers_rad = None

    def refine_overlap_residual(
        self,
        result: OverlapPointResult,
        *,
        trust_euler_deg: float = 2.0,
        maxfev: int = 50,
    ) -> OverlapPointResult:
        """Refine the residual dictionary match against the full-resolution residual."""
        import kikuchipy as kp
        from orix.crystal_map import CrystalMap
        from orix.quaternion import Rotation

        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Residual refinement requires a Kikuchipy master pattern.")
        if self.master.mp_signal is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")
        if result.secondary_euler_rad is None:
            raise RuntimeError("Index the selected residual before refining its match.")
        self._invalidate_overlap_mixture_cache()
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate or load the dictionary before residual refinement.")
        if result.residual is None or result.secondary_simulated is None:
            result = self._materialize_residual_point_result(result)

        idx = int(result.index)
        candidate_store = self.residual_candidate_eulers_rad
        candidate_count = int(candidate_store.shape[1]) if candidate_store is not None else 1
        use_candidates = candidate_store is not None and candidate_count > 1

        if use_candidates:
            candidate_batch = np.asarray(candidate_store[idx], dtype=np.float64).reshape(candidate_count, 3)
            fallback = np.repeat(np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(1, 3), candidate_count, axis=0)
            valid = np.all(np.isfinite(candidate_batch), axis=1)
            if not np.all(valid):
                candidate_batch = np.where(valid[:, np.newaxis], candidate_batch, fallback)

            work = np.repeat(np.asarray(result.residual, dtype=np.float32)[np.newaxis, ...], candidate_count, axis=0)
            signal = kp.signals.EBSD(work.astype(np.float32, copy=False))
            if len(signal.axes_manager.navigation_axes) >= 1:
                signal.axes_manager.navigation_axes[0].name = "x"
                signal.axes_manager.navigation_axes[0].scale = 1.0
                signal.axes_manager.navigation_axes[0].units = "px"

            xmap = CrystalMap(
                rotations=Rotation.from_euler(self._eulers_to_kikuchipy_frame(candidate_batch), degrees=False),
                phase_id=np.full(candidate_count, cache.phase_id, dtype=np.int32),
                x=np.arange(candidate_count, dtype=np.float64),
                phase_list=self._phase_list_for_current_master(cache.phase_id),
                scan_unit="px",
            )
            detector = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=np.repeat(self.current_pc_bruker[idx].reshape(1, 3), candidate_count, axis=0),
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            signal_mask = self._signal_mask_for_full_pattern()
            refined = signal.refine_orientation(
                xmap=xmap,
                detector=detector,
                master_pattern=self.master.mp_signal,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                signal_mask=signal_mask,
                trust_region=[float(trust_euler_deg)] * 3,
                method="minimize",
                method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=int(maxfev), disp=False)),
                compute=True,
                rechunk=False,
            )
            refined_euler_arr = np.asarray(refined.rotations.to_euler(), dtype=np.float64).reshape(candidate_count, 3)
            refined_scores = np.asarray(refined.prop.get("scores", np.full(candidate_count, np.nan)), dtype=np.float64).reshape(-1)
            if refined_scores.size < candidate_count:
                refined_scores = np.pad(refined_scores, (0, candidate_count - refined_scores.size), mode="edge")
            best_scores = np.where(np.isfinite(refined_scores[:candidate_count]), refined_scores[:candidate_count], -np.inf)
            best = int(np.argmax(best_scores))
            best_score = float(refined_scores[best])
            best_euler = self._eulers_from_kikuchipy_frame(refined_euler_arr[best].reshape(1, 3))[0]
            best_sim = self._normalize_pattern_for_overlap(self._simulate_pattern_for_euler(idx, best_euler))
            best_full_score = self._pattern_ncc_for_overlap(result.residual, best_sim)
            result.secondary_euler_rad = np.asarray(best_euler, dtype=np.float64).reshape(3)
            result.secondary_simulated = best_sim
            result.secondary_ncc_kp = best_score if np.isfinite(best_score) else result.secondary_ncc_kp
            result.secondary_ncc_full = best_full_score
            if np.isfinite(best_score):
                result.secondary_refinement_note = (
                    f"Refinement considered {candidate_count} keep_n candidate(s); selected Kikuchipy best score "
                    f"{best_score:.4f}, full-resolution NCC={best_full_score:.4f}."
                )
            else:
                result.secondary_refinement_note = (
                    f"Refinement considered {candidate_count} keep_n candidate(s); full-resolution NCC={best_full_score:.4f}."
                )
            self._clear_residual_candidate_rows(np.asarray([idx], dtype=np.int64))
        else:
            work = np.stack((result.residual, result.residual), axis=0).astype(np.float32, copy=False)
            signal = kp.signals.EBSD(work)
            if len(signal.axes_manager.navigation_axes) >= 1:
                signal.axes_manager.navigation_axes[0].name = "x"
                signal.axes_manager.navigation_axes[0].scale = 1.0
                signal.axes_manager.navigation_axes[0].units = "px"
            initial = np.repeat(np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(1, 3), 2, axis=0)
            xmap = CrystalMap(
                rotations=Rotation.from_euler(self._eulers_to_kikuchipy_frame(initial), degrees=False),
                phase_id=np.full(2, cache.phase_id, dtype=np.int32),
                x=np.arange(2, dtype=np.float64),
                phase_list=self._phase_list_for_current_master(cache.phase_id),
                scan_unit="px",
            )
            detector = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=np.repeat(self.current_pc_bruker[idx].reshape(1, 3), 2, axis=0),
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            signal_mask = self._signal_mask_for_full_pattern()
            refined = signal.refine_orientation(
                xmap=xmap,
                detector=detector,
                master_pattern=self.master.mp_signal,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                signal_mask=signal_mask,
                trust_region=[float(trust_euler_deg)] * 3,
                method="minimize",
                method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=int(maxfev), disp=False)),
                compute=True,
                rechunk=False,
            )
            refined_kp_euler = np.asarray(refined.rotations.to_euler(), dtype=np.float64).reshape(-1, 3)[0]
            refined_euler = self._eulers_from_kikuchipy_frame(refined_kp_euler.reshape(1, 3))[0]
            refined_sim = self._normalize_pattern_for_overlap(
                self._simulate_pattern_for_euler(idx, refined_euler)
            )
            refined_score = self._pattern_ncc_for_overlap(result.residual, refined_sim)
            result.secondary_euler_rad = refined_euler
            result.secondary_simulated = refined_sim
            result.secondary_ncc_kp = refined_score if np.isfinite(refined_score) else result.secondary_ncc_kp
            result.secondary_ncc_full = refined_score
            result.secondary_refinement_note = (
                f"Residual refinement complete; full-resolution NCC={refined_score:.4f}."
                if np.isfinite(refined_score)
                else "Residual refinement complete."
            )
        result.secondary_refined = True
        self.residual_point_results[int(result.index)] = result
        self._ensure_residual_state()
        self.residual_eulers_rad[idx] = np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3)
        self.residual_phases[idx] = int(self.current_phases[idx])
        row, col = self.row_col_from_index(idx)
        if self.last_residual_scores_map is not None:
            self.last_residual_scores_map[row, col] = float(
                result.secondary_ncc_kp if result.secondary_ncc_kp is not None else np.nan
            )
        self._invalidate_residual_color_cache()
        self._invalidate_overlap_mixture_cache()
        return result

    def _batch_refine_residual_points(
        self,
        indices: np.ndarray,
        *,
        trust_euler_deg: float,
        maxfev: int,
        residual_results: dict[int, OverlapPointResult] | None = None,
    ) -> list[OverlapPointResult]:
        import kikuchipy as kp
        from orix.crystal_map import CrystalMap
        from orix.quaternion import Rotation

        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Residual refinement requires a Kikuchipy master pattern.")
        if self.master.mp_signal is None or self.current_pc_bruker is None:
            raise RuntimeError("Session state is not initialized.")
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate or load a Kikuchipy dictionary in tab 2 before refining residuals.")

        selected = np.asarray(indices, dtype=np.int64).ravel()
        if selected.size == 0:
            raise ValueError("No points selected.")

        residual_results = self.residual_point_results if residual_results is None else residual_results
        signal = self._residual_signal_from_indices(
            selected,
            residual_results=residual_results,
            apply_dictionary_binning=False,
        )
        patterns = signal.data
        if hasattr(patterns, "compute"):
            patterns = patterns.compute()
        patterns = np.asarray(patterns, dtype=np.float32)
        if patterns.ndim == 2:
            patterns = patterns[np.newaxis, ...]

        candidate_store = self.residual_candidate_eulers_rad
        candidate_count = int(candidate_store.shape[1]) if candidate_store is not None else 1
        use_candidates = candidate_store is not None and candidate_count > 1

        initial_eulers: list[np.ndarray] = []
        pcs: list[np.ndarray] = []
        batch_results: list[OverlapPointResult] = []
        for idx in selected.tolist():
            result = residual_results.get(int(idx))
            if result is None:
                result = self.analyze_overlap_point(int(idx))
                residual_results[int(idx)] = result
            if result.secondary_euler_rad is None:
                self._batch_index_residual_points(
                    np.asarray([int(idx)], dtype=np.int64),
                    keep_n=candidate_count if use_candidates else 1,
                    reset_candidates=False,
                    residual_results=residual_results,
                    progress_callback=None,
                )
                result = residual_results.get(int(idx))
            if result.secondary_euler_rad is None:
                raise RuntimeError(f"Residual at idx={int(idx)} has not been indexed yet.")
            batch_results.append(result)
            initial_eulers.append(np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3))
            pcs.append(np.asarray(self.current_pc_bruker[int(idx)], dtype=np.float64).reshape(3))

        if patterns.shape[0] != len(batch_results):
            raise RuntimeError(
                f"Residual batch size mismatch: got {patterns.shape[0]} patterns for {len(batch_results)} points."
            )

        if len(batch_results) == 1 and not use_candidates:
            patterns = np.repeat(patterns, 2, axis=0)
            initial_eulers = [initial_eulers[0], initial_eulers[0]]
            pcs = [pcs[0], pcs[0]]

        if use_candidates:
            candidate_batch = np.asarray(candidate_store[selected], dtype=np.float64).reshape(
                len(batch_results),
                candidate_count,
                3,
            )
            fallback = np.repeat(np.asarray(initial_eulers, dtype=np.float64).reshape(len(batch_results), 1, 3), candidate_count, axis=1)
            valid_candidates = np.all(np.isfinite(candidate_batch), axis=2)
            if not np.all(valid_candidates):
                candidate_batch = np.where(valid_candidates[..., np.newaxis], candidate_batch, fallback)

            work_patterns = np.repeat(patterns, candidate_count, axis=0)
            work_initial = candidate_batch.reshape(-1, 3)
            work_pcs = np.repeat(np.asarray(pcs, dtype=np.float64).reshape(-1, 3), candidate_count, axis=0)

            signal = kp.signals.EBSD(work_patterns.astype(np.float32, copy=False))
            if len(signal.axes_manager.navigation_axes) >= 1:
                nav = signal.axes_manager.navigation_axes[0]
                nav.name = "x"
                nav.scale = 1.0
                nav.units = "px"

            xmap = CrystalMap(
                rotations=Rotation.from_euler(self._eulers_to_kikuchipy_frame(work_initial), degrees=False),
                phase_id=np.full(work_initial.shape[0], cache.phase_id, dtype=np.int32),
                x=np.arange(work_initial.shape[0], dtype=np.float64),
                phase_list=self._phase_list_for_current_master(cache.phase_id),
                scan_unit="px",
            )
            detector = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=work_pcs,
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            signal_mask = self._signal_mask_for_full_pattern()
            refined = signal.refine_orientation(
                xmap=xmap,
                detector=detector,
                master_pattern=self.master.mp_signal,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                signal_mask=signal_mask,
                trust_region=[float(trust_euler_deg)] * 3,
                method="minimize",
                method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=int(maxfev), disp=False)),
                compute=True,
                rechunk=False,
            )

            refined_euler_arr = np.asarray(refined.rotations.to_euler(), dtype=np.float64).reshape(
                len(batch_results),
                candidate_count,
                3,
            )
            scores_arr = np.asarray(
                refined.prop.get("scores", np.full(len(batch_results) * candidate_count, np.nan)),
                dtype=np.float64,
            ).reshape(-1)
            if scores_arr.size == 0:
                scores_arr = np.full(len(batch_results) * candidate_count, np.nan, dtype=np.float64)
            elif scores_arr.shape[0] < len(batch_results) * candidate_count:
                scores_arr = np.pad(scores_arr, (0, len(batch_results) * candidate_count - scores_arr.shape[0]), mode="edge")

            updated: list[OverlapPointResult] = []
            for i, (idx, result) in enumerate(zip(selected.tolist(), batch_results, strict=True)):
                point_scores = scores_arr[i * candidate_count : (i + 1) * candidate_count]
                if point_scores.size < candidate_count:
                    point_scores = np.pad(point_scores, (0, candidate_count - point_scores.size), mode="edge")
                best_scores = np.where(np.isfinite(point_scores), point_scores, -np.inf)
                best = int(np.argmax(best_scores))
                best_kp_score = float(point_scores[best])
                best_euler = self._eulers_from_kikuchipy_frame(refined_euler_arr[i, best].reshape(1, 3))[0]
                best_sim = self._normalize_pattern_for_overlap(self._simulate_pattern_for_euler(idx, best_euler))
                result.secondary_euler_rad = np.asarray(best_euler, dtype=np.float64).reshape(3)
                result.secondary_simulated = best_sim
                result.secondary_ncc_kp = best_kp_score if np.isfinite(best_kp_score) else result.secondary_ncc_kp
                result.secondary_ncc_full = None
                result.secondary_refinement_note = (
                    f"Residual dictionary refinement considered {candidate_count} keep_n candidate(s); selected Kikuchipy "
                    f"best score {best_kp_score:.4f}."
                    if np.isfinite(best_kp_score)
                    else f"Residual dictionary refinement considered {candidate_count} keep_n candidate(s)."
                )
                result.secondary_refined = True
                updated.append(result)

            self._clear_residual_candidate_rows(selected)
            return updated

        signal = kp.signals.EBSD(patterns.astype(np.float32, copy=False))
        if len(signal.axes_manager.navigation_axes) >= 1:
            nav = signal.axes_manager.navigation_axes[0]
            nav.name = "x"
            nav.scale = 1.0
            nav.units = "px"

        initial = np.asarray(initial_eulers, dtype=np.float64).reshape(-1, 3)
        xmap = CrystalMap(
            rotations=Rotation.from_euler(self._eulers_to_kikuchipy_frame(initial), degrees=False),
            phase_id=np.full(initial.shape[0], cache.phase_id, dtype=np.int32),
            x=np.arange(initial.shape[0], dtype=np.float64),
            phase_list=self._phase_list_for_current_master(cache.phase_id),
            scan_unit="px",
        )
        detector = kp.detectors.EBSDDetector(
            shape=(self.data.h, self.data.w),
            pc=np.asarray(pcs, dtype=np.float64).reshape(-1, 3),
            convention="bruker",
            sample_tilt=float(self.data.sample_tilt_deg),
            tilt=float(self.data.detector_tilt_deg),
            azimuthal=float(self.data.azimuthal_deg),
            twist=float(self.data.twist_deg),
        )
        signal_mask = self._signal_mask_for_full_pattern()
        refined = signal.refine_orientation(
            xmap=xmap,
            detector=detector,
            master_pattern=self.master.mp_signal,
            energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
            signal_mask=signal_mask,
            trust_region=[float(trust_euler_deg)] * 3,
            method="minimize",
            method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=int(maxfev), disp=False)),
            compute=True,
            rechunk=False,
        )

        refined_euler_arr = np.asarray(refined.rotations.to_euler(), dtype=np.float64).reshape(-1, 3)
        if refined_euler_arr.shape[0] < len(batch_results):
            raise RuntimeError(
                f"Refinement returned {refined_euler_arr.shape[0]} orientations for {len(batch_results)} points."
            )
        scores_arr = np.asarray(refined.prop.get("scores", np.full(refined_euler_arr.shape[0], np.nan)), dtype=np.float64).reshape(-1)
        if scores_arr.size == 0:
            scores_arr = np.full(len(batch_results), np.nan, dtype=np.float64)
        elif scores_arr.shape[0] < len(batch_results):
            scores_arr = np.pad(scores_arr, (0, len(batch_results) - scores_arr.shape[0]), mode="edge")

        updated: list[OverlapPointResult] = []
        for i, (idx, result) in enumerate(zip(selected.tolist(), batch_results, strict=True)):
            point_scores = scores_arr[i * candidate_count : (i + 1) * candidate_count]
            if point_scores.size < candidate_count:
                point_scores = np.pad(point_scores, (0, candidate_count - point_scores.size), mode="edge")
            best_scores = np.where(np.isfinite(point_scores), point_scores, -np.inf)
            best = int(np.argmax(best_scores))
            best_kp_score = float(point_scores[best])
            refined_euler = self._eulers_from_kikuchipy_frame(refined_euler_arr[i * candidate_count + best].reshape(1, 3))[0]
            refined_sim = self._normalize_pattern_for_overlap(self._simulate_pattern_for_euler(idx, refined_euler))
            result.secondary_euler_rad = refined_euler
            result.secondary_ncc_kp = best_kp_score if np.isfinite(best_kp_score) else result.secondary_ncc_kp
            result.secondary_ncc_full = None
            result.secondary_simulated = refined_sim
            result.secondary_refined = True
            result.secondary_refinement_note = (
                f"Residual dictionary refinement considered {candidate_count} keep_n candidate(s); selected Kikuchipy "
                f"best score {best_kp_score:.4f}."
                if np.isfinite(best_kp_score)
                else f"Residual dictionary refinement considered {candidate_count} keep_n candidate(s)."
            )
            updated.append(result)

        self._clear_residual_candidate_rows(selected)
        return updated

    def compute_overlap_residual_indices(
        self,
        indices: np.ndarray,
        *,
        fit_blur_gain: bool = True,
        fit_maxiter: int = 40,
        fit_popsize: int = 8,
        fit_bounds: list[tuple[float, float]] | None = None,
        write_patterns: bool = False,
        residual_output_path: str | None = None,
        selected_index: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")

        selected = np.asarray(indices, dtype=np.int64).ravel()
        if selected.size == 0:
            raise ValueError("No points selected.")
        target = int(selected_index) if selected_index is not None else None
        selected_result: OverlapPointResult | None = None
        self._ensure_residual_state()
        self._invalidate_overlap_mixture_cache()
        writer: ResidualPatternWriter | None = None

        if write_patterns:
            if not residual_output_path:
                raise ValueError("Residual pattern writing is enabled, but no output path was provided.")
            writer = ResidualPatternWriter.create(self.data, residual_output_path)
            self.residual_pattern_output_path = str(writer.output_path)
            self._clear_residual_pattern_source_cache()
        else:
            self.residual_pattern_output_path = None
            self._clear_residual_pattern_source_cache()

        if progress_callback is not None:
            progress_callback(0.0, f"Preparing residual calculations for {selected.size} point(s)...")

        def store_result(result: OverlapPointResult) -> None:
            nonlocal selected_result
            idx = int(result.index)
            if writer is not None and result.residual is not None:
                writer.write(idx, result.residual)
            self._clear_residual_candidate_rows(np.asarray([idx], dtype=np.int64))
            self.residual_point_results[idx] = result if not write_patterns else self._strip_residual_point_result(result)
            if target is not None and idx == target:
                selected_result = result

        def run_sequential() -> None:
            for point_number, idx in enumerate(selected.tolist(), start=1):
                result = self.analyze_overlap_point(
                    int(idx),
                    fit_blur_gain=bool(fit_blur_gain),
                    fit_maxiter=int(fit_maxiter),
                    fit_popsize=int(fit_popsize),
                    fit_bounds=fit_bounds,
                    store_result=False,
                )
                store_result(result)
                if progress_callback is not None:
                    progress_callback(
                        100.0 * point_number / selected.size,
                        f"Computed residual {point_number}/{selected.size} point(s)...",
                    )

        def build_payload(batch_indices: np.ndarray) -> ResidualBatchPayload:
            signal = self._signal_from_indices(
                batch_indices,
                software_binning=1,
                crop_extent=(0, self.data.h, 0, self.data.w),
            )
            data = signal.data
            if hasattr(data, "compute"):
                data = data.compute()
            experimental = np.asarray(data, dtype=np.float32)
            if experimental.ndim == 2:
                experimental = experimental[np.newaxis, ...]
            eulers = np.asarray(self.current_eulers_rad[batch_indices], dtype=np.float64).reshape(-1, 3)
            pc_bruker = np.asarray(self.current_pc_bruker[batch_indices], dtype=np.float64).reshape(-1, 3)
            pc_custom = np.asarray(self.current_pc_custom[batch_indices], dtype=np.float64).reshape(-1, 3)
            return ResidualBatchPayload(
                indices=np.asarray(batch_indices, dtype=np.int64).copy(),
                experimental=np.ascontiguousarray(experimental),
                eulers_rad=np.ascontiguousarray(eulers),
                pc_bruker=np.ascontiguousarray(pc_bruker),
                pc_custom=np.ascontiguousarray(pc_custom),
            )

        use_parallel = selected.size >= 4 and (os.cpu_count() or 1) > 1
        if use_parallel:
            bytes_per_pattern = max(
                1,
                int(self.data.h * self.data.w * np.dtype(np.float32).itemsize),
            )
            worker_count = max(1, min(int(os.cpu_count() or 1), int(selected.size)))
            batch_target = max(1, int(np.ceil(selected.size / max(1, worker_count * 2))))
            memory_target = max(1, int((32 * 1024**2) / bytes_per_pattern))
            batch_size = max(1, min(int(selected.size), 8, batch_target, memory_target))
            batches = [selected[start : start + batch_size] for start in range(0, selected.size, batch_size)]
            if len(batches) > 1 and worker_count > 1:
                if progress_callback is not None:
                    progress_callback(
                        0.0,
                        f"Computing residuals for {selected.size} point(s) on {worker_count} core(s)...",
                    )
                try:
                    initargs = (
                        self.master.kind,
                        self.master.path,
                        self.master.energy_kv,
                        int(self.data.h),
                        int(self.data.w),
                        int(self.data.cols),
                        self.data.rot_sd,
                        self.data.direction_cosines,
                        float(self.data.sample_tilt_deg),
                        float(self.data.detector_tilt_deg),
                        float(self.data.azimuthal_deg),
                        float(self.data.twist_deg),
                        bool(self.data.source_type == "h5oina"),
                        self._overlap_weights(),
                        bool(fit_blur_gain),
                        int(fit_maxiter),
                        int(fit_popsize),
                        fit_bounds,
                    )
                    ctx = get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=ctx,
                        initializer=_init_residual_roi_worker,
                        initargs=initargs,
                    ) as pool:
                        futures = [pool.submit(_compute_residual_roi_batch, build_payload(batch)) for batch in batches]
                        completed = 0
                        for future in as_completed(futures):
                            batch_results = future.result()
                            for result in batch_results:
                                store_result(result)
                            completed += len(batch_results)
                            if progress_callback is not None:
                                progress_callback(
                                    100.0 * completed / selected.size,
                                    f"Computed residual {completed}/{selected.size} point(s)...",
                                )
                except Exception:
                    if progress_callback is not None:
                        progress_callback(0.0, "Residual batching failed; falling back to serial processing...")
                    run_sequential()
            else:
                run_sequential()
        else:
            run_sequential()
        if writer is not None:
            writer.close()

        if target is not None and target in self.residual_point_results:
            selected_result = self.get_residual_point_result(target)
        if selected_result is not None:
            self.last_overlap = selected_result
        elif target is not None and target in self.residual_point_results:
            self.last_overlap = self.get_residual_point_result(target)

        note = f" Residual patterns written to {self.residual_pattern_output_path}." if writer is not None else ""
        return f"Computed primary residuals for {selected.size} point(s) in ROI.{note}"

    def index_overlap_residual_indices(
        self,
        indices: np.ndarray,
        *,
        keep_n: int = 1,
        write_patterns: bool = False,
        selected_index: int | None = None,
        residual_results: dict[int, OverlapPointResult] | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Residual dictionary indexing requires a Kikuchipy master pattern.")
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate or load a Kikuchipy dictionary in tab 2 before indexing a residual.")

        selected = np.asarray(indices, dtype=np.int64).ravel()
        if selected.size == 0:
            raise ValueError("No points selected.")
        target = int(selected_index) if selected_index is not None else None
        residual_results = self.residual_point_results if residual_results is None else residual_results
        self._ensure_residual_state()
        selected_result: OverlapPointResult | None = None

        if progress_callback is not None:
            progress_callback(0.0, f"Preparing residual re-indexing for {selected.size} point(s)...")

        keep_n = max(1, int(keep_n))
        reset_candidates = (
            self.residual_candidate_eulers_rad is None
            or self.residual_candidate_eulers_rad.shape != (self.data.count, keep_n, 3)
            or keep_n <= 1
        )
        self._batch_index_residual_points(
            selected,
            keep_n=keep_n,
            reset_candidates=reset_candidates,
            residual_results=residual_results,
            progress_callback=progress_callback,
        )

        if target is not None and target in residual_results:
            selected_result = self.get_residual_point_result(target)

        if write_patterns:
            # Writing residuals is a storage choice only; the batch indexing above already ran.
            for idx in selected.tolist():
                result = residual_results.get(int(idx))
                if result is not None:
                    residual_results[int(idx)] = self._strip_residual_point_result(result)

        self.last_residual_indexed_indices = selected.copy()
        if target is not None and target in residual_results:
            selected_result = self.get_residual_point_result(target)
        if selected_result is not None:
            self.last_overlap = selected_result
        elif target is not None and target in residual_results:
            self.last_overlap = self.get_residual_point_result(target)

        return f"Residual re-indexed {selected.size} point(s) in ROI (keep_n={int(keep_n)})."

    def refine_overlap_residual_indices(
        self,
        indices: np.ndarray,
        *,
        trust_euler_deg: float = 2.0,
        maxfev: int = 50,
        write_patterns: bool = False,
        selected_index: int | None = None,
        residual_results: dict[int, OverlapPointResult] | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Residual refinement requires a Kikuchipy master pattern.")
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate or load a Kikuchipy dictionary in tab 2 before refining residuals.")

        selected = np.asarray(indices, dtype=np.int64).ravel()
        if selected.size == 0:
            raise ValueError("No points selected.")
        target = int(selected_index) if selected_index is not None else None
        residual_results = self.residual_point_results if residual_results is None else residual_results
        self._ensure_residual_state()
        selected_result: OverlapPointResult | None = None
        candidate_store = self.residual_candidate_eulers_rad
        candidate_count = int(candidate_store.shape[1]) if candidate_store is not None else 1

        if progress_callback is not None:
            progress_callback(0.0, f"Preparing residual refinement for {selected.size} point(s)...")

        missing_secondary = np.asarray(
            [
                int(idx)
                for idx in selected.tolist()
                if (
                    (result := residual_results.get(int(idx))) is None
                    or result.secondary_euler_rad is None
                )
            ],
            dtype=np.int64,
        )
        if missing_secondary.size > 0:
            self._batch_index_residual_points(
                missing_secondary,
                keep_n=candidate_count,
                reset_candidates=False,
                residual_results=residual_results,
                progress_callback=None,
            )

        bytes_per_pattern = max(1, int(self.data.h * self.data.w * np.dtype(np.float32).itemsize))
        candidate_factor = max(1, candidate_count)
        batch_cap = max(1, 256 // candidate_factor)
        memory_cap = max(1, int((128 * 1024**2) / (bytes_per_pattern * candidate_factor)))
        batch_size = max(1, min(int(selected.size), batch_cap, memory_cap))

        for start in range(0, selected.size, batch_size):
            batch_indices = selected[start : start + batch_size]
            batch_results = self._batch_refine_residual_points(
                batch_indices,
                trust_euler_deg=float(trust_euler_deg),
                maxfev=int(maxfev),
                residual_results=residual_results,
            )
            for result in batch_results:
                idx = int(result.index)
                residual_results[idx] = result if not write_patterns else self._strip_residual_point_result(result)
                if result.secondary_euler_rad is not None:
                    self.residual_eulers_rad[idx] = np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3)
                if self.last_residual_scores_map is not None:
                    row, col = self.row_col_from_index(idx)
                    self.last_residual_scores_map[row, col] = float(
                        result.secondary_ncc_kp if result.secondary_ncc_kp is not None else np.nan
                    )
                if target is not None and idx == target:
                    selected_result = result
            if progress_callback is not None:
                completed = min(selected.size, start + batch_indices.size)
                progress_callback(
                    100.0 * completed / selected.size,
                    f"Residual refined {completed}/{selected.size} point(s)...",
                )

        self.last_residual_indexed_indices = selected.copy()
        if selected_result is not None:
            self.last_overlap = selected_result
        elif target is not None and target in residual_results:
            self.last_overlap = self.get_residual_point_result(target)

        self._invalidate_residual_color_cache()
        self._invalidate_overlap_mixture_cache()
        candidate_note = f" using keep_n={candidate_count}" if candidate_count > 1 else ""
        return f"Residual-refined {selected.size} point(s) in ROI{candidate_note}."

    def fit_overlap_mixture_point(
        self,
        index: int,
        *,
        residual_result: OverlapPointResult | None = None,
        fit_maxiter: int = 40,
        fit_popsize: int = 8,
        fit_bounds: list[tuple[float, float]] | None = None,
        store_result: bool = True,
    ) -> OverlapMixtureResult:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")

        idx = int(index)
        secondary_euler, old_primary_ncc, old_secondary_ncc = self._secondary_overlap_orientation_for_index(
            idx,
            residual_result=residual_result,
        )
        if secondary_euler is None:
            raise RuntimeError("Run residual indexing/refinement for this point before overlap optimization.")

        row, col = self.row_col_from_index(idx)
        primary_euler = np.asarray(self.current_eulers_rad[idx], dtype=np.float64).reshape(3)
        experimental_raw = self._processed_pattern_at(idx)
        primary_raw = self._simulate_pattern_for_euler(idx, primary_euler)
        secondary_raw = self._simulate_pattern_for_euler(idx, secondary_euler)
        result = _overlap_mixture_result_from_raw_patterns(
            idx,
            row,
            col,
            experimental_raw,
            primary_raw,
            secondary_raw,
            self._overlap_weights(),
            primary_euler_rad=primary_euler,
            secondary_euler_rad=secondary_euler,
            old_primary_ncc=old_primary_ncc,
            old_secondary_ncc=old_secondary_ncc,
            fit_maxiter=int(fit_maxiter),
            fit_popsize=int(fit_popsize),
            fit_bounds=fit_bounds,
        )
        if store_result:
            self._store_overlap_mixture_result(result, keep_patterns=True)
        return result

    def refine_overlap_mixture_orientations(
        self,
        result: OverlapMixtureResult,
        *,
        trust_euler_deg: float = 1.0,
        maxfev: int = 80,
    ) -> OverlapMixtureResult:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_phases is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(result.index)
        if result.experimental is None or result.primary_simulated is None or result.secondary_simulated is None:
            materialized = self.get_overlap_mixture_result(idx)
            if materialized is None:
                raise RuntimeError("Fit the selected point mixture before refining its orientations.")
            result = materialized
        if result.secondary_euler_rad is None:
            raise RuntimeError("The selected mixture result has no residual orientation.")

        row, col = self.row_col_from_index(idx)
        weights = self._overlap_weights()
        experimental_raw = self._processed_pattern_at(idx)
        primary_start = (
            np.asarray(result.primary_euler_rad, dtype=np.float64).reshape(3)
            if result.primary_euler_rad is not None
            else np.asarray(self.current_eulers_rad[idx], dtype=np.float64).reshape(3)
        )
        secondary_start = np.asarray(result.secondary_euler_rad, dtype=np.float64).reshape(3)
        fixed_params = np.asarray(
            [
                float(result.fitted_sigma),
                *tuple(float(v) for v in result.gain_params[:3]),
                *tuple(float(v) for v in result.ellipse_params[:4]),
            ],
            dtype=np.float64,
        )
        if fixed_params.size != 8:
            raise ValueError("Stored overlap mixture parameters are incomplete.")
        reported_initial_ncc = (
            float(result.ncc_mixture)
            if np.isfinite(float(result.ncc_mixture))
            else None
        )
        trust_rad = float(np.deg2rad(max(0.0, float(trust_euler_deg))))
        maxfev_i = max(0, int(maxfev))

        def retain_existing(note: str, *, fit_message: str | None = None) -> OverlapMixtureResult:
            retained = replace(
                result,
                orientation_refined=bool(result.orientation_refined),
                orientation_refinement_note=str(note),
                initial_mixture_ncc=(
                    result.initial_mixture_ncc
                    if result.initial_mixture_ncc is not None
                    else reported_initial_ncc
                ),
                primary_euler_delta_deg=(
                    tuple(float(v) for v in result.primary_euler_delta_deg)
                    if result.primary_euler_delta_deg
                    else (0.0, 0.0, 0.0)
                ),
                secondary_euler_delta_deg=(
                    tuple(float(v) for v in result.secondary_euler_delta_deg)
                    if result.secondary_euler_delta_deg
                    else (0.0, 0.0, 0.0)
                ),
                fit_message=str(fit_message if fit_message is not None else result.fit_message),
            )
            self._store_overlap_mixture_result(retained, keep_patterns=True)
            return retained

        if trust_rad <= 0.0 or maxfev_i <= 0:
            note = "Mixture orientation refinement skipped because trust region or max evaluations is zero."
            return retain_existing(note, fit_message=note)

        evaluation_cache: dict[tuple[float, ...], tuple[float, OverlapMixtureFit, np.ndarray, np.ndarray]] = {}
        evaluations = 0

        def evaluate_delta(delta: np.ndarray) -> tuple[float, OverlapMixtureFit, np.ndarray, np.ndarray]:
            nonlocal evaluations
            d = np.asarray(delta, dtype=np.float64).reshape(6)
            d = np.clip(d, -trust_rad, trust_rad)
            key = tuple(float(v) for v in np.round(d, 12))
            cached = evaluation_cache.get(key)
            if cached is not None:
                return cached
            primary_euler = primary_start + d[:3]
            secondary_euler = secondary_start + d[3:]
            primary_raw = self._simulate_pattern_for_euler(idx, primary_euler)
            secondary_raw = self._simulate_pattern_for_euler(idx, secondary_euler)
            fit = _evaluate_overlap_mixture_pattern(
                experimental_raw,
                primary_raw,
                secondary_raw,
                weights,
                fixed_params,
            )
            ssr = float(np.sum(np.asarray(weights, dtype=np.float32) * fit.residual * fit.residual))
            value = (ssr, fit, primary_euler, secondary_euler)
            evaluation_cache[key] = value
            evaluations += 1
            return value

        zero_delta = np.zeros(6, dtype=np.float64)
        initial_ssr, initial_fit, _initial_primary, _initial_secondary = evaluate_delta(zero_delta)
        initial_ncc = float(initial_fit.ncc_mixture)
        reported_initial_ncc = reported_initial_ncc if reported_initial_ncc is not None else initial_ncc
        best_delta = zero_delta.copy()
        best_ssr = float(initial_ssr)
        best_fit = initial_fit
        best_primary = primary_start.copy()
        best_secondary = secondary_start.copy()
        for step in (trust_rad, trust_rad * 0.5, trust_rad * 0.25):
            if step <= 0.0:
                continue
            for dim in range(6):
                for sign in (-1.0, 1.0):
                    if evaluations >= maxfev_i:
                        break
                    trial = np.zeros(6, dtype=np.float64)
                    trial[dim] = sign * step
                    ssr, fit, primary_euler, secondary_euler = evaluate_delta(trial)
                    if ssr < best_ssr:
                        best_delta = trial.copy()
                        best_ssr = float(ssr)
                        best_fit = fit
                        best_primary = primary_euler
                        best_secondary = secondary_euler
                if evaluations >= maxfev_i:
                    break
            if evaluations >= maxfev_i:
                break

        optimization = None
        refined_delta = best_delta.copy()
        refined_fit = best_fit
        refined_primary = best_primary
        refined_secondary = best_secondary
        remaining_evals = max(0, maxfev_i - evaluations)
        bounds = [(-trust_rad, trust_rad)] * 6

        def objective(delta: np.ndarray) -> float:
            ssr, _fit, _primary_euler, _secondary_euler = evaluate_delta(delta)
            return ssr

        if remaining_evals > 0:
            optimization = minimize(
                objective,
                best_delta,
                method="Powell",
                bounds=bounds,
                options=dict(maxfev=remaining_evals, disp=False, xtol=1e-5, ftol=1e-6),
            )
            candidate_delta = np.asarray(optimization.x, dtype=np.float64).reshape(6)
            candidate_ssr, candidate_fit, candidate_primary, candidate_secondary = evaluate_delta(candidate_delta)
            if candidate_ssr <= best_ssr:
                refined_delta = candidate_delta
                refined_fit = candidate_fit
                refined_primary = candidate_primary
                refined_secondary = candidate_secondary
        refined_ncc = float(refined_fit.ncc_mixture)
        ncc_gain = refined_ncc - initial_ncc
        primary_delta_deg = tuple(float(v) for v in np.rad2deg(refined_delta[:3]))
        secondary_delta_deg = tuple(float(v) for v in np.rad2deg(refined_delta[3:]))
        max_delta_deg = max(
            [0.0]
            + [abs(float(v)) for v in primary_delta_deg]
            + [abs(float(v)) for v in secondary_delta_deg]
        )
        accepted = np.isfinite(refined_ncc) and max_delta_deg >= 1e-4 and (
            not np.isfinite(initial_ncc) or ncc_gain > 1e-6
        )

        if accepted:
            chosen_fit = refined_fit
            chosen_primary = refined_primary
            chosen_secondary = refined_secondary
            chosen_delta = refined_delta
            note = (
                f"Mixture orientation refinement accepted: combined NCC {reported_initial_ncc:.4f} -> "
                f"{refined_ncc:.4f}; max Euler move {max_delta_deg:.4f} deg."
            )
        else:
            note = (
                f"No orientation update accepted: best trial NCC {refined_ncc:.4f}, "
                f"gain {ncc_gain:.2e}, max Euler move {max_delta_deg:.4f} deg."
            )
            fit_message = str(optimization.message) if optimization is not None else "Probe search only."
            return retain_existing(note, fit_message=fit_message)

        chosen_fit.success = bool(optimization.success) if optimization is not None else True
        chosen_fit.message = str(optimization.message) if optimization is not None else "Accepted coordinate-probe refinement."
        primary_delta_deg = tuple(float(v) for v in np.rad2deg(chosen_delta[:3]))
        secondary_delta_deg = tuple(float(v) for v in np.rad2deg(chosen_delta[3:]))
        refined_result = _overlap_mixture_result_from_fit(
            idx,
            row,
            col,
            chosen_fit,
            weights,
            primary_euler_rad=chosen_primary,
            secondary_euler_rad=chosen_secondary,
            old_primary_ncc=result.old_primary_ncc,
            old_secondary_ncc=result.old_secondary_ncc,
            orientation_refined=bool(accepted),
            orientation_refinement_note=note,
            initial_mixture_ncc=reported_initial_ncc,
            primary_euler_delta_deg=primary_delta_deg,
            secondary_euler_delta_deg=secondary_delta_deg,
        )

        if accepted:
            self.current_eulers_rad[idx] = np.asarray(chosen_primary, dtype=np.float64).reshape(3)
            self._ensure_residual_state()
            self.residual_eulers_rad[idx] = np.asarray(chosen_secondary, dtype=np.float64).reshape(3)
            self.residual_phases[idx] = int(self.current_phases[idx])
            self.residual_point_results.pop(idx, None)
            if self.last_overlap is not None and int(self.last_overlap.index) == idx:
                self.last_overlap = None
            self._invalidate_orientation_cache()
            self._invalidate_residual_color_cache()

        self._store_overlap_mixture_result(refined_result, keep_patterns=True)
        return refined_result

    def _overlap_mixture_inputs_for_indices(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        selected = np.asarray(indices, dtype=np.int64).ravel()
        valid_indices: list[int] = []
        secondary_eulers: list[np.ndarray] = []
        old_primary_ncc: list[float] = []
        old_secondary_ncc: list[float] = []
        skipped = 0
        for idx in selected.tolist():
            secondary_euler, primary_ncc, secondary_ncc = self._secondary_overlap_orientation_for_index(int(idx))
            if secondary_euler is None:
                skipped += 1
                continue
            valid_indices.append(int(idx))
            secondary_eulers.append(np.asarray(secondary_euler, dtype=np.float64).reshape(3))
            old_primary_ncc.append(float(primary_ncc) if primary_ncc is not None else float("nan"))
            old_secondary_ncc.append(float(secondary_ncc) if secondary_ncc is not None else float("nan"))
        if not valid_indices:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                skipped,
            )
        return (
            np.asarray(valid_indices, dtype=np.int64),
            np.asarray(secondary_eulers, dtype=np.float64).reshape(-1, 3),
            np.asarray(old_primary_ncc, dtype=np.float64),
            np.asarray(old_secondary_ncc, dtype=np.float64),
            skipped,
        )

    def compute_overlap_mixture_indices(
        self,
        indices: np.ndarray,
        *,
        fit_maxiter: int = 40,
        fit_popsize: int = 8,
        fit_bounds: list[tuple[float, float]] | None = None,
        selected_index: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")

        requested = np.asarray(indices, dtype=np.int64).ravel()
        if requested.size == 0:
            raise ValueError("No points selected.")
        selected, secondary_eulers, old_primary_ncc, old_secondary_ncc, skipped = self._overlap_mixture_inputs_for_indices(requested)
        if selected.size == 0:
            raise RuntimeError("No ROI points have a residual-indexed secondary orientation. Run step 3 first.")

        target = int(selected_index) if selected_index is not None and np.any(selected == int(selected_index)) else None
        self._ensure_overlap_mixture_state()
        if progress_callback is not None:
            progress_callback(0.0, f"Preparing overlap mixture fits for {selected.size} point(s)...")

        def store_result(result: OverlapMixtureResult) -> None:
            self._store_overlap_mixture_result(result, keep_patterns=(target is not None and int(result.index) == target))

        def run_sequential() -> None:
            for point_number, idx in enumerate(selected.tolist(), start=1):
                result = self.fit_overlap_mixture_point(
                    int(idx),
                    fit_maxiter=int(fit_maxiter),
                    fit_popsize=int(fit_popsize),
                    fit_bounds=fit_bounds,
                    store_result=False,
                )
                store_result(result)
                if progress_callback is not None:
                    progress_callback(
                        100.0 * point_number / selected.size,
                        f"Fitted overlap mixture {point_number}/{selected.size} point(s)...",
                    )

        def build_payload(batch_positions: np.ndarray) -> OverlapMixtureBatchPayload:
            batch_indices = selected[batch_positions]
            signal = self._signal_from_indices(
                batch_indices,
                software_binning=1,
                crop_extent=(0, self.data.h, 0, self.data.w),
            )
            data = signal.data
            if hasattr(data, "compute"):
                data = data.compute()
            experimental = np.asarray(data, dtype=np.float32)
            if experimental.ndim == 2:
                experimental = experimental[np.newaxis, ...]
            return OverlapMixtureBatchPayload(
                indices=np.asarray(batch_indices, dtype=np.int64).copy(),
                experimental=np.ascontiguousarray(experimental),
                primary_eulers_rad=np.ascontiguousarray(
                    np.asarray(self.current_eulers_rad[batch_indices], dtype=np.float64).reshape(-1, 3)
                ),
                secondary_eulers_rad=np.ascontiguousarray(
                    np.asarray(secondary_eulers[batch_positions], dtype=np.float64).reshape(-1, 3)
                ),
                pc_bruker=np.ascontiguousarray(
                    np.asarray(self.current_pc_bruker[batch_indices], dtype=np.float64).reshape(-1, 3)
                ),
                pc_custom=np.ascontiguousarray(
                    np.asarray(self.current_pc_custom[batch_indices], dtype=np.float64).reshape(-1, 3)
                ),
                old_primary_ncc=np.ascontiguousarray(old_primary_ncc[batch_positions]),
                old_secondary_ncc=np.ascontiguousarray(old_secondary_ncc[batch_positions]),
            )

        use_parallel = selected.size >= 4 and (os.cpu_count() or 1) > 1
        if use_parallel:
            bytes_per_pattern = max(1, int(self.data.h * self.data.w * np.dtype(np.float32).itemsize))
            worker_count = max(1, min(int(os.cpu_count() or 1), int(selected.size)))
            batch_target = max(1, int(np.ceil(selected.size / max(1, worker_count * 2))))
            memory_target = max(1, int((32 * 1024**2) / bytes_per_pattern))
            batch_size = max(1, min(int(selected.size), 8, batch_target, memory_target))
            positions = np.arange(selected.size, dtype=np.int64)
            batches = [positions[start : start + batch_size] for start in range(0, selected.size, batch_size)]
            if len(batches) > 1 and worker_count > 1:
                if progress_callback is not None:
                    progress_callback(
                        0.0,
                        f"Fitting overlap mixtures for {selected.size} point(s) on {worker_count} core(s)...",
                    )
                try:
                    initargs = (
                        self.master.kind,
                        self.master.path,
                        self.master.energy_kv,
                        int(self.data.h),
                        int(self.data.w),
                        int(self.data.cols),
                        self.data.rot_sd,
                        self.data.direction_cosines,
                        float(self.data.sample_tilt_deg),
                        float(self.data.detector_tilt_deg),
                        float(self.data.azimuthal_deg),
                        float(self.data.twist_deg),
                        bool(self.data.source_type == "h5oina"),
                        self._overlap_weights(),
                        True,
                        int(fit_maxiter),
                        int(fit_popsize),
                        fit_bounds,
                    )
                    ctx = get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=ctx,
                        initializer=_init_residual_roi_worker,
                        initargs=initargs,
                    ) as pool:
                        futures = [pool.submit(_compute_overlap_mixture_roi_batch, build_payload(batch)) for batch in batches]
                        completed = 0
                        for future in as_completed(futures):
                            batch_results = future.result()
                            for result in batch_results:
                                store_result(result)
                            completed += len(batch_results)
                            if progress_callback is not None:
                                progress_callback(
                                    100.0 * completed / selected.size,
                                    f"Fitted overlap mixture {completed}/{selected.size} point(s)...",
                                )
                except Exception:
                    if progress_callback is not None:
                        progress_callback(0.0, "Overlap mixture batching failed; falling back to serial processing...")
                    run_sequential()
            else:
                run_sequential()
        else:
            run_sequential()

        if target is not None and target in self.overlap_mixture_results:
            self.last_overlap_mixture = self.get_overlap_mixture_result(target)

        skipped_note = f" Skipped {skipped} point(s) without residual orientation." if skipped > 0 else ""
        return f"Fitted overlap mixtures for {selected.size} point(s) in ROI.{skipped_note}"

    def preview_simulated_pattern(self, index: int) -> np.ndarray:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")
        idx = int(index)
        if self.master.kind == "kikuchipy" and self.master.mp_signal is not None:
            import kikuchipy as kp
            from orix.quaternion import Rotation

            rot = Rotation.from_euler(
                self._eulers_to_kikuchipy_frame(self.current_eulers_rad[idx : idx + 1]),
                degrees=False,
            )
            det = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=self.current_pc_bruker[idx],
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            sim_sig = self.master.mp_signal.get_patterns(
                rotations=rot,
                detector=det,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                compute=True,
                show_progressbar=False,
            )
            return self._normalize_pattern_for_overlap(np.asarray(sim_sig.data[0], dtype=np.float32))
        assert self.master.projector is not None
        sim = self.master.projector.project(
            self.current_eulers_rad[idx],
            (
                float(self.current_pc_custom[idx, 0]),
                float(self.current_pc_custom[idx, 1]),
                float(self.current_pc_custom[idx, 2]),
            ),
            self.data.rot_sd,
            direction_cosines=self.data.direction_cosines,
        )
        return self._normalize_pattern_for_overlap(sim)

    def preview_simulated_pattern_with_ncc(
        self,
        index: int,
        *,
        euler_rad_override: np.ndarray | None = None,
        pc_custom_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, float, float]:
        if self.data is None or self.master is None:
            raise RuntimeError("Load both input data and master pattern first.")
        if self.current_eulers_rad is None or self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")

        idx = int(index)
        e_use = (
            np.asarray(euler_rad_override, dtype=np.float64).reshape(3)
            if euler_rad_override is not None
            else self.current_eulers_rad[idx].astype(np.float64, copy=True)
        )
        if pc_custom_override is not None:
            pc_c = np.asarray(pc_custom_override, dtype=np.float64).reshape(3)
            pc_b = _convert_pc_map(
                pc_c.reshape(1, 1, 3),
                src_convention=self.data.pc_output_convention,
                dst_convention="bruker",
                shape=(self.data.h, self.data.w),
                sample_tilt_deg=self.data.sample_tilt_deg,
                detector_tilt_deg=self.data.detector_tilt_deg,
                azimuthal_deg=self.data.azimuthal_deg,
                twist_deg=self.data.twist_deg,
            ).reshape(3)
            pc_use_custom = pc_c
            pc_use_bruker = pc_b
        else:
            pc_use_custom = self.current_pc_custom[idx].astype(np.float64, copy=True)
            pc_use_bruker = self.current_pc_bruker[idx].astype(np.float64, copy=True)

        exp = self._normalize_pattern_for_overlap(self._processed_pattern_at(idx))
        if self.master.kind == "kikuchipy" and self.master.mp_signal is not None:
            import kikuchipy as kp
            from orix.quaternion import Rotation

            rot = Rotation.from_euler(self._eulers_to_kikuchipy_frame(e_use.reshape(1, 3)), degrees=False)
            det = kp.detectors.EBSDDetector(
                shape=(self.data.h, self.data.w),
                pc=pc_use_bruker,
                convention="bruker",
                sample_tilt=float(self.data.sample_tilt_deg),
                tilt=float(self.data.detector_tilt_deg),
                azimuthal=float(self.data.azimuthal_deg),
                twist=float(self.data.twist_deg),
            )
            sim_sig = self.master.mp_signal.get_patterns(
                rotations=rot,
                detector=det,
                energy=float(self.master.energy_kv) if self.master.energy_kv is not None else 20.0,
                compute=True,
                show_progressbar=False,
            )
            sim = np.asarray(sim_sig.data[0], dtype=np.float32)
        else:
            assert self.master.projector is not None
            sim_raw = self.master.projector.project(
                e_use,
                (float(pc_use_custom[0]), float(pc_use_custom[1]), float(pc_use_custom[2])),
                self.data.rot_sd,
                direction_cosines=self.data.direction_cosines,
            )
            sim = sim_raw

        sim = self._normalize_pattern_for_overlap(sim)

        ncc_val = self._pattern_ncc_for_overlap(exp, sim)
        # Original overlap workflow: subtract the NCC-scaled standardized pattern.
        scale = ncc_val
        residual = _zero_unweighted_pixels(exp - scale * sim, self._overlap_weights())
        ncc_residual_sim = self._pattern_ncc_for_overlap(residual, sim)
        return sim, ncc_val, residual, scale, ncc_residual_sim

    # ----------------------------- Save ---------------------------- #

    def save_dictionary(self, output_path: str) -> str:
        """Save the generated binned dictionary and its indexing metadata."""
        cache = self.dictionary_cache
        if cache is None:
            raise RuntimeError("Generate a dictionary before saving it.")
        out = Path(output_path).expanduser().resolve()
        if out.suffix.lower() not in {".h5", ".hdf5"}:
            out = out.with_suffix(".h5")
        out.parent.mkdir(parents=True, exist_ok=True)
        patterns = cache.signal.data
        if hasattr(patterns, "compute"):
            patterns = patterns.compute()
        patterns = np.asarray(patterns, dtype=np.float32).reshape(
            cache.rotation_count,
            *cache.pattern_shape,
        )
        eulers = np.asarray(cache.signal.xmap.rotations.to_euler(), dtype=np.float64).reshape(-1, 3)
        with h5py.File(out, "w") as h5:
            h5.attrs["format"] = "overlap-ebsd-kikuchipy-dictionary-v1"
            h5.attrs["phase_id"] = int(cache.phase_id)
            h5.attrs["resolution_deg"] = float(cache.resolution_deg)
            h5.attrs["software_binning"] = int(cache.software_binning)
            h5.create_dataset(
                "patterns",
                data=patterns,
                dtype=np.float32,
                chunks=(1, *cache.pattern_shape),
                compression="lzf",
            )
            h5.create_dataset("eulers_rad", data=eulers, dtype=np.float64)
            h5.create_dataset("pc_bruker", data=cache.pc_bruker, dtype=np.float64)
            h5.create_dataset("crop_extent", data=np.asarray(cache.crop_extent, dtype=np.int64))
        return (
            f"Saved dictionary with {cache.rotation_count} patterns at binning "
            f"{cache.software_binning} to {out}"
        )

    def load_dictionary(self, input_path: str) -> str:
        """Load a previously saved binned dictionary for indexing and overlap work."""
        import kikuchipy as kp
        from orix.crystal_map import CrystalMap
        from orix.quaternion import Rotation

        if self.data is None or self.master is None or self.master.kind != "kikuchipy":
            raise RuntimeError("Load input data and its Kikuchipy master pattern before loading a dictionary.")
        path = Path(input_path).expanduser().resolve()
        with h5py.File(path, "r") as h5:
            if str(h5.attrs.get("format", "")) != "overlap-ebsd-kikuchipy-dictionary-v1":
                raise ValueError("This is not a supported overlap-EBSD dictionary file.")
            phase_id = int(h5.attrs["phase_id"])
            resolution_deg = float(h5.attrs["resolution_deg"])
            software_binning = int(h5.attrs["software_binning"])
            patterns = np.asarray(h5["patterns"][()], dtype=np.float32)
            eulers = np.asarray(h5["eulers_rad"][()], dtype=np.float64).reshape(-1, 3)
            pc_bruker = np.asarray(h5["pc_bruker"][()], dtype=np.float64).reshape(3)
            crop_extent = tuple(int(v) for v in np.asarray(h5["crop_extent"][()]).reshape(4))
        if patterns.shape[0] != eulers.shape[0]:
            raise ValueError("Dictionary pattern and rotation counts do not match.")
        expected_extent = self._binning_crop_extent(software_binning)
        if crop_extent != expected_extent:
            raise ValueError(
                f"Dictionary crop extent {crop_extent} does not match the loaded pattern shape "
                f"(expected {expected_extent})."
            )
        pattern_shape = tuple(int(v) for v in patterns.shape[-2:])
        expected_shape = (
            (crop_extent[1] - crop_extent[0]) // software_binning,
            (crop_extent[3] - crop_extent[2]) // software_binning,
        )
        if pattern_shape != expected_shape:
            raise ValueError(f"Dictionary pattern shape {pattern_shape} does not match expected {expected_shape}.")
        dictionary = kp.signals.EBSD(patterns)
        if len(dictionary.axes_manager.navigation_axes) >= 1:
            dictionary.axes_manager.navigation_axes[0].name = "x"
            dictionary.axes_manager.navigation_axes[0].scale = 1.0
            dictionary.axes_manager.navigation_axes[0].units = "px"
        rotations = Rotation.from_euler(eulers, degrees=False)
        dictionary.xmap = CrystalMap(
            rotations=rotations,
            phase_id=np.full(rotations.size, phase_id, dtype=np.int32),
            phase_list=self._phase_list_for_current_master(phase_id),
        )
        self.dictionary_cache = DictionaryCache(
            phase_id=phase_id,
            resolution_deg=resolution_deg,
            pc_bruker=pc_bruker.copy(),
            software_binning=software_binning,
            crop_extent=crop_extent,
            pattern_shape=pattern_shape,
            signal=dictionary,
            rotation_count=int(patterns.shape[0]),
        )
        self.dictionary_settings = {
            "phase_id": phase_id,
            "resolution_deg": resolution_deg,
            "pc_bruker": pc_bruker.copy(),
            "software_binning": software_binning,
            "crop_extent": np.asarray(crop_extent, dtype=np.int64),
        }
        return (
            f"Loaded dictionary: {patterns.shape[0]} patterns, shape={pattern_shape[0]}x{pattern_shape[1]}, "
            f"software binning={software_binning}, from {path}"
        )

    def save_workflow_state(self, output_path: str) -> str:
        """Save enough state to reopen refined/indexed work without altering source data."""
        if self.data is None or self.current_eulers_rad is None or self.current_phases is None:
            raise RuntimeError("Load input data first.")
        if self.current_pc_bruker is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")
        out_path = Path(output_path).expanduser().resolve()
        if out_path.suffix.lower() != ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        settings = self.dictionary_settings or {}
        dict_pc = np.asarray(settings.get("pc_bruker", np.full(3, np.nan)), dtype=np.float64).reshape(3)
        np.savez_compressed(
            out_path,
            pattern_path=np.asarray(self.data.pattern_path),
            orientation_path=np.asarray(self.data.orientation_path or ""),
            master_path=np.asarray(self.master.path if self.master is not None else ""),
            initial_eulers_rad=(
                self.initial_eulers_rad
                if self.initial_eulers_rad is not None
                else np.empty((0, 3), dtype=np.float64)
            ),
            eulers_rad=self.current_eulers_rad,
            phases=self.current_phases,
            pc_bruker=self.current_pc_bruker,
            pc_custom=self.current_pc_custom,
            scores=self.last_scores_map if self.last_scores_map is not None else np.empty((0, 0), dtype=np.float32),
            indexed_candidate_eulers_rad=(
                self.indexed_candidate_eulers_rad
                if self.indexed_candidate_eulers_rad is not None
                else np.empty((0, 0, 3), dtype=np.float64)
            ),
            calibration_indices=np.asarray(self.calibration_indices, dtype=np.int64),
            last_indexed_indices=(
                self.last_indexed_indices
                if self.last_indexed_indices is not None
                else np.empty(0, dtype=np.int64)
            ),
            sample_tilt=np.asarray(self.data.sample_tilt_deg),
            detector_tilt=np.asarray(self.data.detector_tilt_deg),
            azimuthal=np.asarray(self.data.azimuthal_deg),
            twist=np.asarray(self.data.twist_deg),
            dictionary_phase=np.asarray(int(settings.get("phase_id", -1))),
            dictionary_resolution=np.asarray(float(settings.get("resolution_deg", np.nan))),
            dictionary_pc_bruker=dict_pc,
            dictionary_software_binning=np.asarray(int(settings.get("software_binning", 1))),
            dictionary_crop_extent=np.asarray(
                settings.get("crop_extent", np.array([-1, -1, -1, -1], dtype=np.int64)),
                dtype=np.int64,
            ),
            pattern_mask_option=np.asarray(int(self.pattern_mask_option), dtype=np.int64),
            dynamic_bg_enabled=np.asarray(bool(self.dynamic_bg_config.enabled)),
            dynamic_bg_std_px=np.asarray(float(self.dynamic_bg_config.std_px)),
            dynamic_bg_truncate=np.asarray(float(self.dynamic_bg_config.truncate)),
            residual_pattern_output_path=np.asarray(self.residual_pattern_output_path or ""),
        )
        return f"Saved workflow state to {out_path}"

    def restore_workflow_state(self, input_path: str) -> str:
        """Reopen a saved workflow, including source data and refined map state."""
        path = Path(input_path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as state:
            pattern_path = str(state["pattern_path"].item())
            orientation_path = str(state["orientation_path"].item()) or None
            master_path = str(state["master_path"].item())
            geom = GeometryConfig(
                sample_tilt_deg=float(state["sample_tilt"].item()),
                detector_tilt_deg=float(state["detector_tilt"].item()),
                azimuthal_deg=float(state["azimuthal"].item()),
                twist_deg=float(state["twist"].item()),
            )
            load_note = self.load_input(pattern_path, orientation_path, geom)
            master_note = self.load_master(master_path) if master_path else "No master pattern stored."
            mask_option = int(state["pattern_mask_option"].item()) if "pattern_mask_option" in state.files else -1
            self.set_pattern_mask_option(mask_option)
            dynamic_bg_enabled = bool(state["dynamic_bg_enabled"].item()) if "dynamic_bg_enabled" in state.files else False
            dynamic_bg_std_px = float(state["dynamic_bg_std_px"].item()) if "dynamic_bg_std_px" in state.files else 0.0
            dynamic_bg_truncate = (
                float(state["dynamic_bg_truncate"].item()) if "dynamic_bg_truncate" in state.files else 4.0
            )
            self.set_dynamic_background(
                dynamic_bg_enabled,
                std_px=dynamic_bg_std_px,
                truncate=dynamic_bg_truncate,
            )
            self.last_action_note = ""
            expected = self.data.count if self.data is not None else 0
            eulers = np.asarray(state["eulers_rad"], dtype=np.float64).reshape(-1, 3)
            initial_eulers = (
                np.asarray(state["initial_eulers_rad"], dtype=np.float64).reshape(-1, 3)
                if "initial_eulers_rad" in state.files and np.asarray(state["initial_eulers_rad"]).size > 0
                else None
            )
            phases = np.asarray(state["phases"], dtype=np.int32).reshape(-1)
            pc_bruker = np.asarray(state["pc_bruker"], dtype=np.float64).reshape(-1, 3)
            pc_custom = np.asarray(state["pc_custom"], dtype=np.float64).reshape(-1, 3)
            if any(arr.shape[0] != expected for arr in (eulers, phases, pc_bruker, pc_custom)):
                raise ValueError("Saved workflow dimensions do not match the source data.")
            self.initial_eulers_rad = eulers.copy() if initial_eulers is None else initial_eulers.copy()
            self.current_eulers_rad = eulers.copy()
            self.current_phases = phases.copy()
            self.current_pc_bruker = pc_bruker.copy()
            self.current_pc_custom = pc_custom.copy()
            scores = np.asarray(state["scores"], dtype=np.float32)
            if self.data is not None and scores.shape == (self.data.rows, self.data.cols):
                self.last_scores_map = scores.copy()
            self.calibration_indices = np.asarray(state["calibration_indices"], dtype=np.int64).tolist()
            self.last_indexed_indices = (
                np.asarray(state["last_indexed_indices"], dtype=np.int64).reshape(-1).copy()
                if "last_indexed_indices" in state.files
                else None
            )
            dict_phase = int(state["dictionary_phase"].item())
            dict_resolution = float(state["dictionary_resolution"].item())
            dict_pc = np.asarray(state["dictionary_pc_bruker"], dtype=np.float64).reshape(3)
            dict_binning = int(state["dictionary_software_binning"].item()) if "dictionary_software_binning" in state.files else 1
            dict_crop = (
                np.asarray(state["dictionary_crop_extent"], dtype=np.int64).reshape(4)
                if "dictionary_crop_extent" in state.files
                else np.asarray(self._binning_crop_extent(dict_binning), dtype=np.int64)
            )
            if dict_phase >= 0 and np.isfinite(dict_resolution) and np.all(np.isfinite(dict_pc)):
                self.dictionary_settings = {
                    "phase_id": dict_phase,
                    "resolution_deg": dict_resolution,
                    "pc_bruker": dict_pc.copy(),
                    "software_binning": dict_binning,
                    "crop_extent": dict_crop.copy(),
                }
            residual_output_path = (
                str(state["residual_pattern_output_path"].item())
                if "residual_pattern_output_path" in state.files
                else ""
            )
            self.dictionary_cache = None
            self._invalidate_orientation_cache()
            self._invalidate_residual_cache()
            self.residual_pattern_output_path = residual_output_path or None
            self._clear_residual_pattern_source_cache()
            indexed_candidates = (
                np.asarray(state["indexed_candidate_eulers_rad"], dtype=np.float64)
                if "indexed_candidate_eulers_rad" in state.files
                else np.empty((0, 0, 3), dtype=np.float64)
            )
            if indexed_candidates.size > 0 and indexed_candidates.ndim == 3 and indexed_candidates.shape[0] == expected:
                self.indexed_candidate_eulers_rad = indexed_candidates.copy()
        return f"Restored workflow from {path}. {load_note} {master_note}"

    def export_reindexed_results(self, output_path: str) -> str:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        if self.current_eulers_rad is None or self.current_phases is None or self.current_pc_custom is None:
            raise RuntimeError("Session state is not initialized.")

        out = str(Path(output_path).expanduser().resolve())
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        if self.data.source_type == "h5oina":
            shutil.copy2(self.data.pattern_path, out)
            with h5py.File(out, "r+") as f:
                e_ds = f["1/Data Processing/Data/Euler"]
                p_ds = f["1/Data Processing/Data/Phase"]
                x_ds = f["1/Data Processing/Data/Pattern Center X"]
                y_ds = f["1/Data Processing/Data/Pattern Center Y"]
                z_ds = f["1/Data Processing/Data/Detector Distance"]

                e_shape = e_ds.shape
                p_shape = p_ds.shape
                e_ds[...] = self.current_eulers_rad.reshape(e_shape).astype(e_ds.dtype, copy=False)
                p_ds[...] = self.current_phases.reshape(p_shape).astype(p_ds.dtype, copy=False)
                pc_map = self.current_pc_custom.reshape(self.data.rows, self.data.cols, 3)
                x_ds[...] = pc_map[..., 0].reshape(x_ds.shape).astype(x_ds.dtype, copy=False)
                y_ds[...] = pc_map[..., 1].reshape(y_ds.shape).astype(y_ds.dtype, copy=False)
                z_ds[...] = pc_map[..., 2].reshape(z_ds.shape).astype(z_ds.dtype, copy=False)
            return f"Saved re-indexed H5OINA to {out}"

        if self.data.source_type == "up_ang":
            if self.data.ang_header_lines is None or self.data.orientation_path is None:
                raise RuntimeError("Missing ANG source metadata in session.")
            e_out = self.current_eulers_rad.copy()
            if self.data.ang_angles_were_degrees:
                e_out = np.rad2deg(e_out)
            phase_out = self.current_phases.astype(np.int32, copy=False)

            pc_custom_map = _convert_pc_map(
                self.current_pc_bruker.reshape(self.data.rows, self.data.cols, 3),
                src_convention="bruker",
                dst_convention=self.data.pc_output_convention,
                shape=(self.data.h, self.data.w),
                sample_tilt_deg=self.data.sample_tilt_deg,
                detector_tilt_deg=self.data.detector_tilt_deg,
                azimuthal_deg=self.data.azimuthal_deg,
                twist_deg=self.data.twist_deg,
            )
            pc_mean = np.mean(pc_custom_map.reshape(-1, 3), axis=0)

            header = list(self.data.ang_header_lines)
            def _replace_header_value(lines: list[str], key: str, value: float) -> list[str]:
                out_lines = []
                key_low = key.lower()
                replaced = False
                for line in lines:
                    if not line.startswith("#"):
                        out_lines.append(line)
                        continue
                    txt = line[1:].strip().lower()
                    if txt.startswith(key_low):
                        out_lines.append(f"# {key:<16} {value:.6f}")
                        replaced = True
                    else:
                        out_lines.append(line)
                if not replaced:
                    out_lines.append(f"# {key:<16} {value:.6f}")
                return out_lines

            header = _replace_header_value(header, "x-star", float(pc_mean[0]))
            header = _replace_header_value(header, "y-star", float(pc_mean[1]))
            header = _replace_header_value(header, "z-star", float(pc_mean[2]))

            with open(out, "w", encoding="utf-8") as f:
                for line in header:
                    f.write(line.rstrip("\n") + "\n")
                i = 0
                with open(self.data.orientation_path, "r", encoding="utf-8", errors="replace") as src:
                    for line in src:
                        if line.startswith("#") or not line.strip():
                            continue
                        vals = np.fromstring(line, dtype=np.float64, sep=" ")
                        if vals.size < 8:
                            continue
                        if i >= e_out.shape[0]:
                            break
                        vals[0:3] = e_out[i]
                        vals[7] = float(phase_out[i])
                        row = []
                        for j, v in enumerate(vals.tolist()):
                            if j == 7:
                                row.append(str(int(np.rint(v))))
                            else:
                                row.append(f"{float(v):.6f}")
                        f.write(" ".join(row) + "\n")
                        i += 1
                if i != e_out.shape[0]:
                    raise RuntimeError(
                        f"ANG export mismatch: wrote {i} rows, expected {e_out.shape[0]}."
                    )
            pc_npz = str(Path(out).with_suffix(".pc_map.npz"))
            extra = {f"pc_{self.data.pc_output_convention.lower()}": pc_custom_map}
            np.savez_compressed(
                pc_npz,
                pc_bruker=self.current_pc_bruker.reshape(self.data.rows, self.data.cols, 3),
                **extra,
            )
            return (
                f"Saved updated ANG to {out} (x/y/z-star in {self.data.pc_output_convention} convention) "
                f"and per-point PC map to {pc_npz}"
            )

        raise RuntimeError(f"Unsupported source type '{self.data.source_type}' for export.")

    def _h5oina_quality_dataset_paths(self, h5: h5py.File) -> list[str]:
        paths: list[str] = []
        for key in ("MAD", "NCC"):
            for candidate in MAP_LAYER_CANDIDATES.get(key, []):
                if candidate in h5 and candidate not in paths:
                    paths.append(candidate)
        if not paths:
            for candidate in MAP_LAYER_CANDIDATES.get("CI", []):
                if candidate in h5 and candidate not in paths:
                    paths.append(candidate)
        if not paths:
            raise RuntimeError("No suitable quality dataset (MAD/NCC/CI) found in the H5OINA file.")
        return paths

    def _export_roi_result_map(
        self,
        bounds: tuple[int, int, int, int],
        output_path: str,
        *,
        residual: bool,
        primary_ncc_threshold: float = 0.15,
    ) -> str:
        if self.data is None:
            raise RuntimeError("Load input data first.")
        r0, c0, nrows, ncols = (int(v) for v in bounds)
        if nrows <= 0 or ncols <= 0:
            raise ValueError("ROI must have a positive size.")
        rows = int(self.data.rows)
        cols = int(self.data.cols)
        if r0 < 0 or c0 < 0 or r0 >= rows or c0 >= cols:
            raise ValueError("ROI start is outside the map.")
        r1 = min(rows, r0 + nrows)
        c1 = min(cols, c0 + ncols)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("ROI does not overlap the map.")

        roi_rows = r1 - r0
        roi_cols = c1 - c0
        out = Path(output_path).expanduser().resolve()

        if self.data.source_type == "h5oina":
            if out.suffix.lower() != ".h5oina":
                out = out.with_suffix(".h5oina")
            source_path = Path(self.data.pattern_path).expanduser().resolve()
        elif self.data.source_type == "up_ang":
            if out.suffix.lower() != ".ang":
                out = out.with_suffix(".ang")
            if self.data.orientation_path is None:
                raise RuntimeError("Missing ANG source metadata in session.")
            source_path = Path(self.data.orientation_path).expanduser().resolve()
        else:
            raise RuntimeError(f"Unsupported source type '{self.data.source_type}' for export.")

        if not source_path.exists():
            raise FileNotFoundError(str(source_path))
        out.parent.mkdir(parents=True, exist_ok=True)

        if residual:
            if self.residual_eulers_rad is None or self.last_residual_scores_map is None:
                raise RuntimeError("Run residual ROI indexing before exporting the residual map.")
            if self.last_scores_map is None:
                raise RuntimeError("Primary indexing scores are required to zero residual points below the primary NCC threshold.")
            phase_source = self.residual_phases if self.residual_phases is not None else self.current_phases
            if phase_source is None:
                raise RuntimeError("Residual phase data is not available for export.")
            euler_source = np.asarray(self.residual_eulers_rad, dtype=np.float64).reshape(rows, cols, 3)
            quality_source = np.asarray(self.last_residual_scores_map, dtype=np.float64).reshape(rows, cols)
            primary_scores = np.asarray(self.last_scores_map, dtype=np.float64).reshape(rows, cols)
            phase_grid = np.asarray(phase_source, dtype=np.int32).reshape(rows, cols)
            euler_grid = euler_source.copy()
            quality_grid = np.nan_to_num(quality_source, nan=0.0).astype(np.float64, copy=False)
            phase_grid = phase_grid.copy()
            primary_roi = primary_scores[r0:r1, c0:c1]
            below_primary = ~np.isfinite(primary_roi) | (primary_roi < float(primary_ncc_threshold))
            roi_euler_block = euler_grid[r0:r1, c0:c1]
            roi_quality_block = quality_grid[r0:r1, c0:c1]
            roi_phase_block = phase_grid[r0:r1, c0:c1]
            if np.any(below_primary):
                roi_euler_block[below_primary] = 0.0
                roi_quality_block[below_primary] = 0.0
                roi_phase_block[below_primary] = 0
            missing = ~np.all(np.isfinite(roi_euler_block), axis=-1) & ~below_primary
            if np.any(missing):
                first = np.argwhere(missing)[0]
                raise RuntimeError(
                    "Residual export needs a residual orientation for every ROI point above the primary NCC threshold; "
                    f"missing at row={r0 + int(first[0])}, col={c0 + int(first[1])}."
                )
            roi_eulers = np.asarray(roi_euler_block, dtype=np.float64)
            roi_quality = np.asarray(roi_quality_block, dtype=np.float64)
            roi_phase = np.asarray(roi_phase_block, dtype=np.int32)
            zero_note = f" Points below the primary NCC threshold ({float(primary_ncc_threshold):.4f}) were exported as zero orientation."
        else:
            if self.current_eulers_rad is None or self.last_scores_map is None or self.current_phases is None:
                raise RuntimeError("Run primary ROI indexing before exporting the primary map.")
            euler_grid = np.asarray(self.current_eulers_rad, dtype=np.float64).reshape(rows, cols, 3)
            quality_grid = np.nan_to_num(np.asarray(self.last_scores_map, dtype=np.float64).reshape(rows, cols), nan=0.0)
            phase_grid = np.asarray(self.current_phases, dtype=np.int32).reshape(rows, cols)
            roi_eulers = np.asarray(euler_grid[r0:r1, c0:c1], dtype=np.float64)
            roi_quality = np.asarray(quality_grid[r0:r1, c0:c1], dtype=np.float64)
            roi_phase = np.asarray(phase_grid[r0:r1, c0:c1], dtype=np.int32)
            if not np.all(np.isfinite(roi_eulers)):
                first = np.argwhere(~np.all(np.isfinite(roi_eulers), axis=-1))[0]
                raise RuntimeError(
                    "Primary export needs a finite orientation for every ROI point; "
                    f"missing at row={r0 + int(first[0])}, col={c0 + int(first[1])}."
                )
            zero_note = ""

        if self.data.source_type == "h5oina":
            shutil.copy2(source_path, out)
            with h5py.File(out, "r+") as h5:
                e_ds = h5["1/Data Processing/Data/Euler"]
                phase_paths = [candidate for candidate in MAP_LAYER_CANDIDATES.get("Phase", []) if candidate in h5]
                quality_paths = self._h5oina_quality_dataset_paths(h5)
                row_eulers = roi_eulers.reshape(roi_rows, roi_cols, 3)
                row_quality = roi_quality.reshape(roi_rows, roi_cols)
                row_phase = roi_phase.reshape(roi_rows, roi_cols)

                def _write_h5_roi_dataset(ds: h5py.Dataset, values: np.ndarray, *, is_euler: bool) -> None:
                    if is_euler:
                        if ds.shape[:2] == (rows, cols) and len(ds.shape) >= 3 and int(ds.shape[-1]) == 3:
                            ds[r0:r1, c0:c1, :] = values.astype(ds.dtype, copy=False)
                            return
                        if len(ds.shape) == 2 and ds.shape[0] == rows * cols and int(ds.shape[1]) == 3:
                            flat_values = values.reshape(-1, 3)
                            offset = 0
                            for rr in range(roi_rows):
                                start = (r0 + rr) * cols + c0
                                stop = start + roi_cols
                                ds[start:stop, :] = flat_values[offset : offset + roi_cols].astype(ds.dtype, copy=False)
                                offset += roi_cols
                            return
                    else:
                        if ds.shape[:2] == (rows, cols):
                            if len(ds.shape) == 2:
                                ds[r0:r1, c0:c1] = values.astype(ds.dtype, copy=False)
                            else:
                                ds[r0:r1, c0:c1, ...] = values.astype(ds.dtype, copy=False)
                            return
                        if len(ds.shape) == 1 and ds.shape[0] == rows * cols:
                            flat_values = values.reshape(-1)
                            offset = 0
                            for rr in range(roi_rows):
                                start = (r0 + rr) * cols + c0
                                stop = start + roi_cols
                                ds[start:stop] = flat_values[offset : offset + roi_cols].astype(ds.dtype, copy=False)
                                offset += roi_cols
                            return
                        if len(ds.shape) == 2 and ds.shape[0] == rows * cols and ds.shape[1] == 1:
                            flat_values = values.reshape(-1)
                            offset = 0
                            for rr in range(roi_rows):
                                start = (r0 + rr) * cols + c0
                                stop = start + roi_cols
                                ds[start:stop, 0] = flat_values[offset : offset + roi_cols].astype(ds.dtype, copy=False)
                                offset += roi_cols
                            return
                    raise RuntimeError(f"Unsupported H5OINA dataset shape {ds.shape} for ROI export.")

                e_ds[...] = 0
                for q_path in quality_paths:
                    h5[q_path][...] = 0
                for p_path in phase_paths:
                    h5[p_path][...] = 0

                _write_h5_roi_dataset(e_ds, row_eulers, is_euler=True)
                for q_path in quality_paths:
                    _write_h5_roi_dataset(h5[q_path], row_quality, is_euler=False)
                for p_path in phase_paths:
                    _write_h5_roi_dataset(h5[p_path], row_phase, is_euler=False)
            return f"Exported {'residual' if residual else 'primary'} ROI map to {out}.{zero_note}"

        # ANG export: rewrite only the ROI rows while preserving the original file layout.
        if self.data.orientation_path is None:
            raise RuntimeError("Missing ANG source metadata in session.")
        eulers_to_write = roi_eulers.copy()
        if self.data.ang_angles_were_degrees:
            eulers_to_write = np.rad2deg(eulers_to_write)
        quality_to_write = np.asarray(roi_quality, dtype=np.float64)
        phase_to_write = np.asarray(roi_phase, dtype=np.int32)

        with open(source_path, "r", encoding="utf-8", errors="replace") as src, open(out, "w", encoding="utf-8") as dst:
            data_idx = 0
            for line in src:
                if line.startswith("#") or not line.strip():
                    dst.write(line if line.endswith("\n") else line + "\n")
                    continue
                vals = np.fromstring(line, dtype=np.float64, sep=" ")
                if vals.size < 8:
                    dst.write(line if line.endswith("\n") else line + "\n")
                    continue
                row = data_idx // cols
                col = data_idx % cols
                if r0 <= row < r1 and c0 <= col < c1:
                    lr = row - r0
                    lc = col - c0
                    if residual and below_primary[lr, lc]:
                        vals[0:3] = 0.0
                        vals[5] = 0.0
                        vals[6] = 0.0
                        vals[7] = 0.0
                    else:
                        vals[0:3] = eulers_to_write[lr, lc]
                        vals[6] = float(quality_to_write[lr, lc]) if np.isfinite(quality_to_write[lr, lc]) else 0.0
                        vals[7] = float(phase_to_write[lr, lc])
                else:
                    vals[0:3] = 0.0
                    vals[5] = 0.0
                    vals[6] = 0.0
                    vals[7] = 0.0
                row_out = []
                for j, v in enumerate(vals.tolist()):
                    if j == 7:
                        row_out.append(str(int(np.rint(v))))
                    else:
                        row_out.append(f"{float(v):.6f}")
                dst.write(" ".join(row_out) + "\n")
                data_idx += 1

        return f"Exported {'residual' if residual else 'primary'} ROI map to {out}.{zero_note}"

    def export_primary_roi_results(self, bounds: tuple[int, int, int, int], output_path: str) -> str:
        return self._export_roi_result_map(bounds, output_path, residual=False)

    def export_residual_roi_results(
        self,
        bounds: tuple[int, int, int, int],
        output_path: str,
        *,
        primary_ncc_threshold: float = 0.15,
    ) -> str:
        return self._export_roi_result_map(
            bounds,
            output_path,
            residual=True,
            primary_ncc_threshold=float(primary_ncc_threshold),
        )
