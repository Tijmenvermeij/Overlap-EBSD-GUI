"""Full-scan H5OINA component exports from saved optimized mixture parameters."""
from copy import copy
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


def full_range_pattern(pattern, dtype, mask=None):
    values = np.asarray(pattern, dtype=np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    out = np.zeros(values.shape, dtype=dtype)
    if valid.any():
        low, high = float(values[valid].min()), float(values[valid].max())
        if high > low:
            out[valid] = np.rint(np.clip((values[valid]-low)/(high-low), 0, 1) * np.iinfo(dtype).max).astype(dtype)
    return out


def export_fitted_patterns(session, output_path, *, secondary, accepted_only,
                           primary_subtract_residual, progress_callback=None):
    from .core import (_output_path_with_single_suffix, _normalize_weighted, _power_gain_map,
                       _weighted_ncc, _read_h5_patterns, H5OINA_EXPORT_DATA_GROUP, H5OINA_NCC_DATASET)
    if session.data is None or not session.overlap_mixture_results:
        raise RuntimeError('Complete at least one mixture fit in tab 4 before exporting fitted patterns.')
    if not getattr(session.data, 'patterns_available', True):
        raise RuntimeError('Load the measured patterns before exporting fitted patterns.')
    out = _output_path_with_single_suffix(output_path, '.h5oina').resolve()
    for source in (session.data.pattern_path, session.residual_pattern_output_path):
        if source and (out == Path(source).resolve() or (out.exists() and Path(source).exists() and os.path.samefile(out, source))):
            raise ValueError('Choose an export filename different from the input and residual working files.')
    data = session.data
    count, rows, cols = data.count, data.rows, data.cols
    # _pattern_at converts H5 input to float32 for computation; inspect the
    # stored dataset to preserve 16-bit acquisition precision on export.
    source = session._direct_h5_pattern_source()
    if source is not None:
        with h5py.File(source[1], 'r') as h5:
            source_dtype = h5[source[2]].dtype
    else:
        source_dtype = np.asarray(session._pattern_at(0)).dtype
    dtype = np.dtype(np.uint16 if source_dtype == np.uint16 else np.uint8)
    weights = session._overlap_weights()
    mask = weights > 0
    overlap = np.zeros(count, dtype=bool)
    for index, result in session.overlap_mixture_results.items():
        if not 0 <= index < count:
            continue
        eligible = bool(result.overlap_accepted) if accepted_only else True
        if eligible and result.primary_euler_rad is not None and result.secondary_euler_rad is not None:
            if not (np.isfinite(result.primary_euler_rad).all() and np.isfinite(result.secondary_euler_rad).all()
                    and np.isfinite([result.primary_coefficient, result.secondary_coefficient]).all()):
                raise ValueError(f'Point {index} has invalid optimized fit parameters.')
            overlap[index] = True
    view = copy(session)
    view._borrowed_phase_context = True
    view.current_eulers_rad = np.zeros((count, 3)) if secondary else session.current_eulers_rad.copy()
    view.current_phases = np.zeros(count, dtype=np.int32) if secondary else session.current_phases.copy()
    view.last_scores_map = np.zeros((rows, cols), dtype=np.float32) if secondary else session.last_scores_map.copy()
    view.indexed_mask = np.ones(count, dtype=bool)
    for index in np.flatnonzero(overlap):
        result = session.overlap_mixture_results[int(index)]
        key = result.secondary_phase_key if secondary else result.primary_phase_key
        view.current_eulers_rad[index] = result.secondary_euler_rad if secondary else result.primary_euler_rad
        if key:
            view.current_phases[index] = session.phase_registry.by_key(key).output_id
        else:
            view.current_phases[index] = session.residual_phases[index] if secondary else session.current_phases[index]
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.'+out.stem+'-', suffix='.h5oina', dir=out.parent)
    os.close(handle)
    temporary = Path(name)
    contexts = {}
    def simulate(index, key, angle, is_secondary):
        if key and session.phase_masters:
            if key not in contexts:
                contexts[key] = session._phase_context(key, private_arrays=False)
            return contexts[key]._simulate_pattern_for_euler(index, angle)
        method = session._simulate_secondary_pattern if is_secondary else session._simulate_pattern_for_euler
        return method(index, angle)
    try:
        view._export_roi_result_map((0, 0, rows, cols), str(temporary), residual=False,
            progress_callback=(lambda v,m: progress_callback(v*.1,m)) if progress_callback else None)
        with h5py.File(temporary, 'r+') as h5, ExitStack() as inputs:
            original = inputs.enter_context(h5py.File(source[1], 'r'))[source[2]] if source is not None else None
            cached_start, cached_patterns = -1, None
            batch_size = max(1, min(64, (32*1024**2)//max(1, data.h*data.w*4)))
            def raw_pattern(index):
                nonlocal cached_start, cached_patterns
                if original is None:
                    return session._pattern_at(index)
                if cached_patterns is None or not cached_start <= index < cached_start+len(cached_patterns):
                    cached_start = index
                    cached_patterns = _read_h5_patterns(original, np.arange(index,min(count,index+batch_size)), rows=rows,cols=cols)
                return cached_patterns[index-cached_start]
            root = data.h5_analysis_root if data.h5_analysis_root is not None else '1'
            prefix = root+'/' if root else ''
            parent = h5.require_group(prefix+'EBSD/Data')
            if 'Processed Patterns' in parent:
                del parent['Processed Patterns']
            patterns = parent.create_dataset('Processed Patterns', shape=(count, data.h, data.w), dtype=dtype,
                chunks=(1, data.h, data.w), compression='lzf', fillvalue=0)
            quality = view.last_scores_map.reshape(-1)
            for index in range(count):
                if progress_callback and index % 32 == 0:
                    progress_callback(10+90*index/count, f'Exporting fitted patterns {index}/{count}…')
                if not overlap[index]:
                    if not secondary:
                        patterns[index] = full_range_pattern(raw_pattern(index), dtype)
                    continue
                result = session.overlap_mixture_results[index]
                experimental = _normalize_weighted(session._apply_dynamic_background_to_patterns(raw_pattern(index)), weights)
                gain = _power_gain_map(experimental.shape, result.gain_params, result.ellipse_params)
                def component(key, angle, is_secondary):
                    raw = simulate(index, key, angle, is_secondary)
                    return _normalize_weighted(gaussian_filter(raw, sigma=result.fitted_sigma)*gain, weights)
                primary = component(result.primary_phase_key, result.primary_euler_rad, False)
                residual = component(result.secondary_phase_key, result.secondary_euler_rad, True)
                if secondary:
                    isolated = experimental - result.primary_coefficient*primary
                    target = residual
                elif primary_subtract_residual:
                    isolated = result.primary_coefficient*primary
                    target = primary
                else:
                    isolated = experimental - result.secondary_coefficient*residual
                    target = primary
                patterns[index] = full_range_pattern(isolated, dtype, mask)
                quality[index] = _weighted_ncc(isolated, target, weights)
            h5[prefix+H5OINA_NCC_DATASET][...] = quality.reshape(h5[prefix+H5OINA_NCC_DATASET].shape)
            # Parent-run residual/mixture metadata must not describe these new
            # component patterns. The phase catalogue lives separately.
            group_path = prefix+H5OINA_EXPORT_DATA_GROUP
            if group_path in h5:
                del h5[group_path]
            group = h5.require_group(group_path)
            group.attrs['Export Type'] = 'optimized residual' if secondary else 'optimized primary'
            group.attrs['Pattern Scaling'] = f'Per-pattern min/max to 0–{np.iinfo(dtype).max}; constant patterns are zero'
            group.attrs['Overlap Selection'] = 'accepted fits' if accepted_only else 'all completed fits'
            group.attrs['Pattern Formula'] = ('E - a1*S1' if secondary else 'a1*S1' if primary_subtract_residual else 'E - a2*S2')
            group.attrs['Primary Patterns Included'] = np.uint8(not secondary)
            group.attrs['Residual Patterns Included'] = np.uint8(secondary)
            datasets = [('Optimized Overlap Mask', overlap.reshape(rows, cols).astype(np.uint8)),
                        ('ROI Mask', np.ones((rows, cols), dtype=np.uint8)),
                        ('Residual NCC' if secondary else 'Primary NCC', view.last_scores_map)]
            if secondary:
                datasets.append(('Residual Pattern Available', overlap.reshape(rows, cols).astype(np.uint8)))
            for key, array in datasets:
                if key in group: del group[key]
                group.create_dataset(key, data=array)
        if progress_callback:
            progress_callback(100, 'Fitted-pattern H5OINA export complete.')
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return f'Exported {count} {"residual" if secondary else "primary"} solutions and patterns; {overlap.sum()} overlap pixels: {out}'
