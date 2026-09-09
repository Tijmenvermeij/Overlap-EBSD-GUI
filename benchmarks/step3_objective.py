"""Investigative Step 3 benchmark; does not change the application fitter.

Run from the repository root with a prepared NPZ containing experimental and
simulated arrays (N,H,W), weights (H,W), and indices (N,):
  .venv/bin/python benchmarks/step3_objective.py /tmp/step3_real_pairs.npz
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core
from benchmarks.cpu_workflow_validation import reference_fitter

BOUNDS = [(0.1, 5), (-1.5, 4.5), (0, 12.5), (0.1, 10),
          (0.6, 1.4), (0.6, 1.4), (-0.15, 0.15), (-0.15, 0.15)]


def original_objective(experimental_raw, simulated_raw, weights):
    experimental = core._normalize_weighted(experimental_raw, weights)

    def objective(params):
        blurred = core._normalize_weighted(
            gaussian_filter(simulated_raw, sigma=float(params[0])), weights)
        gain = core._power_gain_map(experimental.shape, params[1:4], params[4:8])
        processed = core._normalize_weighted(blurred * gain, weights)
        score = core._weighted_ncc(experimental, processed, weights)
        residual = experimental - score * processed
        return float(np.sum(weights * residual * residual))

    return objective


def lean_objective(experimental_raw, simulated_raw, weights):
    """Same nondegenerate objective in real arithmetic; score from moments.

    This is an investigation prototype, not a replacement with the application's
    complete low-variance/zero-weight behavior. Uses float64 reductions to avoid
    loss of precision when subtracting means. Avoids full residual construction
    and repeated z-score normalizations for each candidate.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    w = w / total
    experimental = np.array(experimental_raw, dtype=np.float64, copy=True)
    experimental -= np.sum(w * experimental)
    experimental /= np.sqrt(np.sum(w * experimental**2))
    wy = w * experimental
    h, width = experimental.shape
    yy, xx = np.ogrid[:h, :width]

    def objective(params):
        sigma, gmin, gmax, power, a, b, cy, cx = params
        blurred = gaussian_filter(simulated_raw, sigma=float(sigma))
        # Standard deviation cancels after multiplying by the gain and
        # normalizing again. Subtracting the blurred mean remains necessary.
        centered = blurred - np.sum(w * blurred)
        radius = np.sqrt(((yy - h/2 - cy*h) / (a*h/2))**2
                         + ((xx - width/2 - cx*width) / (b*width/2))**2)
        gain = gmin + (1 - np.clip(radius, 0, 1))**power * (gmax-gmin)
        candidate = centered * gain
        mean = np.sum(w * candidate)
        variance = np.sum(w * candidate**2) - mean**2
        if variance <= 1e-12:
            return float(total)
        covariance = np.sum(wy * candidate)
        return float(total * max(0.0, 1 - covariance**2 / variance))

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/tmp/step3_objective_results.json'))
    parser.add_argument('--profile', type=Path, default=Path('/tmp/step3_profile.txt'))
    args = parser.parse_args()
    records = []
    baseline_fit = reference_fitter('_fit_overlap_primary_pattern')
    data = np.load(args.pairs)
    weights = data['weights']
    rng = np.random.default_rng(812)
    bounds = np.asarray(BOUNDS)
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], (500, 8))
    for number, (idx, exp, sim) in enumerate(zip(data['indices'], data['experimental'], data['simulated'])):
        original = original_objective(exp, sim, weights)
        lean = lean_objective(exp, sim, weights)
        micro = {}
        scores = {}
        for name, func in [('original', original), ('lean', lean)]:
            func(candidates[0])
            durations = []
            for _ in range(3):
                start = perf_counter()
                scores[name] = np.asarray([func(p) for p in candidates])
                durations.append((perf_counter() - start) / len(candidates))
            micro[name + '_seconds_per_call'] = float(np.median(durations))
        micro['max_objective_difference'] = float(np.max(np.abs(scores['original'] - scores['lean'])))
        print('micro', int(idx), micro, flush=True)
        for maxiter, popsize in [(80, 15), (40, 8)]:
            captured = {}

            def instrument(*a, **kw):
                result = differential_evolution(*a, **kw)
                captured.update(nfev=result.nfev, nit=result.nit, objective=float(result.fun))
                return result

            start = perf_counter()
            with patch.dict(baseline_fit.__globals__, differential_evolution=instrument):
                fit = baseline_fit(exp, sim, weights, maxiter=maxiter,
                                                       popsize=popsize, seed=int(idx))
            record = dict(index=int(idx), maxiter=maxiter, popsize=popsize,
                          seconds=perf_counter()-start, ncc=fit.ncc_fitted,
                          sigma=fit.sigma, success=fit.success, **captured, **micro)
            records.append(record)
            print(json.dumps(record), flush=True)
        if number == 0:
            profiler = cProfile.Profile()
            profiler.enable()
            baseline_fit(exp, sim, weights, maxiter=80, popsize=15, seed=int(idx))
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumtime').print_stats(30)
            args.profile.write_text(stream.getvalue())
    args.output.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
