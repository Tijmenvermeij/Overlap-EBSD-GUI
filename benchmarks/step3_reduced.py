#!/usr/bin/env python3
"""Investigation-only Step 3 optimizer comparison; no production edits.

Run with the project interpreter, e.g.:
  .venv/bin/python benchmarks/step3_reduced.py --pairs /tmp/step3_real_pairs.npz

The analytic gain elimination is valid for the DEFAULT gain bounds only.
Those bounds contain arbitrarily small representatives of every gain direction;
custom bounds can restrict those directions and require a constrained solver.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution, minimize, minimize_scalar

BOUNDS = [(.1, 5.), (-1.5, 4.5), (0., 12.5), (.1, 10.), (.6, 1.4), (.6, 1.4), (-.15, .15), (-.15, .15)]
REDUCED_BOUNDS = [BOUNDS[i] for i in [0, 3, 4, 5, 6, 7]]


def original_functions():
    """Load the pre-optimization fitter before starting any timers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from multistep_overlap_ebsd import core
    from benchmarks.cpu_workflow_validation import reference_fitter
    names = {'_normalize_weighted', '_weighted_ncc', '_power_gain_map', '_fit_overlap_primary_pattern'}
    functions = {name: getattr(core, name) for name in names}
    functions['_fit_overlap_primary_pattern'] = reference_fitter('_fit_overlap_primary_pattern')
    return functions


class ProjectedObjective:
    """Six nonlinear dimensions, weighted two-column least squares for gain.

    B is the centered blurred pattern; q=(1-r)**p; the model is
    a*B + (b-a)*(B*q - weighted_mean(B*q)). Fitting these two columns
    simultaneously eliminates both gain values and final amplitude exactly.
    Float64 arithmetic gives local optimization a smooth objective. Production
    code reconstructs the exported residual with float32, compared separately.
    """
    def __init__(self, experimental, simulated, weights, cache_size=32):
        if experimental.shape != simulated.shape or experimental.shape != weights.shape:
            raise ValueError('Experimental, simulated, and weight arrays must have matching shapes.')
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError('This prototype requires finite nonnegative weights with positive sum.')
        if not np.all(np.isfinite(experimental)) or not np.all(np.isfinite(simulated)):
            raise ValueError('This prototype requires finite patterns.')
        self.shape = experimental.shape
        self.sim = np.asarray(simulated, dtype=np.float64)
        self.valid = np.asarray(weights).reshape(-1) > 0
        self.w = np.asarray(weights, dtype=np.float64).reshape(-1)[self.valid]
        self.w /= self.w.sum()
        y = np.asarray(experimental, dtype=np.float64).reshape(-1)[self.valid]
        y -= np.dot(self.w, y)
        y_var = np.dot(self.w, y*y)
        if y_var <= 1e-12:
            raise ValueError('This prototype requires a nonconstant experimental pattern.')
        self.y = y / np.sqrt(y_var)
        self.yy = float(np.dot(self.w, self.y*self.y))
        self.wy = self.w*self.y
        h, w = self.shape
        yy, xx = np.indices(self.shape, dtype=np.float64)
        self.yy_grid = ((yy/h-.5)*2).reshape(-1)[self.valid]
        self.xx_grid = ((xx/w-.5)*2).reshape(-1)[self.valid]
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.calls = 0
        self.blurs = 0

    def blurred(self, sigma):
        sigma = float(sigma)
        if sigma in self.cache:
            self.cache.move_to_end(sigma)
            return self.cache[sigma]
        full = gaussian_filter(self.sim, sigma=sigma)
        b = full.reshape(-1)[self.valid]
        b -= np.dot(self.w, b)
        wb = self.w*b
        entry = (b, wb, float(np.dot(wb, b)), float(np.dot(self.wy, b)))
        self.blurs += 1
        self.cache[sigma] = entry
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return entry

    def evaluate(self, params, details=False):
        self.calls += 1
        sigma, power, a_scale, b_scale, cy, cx = params
        b, wb, bb, yb = self.blurred(sigma)
        r2 = ((self.yy_grid-2*cy)/a_scale)**2 + ((self.xx_grid-2*cx)/b_scale)**2
        q = (1-np.minimum(np.sqrt(r2), 1.))**power
        v = b*q
        v -= np.dot(self.w, v)
        bv = float(np.dot(wb, v))
        vv = float(np.dot(self.w*v, v))
        yv = float(np.dot(self.wy, v))
        if bb <= 1e-24:
            alpha, beta, ssr = 0., 0., self.yy
        else:
            ratio = bv/bb
            orth_var = max(0., vv-bv*ratio)
            orth_y = yv-ratio*yb
            beta = orth_y/orth_var if orth_var > 1e-12*max(vv, 1e-30) else 0.
            alpha = yb/bb-ratio*beta
            ssr = max(0., self.yy-alpha*yb-beta*yv)
        if not details:
            return ssr
        gain_min, gain_max = alpha, alpha+beta
        if gain_max < 0:
            gain_min, gain_max = -gain_min, -gain_max
        shrink = 1.
        if gain_min > 0: shrink = min(shrink, 4.5/gain_min)
        if gain_min < 0: shrink = min(shrink, 1.5/abs(gain_min))
        if gain_max > 0: shrink = min(shrink, 12.5/gain_max)
        gain_min *= .9*shrink
        gain_max *= .9*shrink
        full_params = [sigma, gain_min, gain_max, power, a_scale, b_scale, cy, cx]
        return ssr, full_params, self.y-alpha*b-beta*v

    __call__ = evaluate


def score_reconstruction(ns, experimental, simulated, weights, params):
    norm = ns['_normalize_weighted']
    y = norm(experimental, weights)
    b = norm(gaussian_filter(simulated, sigma=params[0]), weights)
    g = ns['_power_gain_map'](y.shape, params[1:4], params[4:8])
    p = norm(b*g, weights)
    ncc = ns['_weighted_ncc'](y, p, weights)
    residual = y-ncc*p
    return float(np.sum(weights*residual*residual)), float(ncc), residual


def attach_residual_comparisons(ns, experimental, simulated, weights, results):
    """Add cheap spatial-quality checks from fitted parameters; no new fitting."""
    baseline = next(item for item in results if item['method'] == 'current8')
    _, _, reference = score_reconstruction(ns, experimental, simulated, weights, baseline['params'])
    wn = np.array(weights, dtype=np.float64, copy=True)
    wn /= wn.sum()
    reference_rms = float(np.sqrt(np.sum(wn*reference*reference)))
    for item in results:
        _, _, residual = score_reconstruction(ns, experimental, simulated, weights, item['params'])
        difference = np.asarray(residual, dtype=np.float64)-reference
        difference_rms = float(np.sqrt(np.sum(wn*difference*difference)))
        item['residual_rms_difference_from_current'] = difference_rms
        item['residual_relative_rms_difference_from_current'] = difference_rms/reference_rms
        item['residual_ncc_with_current'] = float(ns['_weighted_ncc'](residual, reference, weights))
    return results


def run_case(ns, experimental, simulated, weights, label, maxiter, popsize, seed):
    results = []
    start = time.perf_counter()
    fit = ns['_fit_overlap_primary_pattern'](experimental, simulated, weights, maxiter=maxiter, popsize=popsize, seed=seed)
    results.append(dict(case=label, method='current8', seconds=time.perf_counter()-start,
                        ssr=float(np.sum(weights*fit.residual*fit.residual)), ncc=fit.ncc_fitted,
                        sigma=fit.sigma, params=[fit.sigma, *fit.gain_params, *fit.ellipse_params], success=fit.success))
    print(json.dumps(results[-1]), flush=True)

    projected = ProjectedObjective(experimental, simulated, weights)
    start = time.perf_counter()
    joint = differential_evolution(projected, REDUCED_BOUNDS, maxiter=maxiter, popsize=popsize,
                                   seed=seed, polish=True, updating='deferred', workers=1)
    seconds = time.perf_counter()-start
    projected_ssr, params, _ = projected.evaluate(joint.x, details=True)
    ssr, ncc, _ = score_reconstruction(ns, experimental, simulated, weights, params)
    results.append(dict(case=label, method='projected6', seconds=seconds, ssr=ssr, ncc=ncc,
                        projected_ssr=projected_ssr, sigma=params[0], params=params,
                        evaluations=projected.calls, blurs=projected.blurs, success=bool(joint.success)))
    print(json.dumps(results[-1]), flush=True)

    staged = ProjectedObjective(experimental, simulated, weights)
    start = time.perf_counter()
    def uniform_blur(sigma):
        _, _, bb, yb = staged.blurred(sigma)
        return staged.yy-yb*yb/bb if bb > 1e-24 else staged.yy
    sigma_fit = minimize_scalar(uniform_blur, bounds=BOUNDS[0], method='bounded', options={'xatol': .005})
    sigma = float(sigma_fit.x)
    gain = differential_evolution(lambda g: staged([sigma, *g]), REDUCED_BOUNDS[1:],
                                  maxiter=maxiter, popsize=popsize, seed=seed, polish=True,
                                  updating='deferred', workers=1)
    x_staged = np.array([sigma, *gain.x])
    stage_seconds = time.perf_counter()-start
    projected_ssr, params, _ = staged.evaluate(x_staged, details=True)
    ssr, ncc, _ = score_reconstruction(ns, experimental, simulated, weights, params)
    results.append(dict(case=label, method='blur_then_gain5', seconds=stage_seconds,
                        ssr=ssr, ncc=ncc, projected_ssr=projected_ssr, sigma=params[0], params=params,
                        evaluations=staged.calls, blurs=staged.blurs, success=bool(gain.success)))
    print(json.dumps(results[-1]), flush=True)

    start = time.perf_counter()
    local = minimize(staged, x_staged, method='Powell', bounds=REDUCED_BOUNDS,
                     options={'maxiter': 30, 'maxfev': 1500, 'xtol': 1e-3, 'ftol': 1e-7})
    polish_seconds = time.perf_counter()-start
    # Preserve the better point when a bounded local run terminates early.
    x_final = local.x if local.fun < staged(x_staged) else x_staged
    projected_ssr, params, _ = staged.evaluate(x_final, details=True)
    ssr, ncc, _ = score_reconstruction(ns, experimental, simulated, weights, params)
    results.append(dict(case=label, method='staged_plus_joint6', seconds=stage_seconds+polish_seconds,
                        ssr=ssr, ncc=ncc, projected_ssr=projected_ssr, sigma=params[0], params=params,
                        evaluations=staged.calls, blurs=staged.blurs, success=bool(local.success)))
    print(json.dumps(results[-1]), flush=True)
    return attach_residual_comparisons(ns, experimental, simulated, weights, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/step3_reduced_results.json'))
    parser.add_argument('--maxiter', type=int, default=80)
    parser.add_argument('--popsize', type=int, default=15)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    ns = original_functions()
    data = np.load(args.pairs)
    weights = data['weights']
    all_results = []
    for k in range(min(args.limit, len(data['experimental']))):
        all_results += run_case(ns, data['experimental'][k], data['simulated'][k], weights,
                                f"real_{int(data['indices'][k])}", args.maxiter, args.popsize, int(data['indices'][k]))
    sim = data['simulated'][0]
    norm = ns['_normalize_weighted']
    planted = [1.6, .35, 3.2, 2., 1.15, .8, .08, -.06]
    primary = norm(norm(gaussian_filter(sim, sigma=planted[0]), weights)
                   * ns['_power_gain_map'](sim.shape, planted[1:4], planted[4:8]), weights)
    secondary = norm(data['simulated'][-1], weights)
    noise = np.random.default_rng(917).normal(0., .03, size=sim.shape).astype(np.float32)
    synthetic = (primary+.4*secondary+noise).astype(np.float32)
    all_results += run_case(ns, synthetic, sim, weights, 'synthetic_known_blur_gain_plus_secondary',
                            args.maxiter, args.popsize, 917)
    args.output.write_text(json.dumps(dict(results=all_results, synthetic_params=planted,
                                          synthetic_secondary_amplitude=.4, maxiter=args.maxiter,
                                          popsize=args.popsize), indent=2)+'\n')


if __name__ == '__main__':
    main()
