"""CPU optimization strategies for the primary and shared-gain mixture fits.

The search uses weighted moments instead of constructing fit/residual images for
all candidates. Float64 arithmetic also makes the local optimizer's derivatives
useful. The caller reconstructs the selected fit with the existing image model.
No gain shape or mixture fraction model is changed by these optimizations.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import OptimizeResult, differential_evolution, minimize, minimize_scalar

FIT_METHOD_DEFAULT = "staged"
FIT_METHOD_LABELS = {
    "staged": "Blur then gain (fastest)",
    "staged_joint": "Blur then gain + joint refinement",
    "joint": "Joint blur and gain",
}
DEFAULT_FIT_BOUNDS = (
    (0.1, 5.0), (-1.5, 4.5), (0.0, 12.5), (0.1, 10.0),
    (0.6, 1.4), (0.6, 1.4), (-0.15, 0.15), (-0.15, 0.15),
)
_PROJECTED_INDICES = (0, 3, 4, 5, 6, 7)
_VARIANCE_FLOOR = 1e-12


def validate_fit_method(method: str) -> str:
    if not isinstance(method, str) or method not in FIT_METHOD_LABELS:
        raise ValueError(f"Unknown fit method {method!r}; choose one of {tuple(FIT_METHOD_LABELS)}.")
    return method


def _validate_bounds(fit_bounds, name):
    try:
        bounds = [tuple(map(float, pair)) for pair in
                  (DEFAULT_FIT_BOUNDS if fit_bounds is None else fit_bounds)]
        if len(bounds) != 8 or any(len(pair) != 2 for pair in bounds):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name} fit bounds must contain eight (low, high) pairs.") from None
    if any(not np.isfinite(low) or not np.isfinite(high) or high <= low for low, high in bounds):
        raise ValueError(f"{name} fit bounds must be finite and strictly increasing.")
    return bounds


def _normalization_scale(variance):
    return np.sqrt(variance) if variance > _VARIANCE_FLOOR else 1.0


class _WeightedObjective:
    """Common active-pixel arrays, gain coordinates and a bounded exact-key cache."""

    def __init__(self, experimental, patterns, weights, cache_size=16):
        self.experimental_raw = np.asarray(experimental)
        self.patterns_raw = tuple(np.asarray(pattern) for pattern in patterns)
        self.weights_raw = np.asarray(weights)
        self.shape = self.experimental_raw.shape
        if len(self.shape) != 2 or any(p.shape != self.shape for p in self.patterns_raw) or self.weights_raw.shape != self.shape:
            raise ValueError("Experimental, simulated, and weight arrays must have matching two-dimensional shapes.")
        if any(not np.all(np.isfinite(p)) for p in (self.experimental_raw, *self.patterns_raw)):
            raise ValueError("Fit patterns must be finite.")
        if not np.all(np.isfinite(self.weights_raw)) or np.any(self.weights_raw < 0):
            raise ValueError("Fit weights must be finite and nonnegative.")
        self.total = float(np.sum(self.weights_raw, dtype=np.float64))
        self.valid = self.weights_raw.reshape(-1) > 0
        # A zero-weight fit is short-circuited by _optimize, before evaluation.
        self.w = np.asarray(self.weights_raw, dtype=np.float64).reshape(-1)[self.valid]
        if self.total > 0:
            self.w = self.w / self.total
        y = np.asarray(self.experimental_raw, dtype=np.float64).reshape(-1)[self.valid].copy()
        y -= np.dot(self.w, y)
        self.raw_y_variance = float(np.dot(self.w * y, y))
        self.y = y / _normalization_scale(self.raw_y_variance)
        self.wy = self.w * self.y
        self.yy = float(np.dot(self.wy, self.y))
        h, width = self.shape
        yy, xx = np.indices(self.shape, dtype=np.float64)
        self.y_grid = ((yy / h - 0.5) * 2).reshape(-1)[self.valid]
        self.x_grid = ((xx / width - 0.5) * 2).reshape(-1)[self.valid]
        self.patterns = tuple(np.asarray(p, dtype=np.float64) for p in self.patterns_raw)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.calls = 0
        self.blurs = 0
        self.exact_search = 0 < self.total <= _VARIANCE_FLOOR

    def _raw_blurs(self, sigma):
        sigma = float(sigma)
        if sigma in self.cache:
            self.cache.move_to_end(sigma)
            return self.cache[sigma]
        images = tuple(gaussian_filter(p, sigma=sigma).reshape(-1)[self.valid] for p in self.patterns)
        self.blurs += len(images)
        self.cache[sigma] = images
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return images

    def _gain_shape(self, power, a, b, cy, cx):
        # Match _power_gain_map, including its minimum ellipse radius.
        r2 = ((self.y_grid - 2 * cy) / max(0.1, a)) ** 2
        r2 += ((self.x_grid - 2 * cx) / max(0.1, b)) ** 2
        return (1.0 - np.minimum(np.sqrt(r2), 1.0)) ** power

    def _gain(self, params):
        gmin, gmax = params[1:3]
        return gmin + (gmax - gmin) * self._gain_shape(*params[3:8])

    def full_params(self, params):
        return np.asarray(params, dtype=np.float64)


class _PrimaryObjective(_WeightedObjective):
    """One normalized component; optionally eliminate the two gain amplitudes.

    B is the centered blur and V is centered B*q. With default gain bounds,
    every direction alpha*B + beta*V has a bounded representative, up to a
    global sign (the primary amplitude is signed NCC). Projecting onto these
    two columns therefore eliminates g_min and g_max exactly for ordinary
    nondegenerate patterns. Custom gain bounds retain the full bounded search.
    """

    def __init__(self, experimental, simulated, weights, *, projected=True, cache_size=16):
        super().__init__(experimental, (simulated,), weights, cache_size=cache_size)
        self.projected = bool(projected)
        self.centered_cache = OrderedDict()

    def _blur(self, sigma):
        sigma = float(sigma)
        if sigma in self.centered_cache:
            self.centered_cache.move_to_end(sigma)
            return self.centered_cache[sigma]
        raw, = self._raw_blurs(sigma)
        b = raw - np.dot(self.w, raw)
        variance = float(np.dot(self.w * b, b))
        b /= _normalization_scale(variance)
        wb = self.w * b
        result = (b, wb, float(np.dot(wb, b)), float(np.dot(self.wy, b)), variance)
        self.centered_cache[sigma] = result
        if len(self.centered_cache) > self.cache_size:
            self.centered_cache.popitem(last=False)
        # Retain only the centered version of this primary blur.
        self.cache.pop(sigma, None)
        return result

    def uniform_blur(self, sigma):
        self.calls += 1
        _, _, bb, yb, _ = self._blur(sigma)
        return self.total * self._moment_score(bb, yb)

    def _moment_score(self, vv, yv):
        scale = _normalization_scale(vv)
        covariance = yv / scale
        variance = vv / (scale * scale)
        ncc = covariance / _normalization_scale(self.yy) / _normalization_scale(variance)
        return max(0.0, self.yy - 2 * ncc * covariance + ncc * ncc * variance)

    def _projection(self, params):
        sigma, power, a, bscale, cy, cx = params
        b, wb, bb, yb, _ = self._blur(sigma)
        v = b * self._gain_shape(power, a, bscale, cy, cx)
        v -= np.dot(self.w, v)
        bv = float(np.dot(wb, v))
        vv = float(np.dot(self.w * v, v))
        yv = float(np.dot(self.wy, v))
        if bb <= 1e-24:
            return self.yy, 0.0, 0.0
        ratio = bv / bb
        orth_var = max(0.0, vv - bv * ratio)
        orth_y = yv - ratio * yb
        beta = orth_y / orth_var if orth_var > 1e-12 * max(vv, 1e-30) else 0.0
        alpha = yb / bb - ratio * beta
        return max(0.0, self.yy - alpha * yb - beta * yv), alpha, beta

    def __call__(self, params):
        self.calls += 1
        if self.exact_search:
            return self.exact_score(params)
        if self.projected:
            return self.total * self._projection(params)[0]
        b, _, _, _, _ = self._blur(params[0])
        candidate = b * self._gain(params)
        candidate -= np.dot(self.w, candidate)
        vv = float(np.dot(self.w * candidate, candidate))
        yv = float(np.dot(self.wy, candidate))
        return self.total * self._moment_score(vv, yv)

    def full_params(self, params):
        if not self.projected:
            return super().full_params(params)
        _, alpha, beta = self._projection(params)
        gmin, gmax = alpha, alpha + beta
        if gmax < 0:
            gmin, gmax = -gmin, -gmax
        magnitude = max(abs(gmin), abs(gmax))
        if magnitude <= 1e-30:
            gmin = gmax = 1.0
        else:
            # Select a well-scaled representative to avoid tiny float32 gains.
            gmin, gmax = gmin / magnitude, gmax / magnitude
            scales = []
            if gmin > 0:
                scales.append(DEFAULT_FIT_BOUNDS[1][1] / gmin)
            elif gmin < 0:
                scales.append(DEFAULT_FIT_BOUNDS[1][0] / gmin)
            if gmax > 0:
                scales.append(DEFAULT_FIT_BOUNDS[2][1] / gmax)
            scale = 0.9 * min(scales)
            gmin *= scale
            gmax *= scale
        return np.asarray([params[0], gmin, gmax, *params[1:]], dtype=np.float64)

    def exact_score(self, params):
        # Lazy import avoids a core import cycle. This path is used only for
        # final comparisons or unusual low-variance/near-zero-weight inputs.
        from .core import _normalize_weighted, _power_gain_map, _weighted_ncc
        y = _normalize_weighted(self.experimental_raw, self.weights_raw)
        b = _normalize_weighted(gaussian_filter(self.patterns_raw[0], sigma=float(params[0])), self.weights_raw)
        gain = _power_gain_map(self.shape, params[1:4], params[4:8])
        p = _normalize_weighted(b * gain, self.weights_raw)
        ncc = _weighted_ncc(y, p, self.weights_raw)
        # PrimaryPatternFit stores float32 residuals; compare that same image
        # when deciding whether to accept a joint refinement.
        residual = np.asarray(y - ncc * p, dtype=np.float32)
        return float(np.sum(self.weights_raw * residual * residual))


class _MixtureObjective(_WeightedObjective):
    """Shared gain on raw blurs, then separate component normalizations.

    Only the two nonnegative mixture amplitudes are projected. Fitting four
    independent gain amplitudes would change the shared-gain model and is
    deliberately avoided.
    """

    def __init__(self, experimental, primary, secondary, weights, *, cache_size=16):
        super().__init__(experimental, (primary, secondary), weights, cache_size=cache_size)

    def _score(self, sigma, gain):
        raw1, raw2 = self._raw_blurs(sigma)
        p = raw1 * gain
        s = raw2 * gain
        p -= np.dot(self.w, p)
        s -= np.dot(self.w, s)
        g11 = float(np.dot(self.w * p, p))
        g22 = float(np.dot(self.w * s, s))
        scale1 = _normalization_scale(g11)
        scale2 = _normalization_scale(g22)
        g12 = float(np.dot(self.w * p, s)) / (scale1 * scale2)
        b1 = float(np.dot(self.wy, p)) / scale1
        b2 = float(np.dot(self.wy, s)) / scale2
        g11 /= scale1 * scale1
        g22 /= scale2 * scale2
        # Match the original NNLS candidate thresholds in unnormalized weights.
        cutoff = _VARIANCE_FLOOR / self.total
        best = self.yy
        if g11 > cutoff:
            a1 = max(0.0, b1 / g11)
            best = min(best, self.yy - 2 * a1 * b1 + a1 * a1 * g11)
        if g22 > cutoff:
            a2 = max(0.0, b2 / g22)
            best = min(best, self.yy - 2 * a2 * b2 + a2 * a2 * g22)
        det = g11 * g22 - g12 * g12
        if det > _VARIANCE_FLOOR / (self.total * self.total):
            a1 = (b1 * g22 - b2 * g12) / det
            a2 = (b2 * g11 - b1 * g12) / det
            if a1 >= 0.0 and a2 >= 0.0:
                value = self.yy - 2 * (a1 * b1 + a2 * b2)
                value += a1 * a1 * g11 + 2 * a1 * a2 * g12 + a2 * a2 * g22
                best = min(best, value)
        return self.total * max(0.0, best)

    def uniform_blur(self, sigma):
        self.calls += 1
        return self._score(sigma, 1.0)

    def __call__(self, params):
        self.calls += 1
        if self.exact_search:
            return self.exact_score(params)
        return self._score(params[0], self._gain(params))

    def exact_score(self, params):
        from .core import _evaluate_overlap_mixture_pattern
        fit = _evaluate_overlap_mixture_pattern(
            self.experimental_raw, *self.patterns_raw, self.weights_raw,
            np.asarray(params, dtype=np.float64),
        )
        return float(np.sum(np.asarray(self.weights_raw, dtype=np.float32) * fit.residual * fit.residual))


def _de(objective, bounds, *, maxiter, popsize, seed):
    return differential_evolution(
        objective, bounds=bounds, maxiter=max(1, int(maxiter)),
        popsize=max(4, int(popsize)), polish=True, seed=int(seed),
        disp=False, updating="deferred", workers=1,
    )


def _optimize(objective, bounds, *, maxiter, popsize, seed, method):
    if objective.total == 0:
        params = np.asarray([(low + high) / 2 for low, high in bounds])
        return OptimizeResult(x=objective.full_params(params), fun=0.0, success=True,
                              message="All fit weights are zero; objective is zero.",
                              nfev=0, nit=0, fit_method=method, nblur=0)
    if method == "joint":
        optimization = _de(objective, bounds, maxiter=maxiter, popsize=popsize, seed=seed)
    else:
        blur = minimize_scalar(objective.uniform_blur, bounds=bounds[0], method="bounded",
                               options={"xatol": 0.005})
        # The optimum may lie at a bound; bounded scalar minimization never
        # actually evaluates either endpoint.
        sigma = min((float(blur.x), *bounds[0]), key=objective.uniform_blur)
        gain = _de(lambda params: objective(np.asarray([sigma, *params])), bounds[1:],
                   maxiter=maxiter, popsize=popsize, seed=seed)
        optimization = OptimizeResult(gain)
        optimization.x = np.asarray([sigma, *gain.x])
        optimization.message = f"Blur then gain: {gain.message}"
        if method == "staged_joint":
            original_x = optimization.x.copy()
            local = minimize(objective, original_x, method="Powell", bounds=bounds,
                             options={"maxiter": min(30, max(1, int(maxiter))),
                                      "maxfev": 1500, "xtol": 1e-3, "ftol": 1e-7})
            # Compare actual reconstructed float32 residuals as well: a tiny
            # surrogate improvement must not make the exported fit worse.
            before = objective.exact_score(objective.full_params(original_x))
            after = objective.exact_score(objective.full_params(local.x))
            if np.isfinite(after) and after < before:
                optimization = OptimizeResult(local)
                optimization.message = f"Blur then gain with joint refinement: {local.message}"
            else:
                optimization.message = f"Blur then gain; joint refinement retained the staged solution. {gain.message}"
    optimization.x = objective.full_params(optimization.x)
    optimization.fun = objective.exact_score(optimization.x)
    optimization.nfev = objective.calls
    optimization.nblur = objective.blurs
    optimization.fit_method = method
    return optimization


def optimize_primary(experimental, simulated, weights, *, maxiter, popsize, seed,
                     fit_bounds=None, method=FIT_METHOD_DEFAULT):
    """Optimize Step 3 and return eight original model parameters.

    With the default gain bounds, the two gain amplitudes are solved by a
    two-column projection. Restricted custom gain bounds use a full bounded
    objective instead, so every returned parameter respects the user's bounds.
    """
    method = validate_fit_method(method)
    bounds = _validate_bounds(fit_bounds, "Primary")
    projected = bounds[1:3] == list(DEFAULT_FIT_BOUNDS[1:3])
    objective = _PrimaryObjective(experimental, simulated, weights, projected=projected)
    # At normalization's low-variance threshold gain scale stops cancelling.
    # Keep the exact original objective for these atypical cases. Checking the
    # blur extremes also catches tiny/high-frequency simulated patterns.
    if objective.total > 0:
        variances = [objective._blur(sigma)[4] for sigma in (bounds[0][0], bounds[0][1])]
        if objective.raw_y_variance <= _VARIANCE_FLOOR or min(variances) <= 1e-10:
            objective.exact_search = True
        if objective.exact_search:
            objective.projected = False
    nonlinear_bounds = [bounds[i] for i in _PROJECTED_INDICES] if objective.projected else bounds
    return _optimize(objective, nonlinear_bounds, maxiter=maxiter, popsize=popsize,
                     seed=seed, method=method)


def optimize_mixture(experimental, primary, secondary, weights, *, maxiter, popsize, seed,
                     fit_bounds=None, method=FIT_METHOD_DEFAULT):
    """Optimize Step 4 with shared blur/gain and nonnegative mixture amplitudes."""
    method = validate_fit_method(method)
    bounds = _validate_bounds(fit_bounds, "Mixture")
    objective = _MixtureObjective(experimental, primary, secondary, weights)
    return _optimize(objective, bounds, maxiter=maxiter, popsize=popsize, seed=seed, method=method)
