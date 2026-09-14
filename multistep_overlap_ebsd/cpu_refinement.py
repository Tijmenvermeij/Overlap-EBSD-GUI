"""Compiled orientation refinement using Kikuchipy's projection and NCC kernels.

The simplex update rules follow SciPy's non-adaptive Nelder-Mead implementation.
See third_party/SCIPY_LICENSE.txt. Unsupported solver options use SciPy directly.

Keep fastmath disabled in the simplex arithmetic: small reassociations can
change which candidate wins. Projection and NCC use Kikuchipy's own kernels.
"""
import numpy as np
from numba import njit
from scipy.optimize import OptimizeResult, minimize

from kikuchipy._utils.numba import rotation_from_euler
from kikuchipy.indexing._refinement._objective_functions import _refine_orientation_objective_function
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import _ncc_single_patterns_1d_float32_exp_centered
from kikuchipy.signals.util._master_pattern import _project_single_pattern_from_master_pattern


@njit(cache=True, nogil=True)
def _orientation_objective(x, parameters):
    simulated = _project_single_pattern_from_master_pattern(
        rotation_from_euler(x[0], x[1], x[2]), parameters[1], parameters[2], parameters[3],
        parameters[4], parameters[5], parameters[6], False, 0, 1, np.float32,
    )
    return 1.0 - np.float64(_ncc_single_patterns_1d_float32_exp_centered(
        parameters[0], simulated, parameters[7]))


@njit(cache=True, nogil=True)
def _nelder_mead(objective, x0, parameters, lower, upper, maxfev, maxiter, xatol, fatol):
    """Bounded simplex search, including SciPy's evaluation-budget semantics."""
    dimensions = len(x0)
    simplex = np.empty((dimensions + 1, dimensions), dtype=np.float64)
    start = np.minimum(np.maximum(x0, lower), upper)
    for row in range(dimensions + 1):
        simplex[row] = start
        if row:
            col = row - 1
            simplex[row, col] = 1.05 * start[col] if start[col] != 0 else .00025
        for col in range(dimensions):
            if simplex[row, col] > upper[col]:
                simplex[row, col] = 2 * upper[col] - simplex[row, col]
            simplex[row, col] = min(max(simplex[row, col], lower[col]), upper[col])
    values = np.full(dimensions + 1, np.inf)
    evaluations = 0
    for row in range(dimensions + 1):
        if evaluations >= maxfev:
            break
        values[row] = objective(simplex[row], parameters)
        evaluations += 1
    # SciPy sorts twice after evaluating the initial simplex.
    for _ in range(2):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
    iterations = 1
    while evaluations < maxfev and iterations < maxiter:
        if np.max(np.abs(simplex[1:] - simplex[0])) <= xatol and np.max(np.abs(values[0] - values[1:])) <= fatol:
            break
        center = np.zeros(dimensions)
        for row in range(dimensions):
            center += simplex[row]
        center /= dimensions
        reflected = np.minimum(np.maximum(2 * center - simplex[-1], lower), upper)
        reflected_value = objective(reflected, parameters)
        evaluations += 1
        shrink = False
        if reflected_value < values[0]:
            if evaluations >= maxfev:
                break
            expanded = np.minimum(np.maximum(3 * center - 2 * simplex[-1], lower), upper)
            expanded_value = objective(expanded, parameters)
            evaluations += 1
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        else:
            if evaluations >= maxfev:
                break
            if reflected_value < values[-1]:
                contracted = np.minimum(np.maximum(1.5 * center - .5 * simplex[-1], lower), upper)
                contracted_value = objective(contracted, parameters)
                evaluations += 1
                if contracted_value <= reflected_value:
                    simplex[-1], values[-1] = contracted, contracted_value
                else:
                    shrink = True
            else:
                contracted = np.minimum(np.maximum(.5 * center + .5 * simplex[-1], lower), upper)
                contracted_value = objective(contracted, parameters)
                evaluations += 1
                if contracted_value < values[-1]:
                    simplex[-1], values[-1] = contracted, contracted_value
                else:
                    shrink = True
        interrupted = False
        if shrink:
            for row in range(1, dimensions + 1):
                simplex[row] = np.minimum(np.maximum(simplex[0] + .5 * (simplex[row] - simplex[0]), lower), upper)
                # SciPy updates the vertex before checking the function budget.
                if evaluations >= maxfev:
                    interrupted = True
                    break
                values[row] = objective(simplex[row], parameters)
                evaluations += 1
        if not interrupted:
            iterations += 1
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if interrupted:
            break
    order = np.argsort(values)
    simplex, values = simplex[order], values[order]
    return simplex, values, evaluations, iterations


def minimize_orientation(fun, x0, args=(), *, bounds=None, callback=None, **kwargs):
    """SciPy custom-method adapter for the application's orientation-only fits."""
    ignored = {'jac', 'hess', 'hessp', 'constraints'}
    options = {name: value for name, value in kwargs.items() if name not in ignored}
    allowed = {'maxfev', 'maxiter', 'xatol', 'fatol', 'disp'}
    start = np.asarray(x0, dtype=np.float64)
    if bounds is None:
        lower, upper = np.full(start.size, -np.inf), np.full(start.size, np.inf)
    elif hasattr(bounds, 'lb'):
        lower, upper = np.asarray(bounds.lb), np.asarray(bounds.ub)
    else:
        lower, upper = np.asarray(bounds, dtype=np.float64).T
    # The compiled loop indexes every bound explicitly. Reject shapes outside
    # Kikuchipy's three-Euler contract rather than relying on NumPy broadcasting.
    supported = (
        fun is _refine_orientation_objective_function and start.shape == (3,) and len(args) == 8
        and lower.shape == (3,) and upper.shape == (3,)
        and set(options).issubset(allowed) and not callback and not options.get('disp', False)
        and not kwargs.get('constraints') and np.isfinite(start).all()
        and np.all(lower <= start) and np.all(start <= upper)
        and np.isfinite(args[-1]) and args[-1] > 0
    )
    if not supported:
        return minimize(fun, x0, args=args, method='Nelder-Mead', bounds=bounds,
                        callback=callback, options=options)
    maxfev, maxiter = options.get('maxfev'), options.get('maxiter')
    if maxfev is None and maxiter is None:
        maxfev = maxiter = 600
    elif maxiter is None:
        maxiter = 600 if maxfev == np.inf else np.inf
    elif maxfev is None:
        maxfev = 600 if maxiter == np.inf else np.inf
    maxfev = 2**31-1 if maxfev is None or maxfev == np.inf else int(maxfev)
    maxiter = 2**31-1 if maxiter is None or maxiter == np.inf else int(maxiter)
    simplex, values, nfev, nit = _nelder_mead(
        _orientation_objective, start, args, lower, upper, maxfev, maxiter,
        float(options.get('xatol', 1e-4)), float(options.get('fatol', 1e-4)),
    )
    status = 1 if nfev >= maxfev else 2 if nit >= maxiter else 0
    message = ('Optimization terminated successfully.',
               'Maximum number of function evaluations has been exceeded.',
               'Maximum number of iterations has been exceeded.')[status]
    return OptimizeResult(x=simplex[0], fun=np.min(values), nit=nit, nfev=nfev,
                          status=status, success=status == 0, message=message,
                          final_simplex=(simplex, values))
