from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter

from multistep_overlap_ebsd import core
from multistep_overlap_ebsd import cpu_fitting as cpu


class CPUFittingTests(unittest.TestCase):
    @staticmethod
    def patterns():
        yy, xx = np.indices((36, 42))
        primary = (9 + np.sin(xx / 4) + np.cos(yy / 3) + 0.4 * np.sin((xx + yy) / 2)).astype(np.float32)
        secondary = (8 + np.cos(xx / 2) + np.sin(yy / 6) + 0.3 * np.cos((xx - yy) / 3)).astype(np.float32)
        mask = ((xx - 21) / 21) ** 2 + ((yy - 18) / 18) ** 2 < 0.8
        weights = (np.random.default_rng(70).uniform(0.1, 1, primary.shape) * mask).astype(np.float32)
        params = np.asarray([1.2, 0.5, 4, 2.1, 1.1, 0.8, 0.08, -0.03])
        gain = core._power_gain_map(primary.shape, params[1:4], params[4:])
        blurred = core._normalize_weighted(gaussian_filter(primary, params[0]), weights)
        experimental = (core._normalize_weighted(blurred * gain, weights)
                        + 0.2 * core._normalize_weighted(secondary, weights)).astype(np.float32)
        return experimental, primary, secondary, weights, params

    def assert_parameters_in_bounds(self, params, bounds):
        self.assertEqual(np.asarray(params).shape, (8,))
        self.assertTrue(np.all(np.isfinite(params)))
        for value, (low, high) in zip(params, bounds):
            self.assertGreaterEqual(value, low - 1e-12)
            self.assertLessEqual(value, high + 1e-12)

    def test_lean_objectives_match_existing_image_models(self):
        experimental, primary, secondary, masked_weights, _ = self.patterns()
        bounds = np.asarray(cpu.DEFAULT_FIT_BOUNDS)
        candidates = np.random.default_rng(37).uniform(bounds[:, 0], bounds[:, 1], (24, 8))
        for weights in (masked_weights, np.ones(primary.shape, np.float32)):
            primary_objective = cpu._PrimaryObjective(experimental, primary, weights, projected=False)
            mixture_objective = cpu._MixtureObjective(experimental, primary, secondary, weights)
            for params in candidates:
                with self.subTest(masked=bool(np.any(weights == 0)), params=params):
                    for objective in (primary_objective, mixture_objective):
                        self.assertAlmostEqual(objective(params) / weights.sum(),
                                               objective.exact_score(params) / weights.sum(), delta=2e-5)

    def test_projection_reconstructs_bounded_original_primary_model(self):
        experimental, primary, _, weights, _ = self.patterns()
        objective = cpu._PrimaryObjective(experimental, primary, weights)
        bounds = np.asarray(cpu.DEFAULT_FIT_BOUNDS)[list(cpu._PROJECTED_INDICES)]
        candidates = np.random.default_rng(29).uniform(bounds[:, 0], bounds[:, 1], (24, 6))
        for params in candidates:
            with self.subTest(params=params):
                full = objective.full_params(params)
                self.assert_parameters_in_bounds(full, cpu.DEFAULT_FIT_BOUNDS)
                self.assertAlmostEqual(objective(params) / weights.sum(),
                                       objective.exact_score(full) / weights.sum(), delta=2e-5)
                # Analytic gain fitting cannot lose to uniform gain at the same blur.
                self.assertLessEqual(objective(params), objective.uniform_blur(params[0]) + 1e-10)

    def test_mixture_preserves_shared_gain_on_uncentered_raw_blurs(self):
        _, primary, secondary, weights, params = self.patterns()
        gain = core._power_gain_map(primary.shape, params[1:4], params[4:])
        processed1 = core._normalize_weighted(gaussian_filter(primary, params[0]) * gain, weights)
        processed2 = core._normalize_weighted(gaussian_filter(secondary, params[0]) * gain, weights)
        experimental = core._normalize_weighted(0.7 * processed1 + 0.3 * processed2, weights)
        objective = cpu._MixtureObjective(experimental, primary, secondary, weights)
        self.assertLess(objective(params) / weights.sum(), 1e-10)
        fit = core._evaluate_overlap_mixture_pattern(experimental, primary, secondary, weights, params)
        self.assertAlmostEqual(fit.primary_fraction, 0.7, delta=2e-5)
        self.assertAlmostEqual(fit.secondary_fraction, 0.3, delta=2e-5)
        self.assertGreaterEqual(fit.primary_coefficient, 0)
        self.assertGreaterEqual(fit.secondary_coefficient, 0)
        # Centering before applying gain is Step 3's model and would give a
        # substantially different Step 4 answer for these nonzero-mean inputs.
        wrong = core._normalize_weighted(core._normalize_weighted(gaussian_filter(primary, params[0]), weights) * gain, weights)
        self.assertLess(core._weighted_ncc(wrong, processed1, weights), 0.9)

    def test_mixture_nnls_keeps_boundary_solution_for_negative_component(self):
        _, primary, secondary, weights, params = self.patterns()
        params[1:3] = 1.0
        p = core._normalize_weighted(gaussian_filter(primary, params[0]), weights)
        s = core._normalize_weighted(gaussian_filter(secondary, params[0]), weights)
        experimental = core._normalize_weighted(p - 0.8 * s, weights)
        objective = cpu._MixtureObjective(experimental, primary, secondary, weights)
        fit = core._evaluate_overlap_mixture_pattern(experimental, primary, secondary, weights, params)
        self.assertEqual(fit.secondary_coefficient, 0.0)
        self.assertGreater(fit.primary_coefficient, 0.0)
        self.assertAlmostEqual(objective(params) / weights.sum(),
                               objective.exact_score(params) / weights.sum(), delta=2e-5)

    def test_mixture_collinear_components_are_finite(self):
        experimental, primary, _, weights, params = self.patterns()
        for secondary in (primary, primary * (1 + 1e-7), -primary):
            objective = cpu._MixtureObjective(experimental, primary, secondary, weights)
            value = objective(params)
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertAlmostEqual(value / weights.sum(),
                                   objective.exact_score(params) / weights.sum(), delta=2e-5)

    def test_staged_joint_cannot_regress_the_exported_residual(self):
        experimental, primary, secondary, weights, _ = self.patterns()
        for optimize, args in ((cpu.optimize_primary, (experimental, primary, weights)),
                               (cpu.optimize_mixture, (experimental, primary, secondary, weights))):
            with self.subTest(optimizer=optimize.__name__):
                staged = optimize(*args, maxiter=3, popsize=4, seed=13, method="staged")
                refined = optimize(*args, maxiter=3, popsize=4, seed=13, method="staged_joint")
                self.assertLessEqual(refined.fun, staged.fun)
                self.assert_parameters_in_bounds(staged.x, cpu.DEFAULT_FIT_BOUNDS)
                self.assert_parameters_in_bounds(refined.x, cpu.DEFAULT_FIT_BOUNDS)
                # The fixed-blur stage reuses convolution for gain candidates.
                self.assertLess(staged.nblur, staged.nfev)
                if optimize is cpu.optimize_primary:
                    for result in (staged, refined):
                        # Check the actual stored-image precision independently
                        # of the optimizer's internal score implementation.
                        y = core._normalize_weighted(experimental, weights)
                        b = core._normalize_weighted(gaussian_filter(primary, result.x[0]), weights)
                        g = core._power_gain_map(primary.shape, result.x[1:4], result.x[4:])
                        p = core._normalize_weighted(b*g, weights)
                        stored = np.asarray(y - core._weighted_ncc(y, p, weights)*p, dtype=np.float32)
                        self.assertEqual(result.fun, float(np.sum(weights*stored*stored)))

    def test_joint_strategy_returns_original_parameterization(self):
        experimental, primary, secondary, weights, _ = self.patterns()
        for optimize, args in ((cpu.optimize_primary, (experimental, primary, weights)),
                               (cpu.optimize_mixture, (experimental, primary, secondary, weights))):
            fit = optimize(*args, maxiter=2, popsize=4, seed=11, method="joint")
            self.assert_parameters_in_bounds(fit.x, cpu.DEFAULT_FIT_BOUNDS)
            self.assertTrue(np.isfinite(fit.fun))
            self.assertEqual(fit.fit_method, "joint")

    def test_custom_gain_bounds_are_obeyed_in_all_methods(self):
        experimental, primary, secondary, weights, _ = self.patterns()
        bounds = list(cpu.DEFAULT_FIT_BOUNDS)
        bounds[0] = (0.7, 1.5)
        bounds[1] = (2.8, 3.0)
        bounds[2] = (0.3, 0.4)
        for method in cpu.FIT_METHOD_LABELS:
            for optimize, args in ((cpu.optimize_primary, (experimental, primary, weights)),
                                   (cpu.optimize_mixture, (experimental, primary, secondary, weights))):
                with self.subTest(method=method, optimizer=optimize.__name__):
                    fit = optimize(*args, maxiter=2, popsize=4, seed=11, method=method, fit_bounds=bounds)
                    self.assert_parameters_in_bounds(fit.x, bounds)
                    self.assertTrue(np.isfinite(fit.fun))

    def test_blurs_are_cached_by_exact_sigma(self):
        experimental, primary, secondary, weights, params = self.patterns()
        for objective, expected in ((cpu._PrimaryObjective(experimental, primary, weights, projected=False), 1),
                                    (cpu._MixtureObjective(experimental, primary, secondary, weights), 2)):
            with patch.object(cpu, "gaussian_filter", wraps=gaussian_filter) as blur:
                objective(params)
                changed_gain = params.copy()
                changed_gain[2] += 1.0
                objective(changed_gain)
                self.assertEqual(blur.call_count, expected)
                changed_gain[0] = np.nextafter(params[0], np.inf)
                objective(changed_gain)
                self.assertEqual(blur.call_count, 2 * expected)

    def test_constant_and_near_zero_patterns_keep_original_threshold_behavior(self):
        experimental, primary, secondary, weights, _ = self.patterns()
        for y, p, w in ((np.zeros_like(experimental), primary, weights),
                        (experimental * 1e-9, primary, weights),
                        (experimental, np.ones_like(primary), weights),
                        (experimental, primary * 1e-9, weights),
                        (experimental, primary, weights * 1e-18)):
            for optimize, args, reference in (
                    (cpu.optimize_primary, (y, p, w), cpu._PrimaryObjective(y, p, w, projected=False)),
                    (cpu.optimize_mixture, (y, p, secondary, w), cpu._MixtureObjective(y, p, secondary, w))):
                with self.subTest(optimizer=optimize.__name__, y_variance=float(y.var()), p_variance=float(p.var()), weight_sum=float(w.sum())):
                    fit = optimize(*args, maxiter=1, popsize=4, seed=11)
                    self.assertTrue(np.isfinite(fit.fun))
                    self.assertEqual(fit.fun, reference.exact_score(fit.x))
                    self.assert_parameters_in_bounds(fit.x, cpu.DEFAULT_FIT_BOUNDS)

    def test_zero_weights_are_a_successful_zero_objective(self):
        experimental, primary, secondary, weights, _ = self.patterns()
        weights.fill(0)
        for optimize, args in ((cpu.optimize_primary, (experimental, primary, weights)),
                               (cpu.optimize_mixture, (experimental, primary, secondary, weights))):
            fit = optimize(*args, maxiter=1, popsize=4, seed=11)
            self.assertEqual(fit.fun, 0)
            self.assertEqual(fit.nfev, 0)
            self.assertTrue(fit.success)
            self.assert_parameters_in_bounds(fit.x, cpu.DEFAULT_FIT_BOUNDS)

    def test_invalid_options_and_inputs_fail_clearly(self):
        experimental, primary, secondary, weights, _ = self.patterns()
        self.assertEqual(cpu.FIT_METHOD_DEFAULT, "staged")
        for method in cpu.FIT_METHOD_LABELS:
            self.assertEqual(cpu.validate_fit_method(method), method)
        for method in ("unknown", None, []):
            with self.assertRaisesRegex(ValueError, "Unknown fit method"):
                cpu.validate_fit_method(method)
        for bounds in ([(0, 1)], [None] * 8, [(0, 0)] * 8, [(0, np.inf)] * 8):
            with self.assertRaisesRegex(ValueError, "fit bounds"):
                cpu.optimize_primary(experimental, primary, weights, maxiter=1, popsize=4, seed=1, fit_bounds=bounds)
        with self.assertRaisesRegex(ValueError, "matching"):
            cpu._PrimaryObjective(experimental, primary[:-1], weights)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            cpu._MixtureObjective(experimental, primary, secondary, -weights)
        nonfinite = experimental.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            cpu._PrimaryObjective(nonfinite, primary, weights)


if __name__ == "__main__":
    unittest.main()
