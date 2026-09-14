"""Numerical equivalence of compiled orientation fitting and SciPy/Kikuchipy."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dask
import kikuchipy as kp
from numba import njit
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
from scipy.optimize import minimize

from multistep_overlap_ebsd import core
from multistep_overlap_ebsd.cpu_refinement import _nelder_mead, minimize_orientation


@njit
def objective(x, kind):
    if kind == 0:
        return (x[0]-.13)**2 + 2*(x[1]+.1)**2 + 3*(x[2]-.7)**2
    if kind == 1:
        return 1.  # Ties, shrinking and exhausted budgets within a shrink.
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 + (x[2]-.5)**2


class SimplexTests(unittest.TestCase):
    def test_scipy_simplex_evaluation_and_iteration_equivalence(self):
        rng = np.random.default_rng(17)
        for kind in range(3):
            for start in [np.zeros(3), np.ones(3), *rng.uniform(-1, 1, (4, 3))]:
                for budget in [*range(1, 41), 100, 600]:
                    for bounded in (False, True):
                        with self.subTest(kind=kind, start=start, budget=budget, bounded=bounded):
                            lower, upper = ((start-.1, start+.1) if bounded else
                                            (np.full(3, -np.inf), np.full(3, np.inf)))
                            reference = minimize(lambda x: objective(x, kind), start, method='Nelder-Mead',
                                bounds=list(zip(lower, upper)), options={'maxfev': budget})
                            simplex, values, nfev, nit = _nelder_mead(
                                objective, start, kind, lower, upper, budget, 2**31-1, 1e-4, 1e-4)
                            np.testing.assert_array_equal(simplex, reference.final_simplex[0])
                            np.testing.assert_array_equal(values, reference.final_simplex[1])
                            self.assertEqual((nfev, nit), (reference.nfev, reference.nit))

    def test_other_objectives_and_custom_options_use_scipy(self):
        callback = Mock()
        expected = minimize(lambda x: np.sum(x*x), np.ones(3), method='Nelder-Mead',
                            options={'maxfev': 40, 'adaptive': True})
        actual = minimize(lambda x: np.sum(x*x), np.ones(3), method=minimize_orientation,
                          callback=callback, options={'maxfev': 40, 'adaptive': True})
        np.testing.assert_array_equal(actual.x, expected.x)
        self.assertEqual(actual.fun, expected.fun)
        self.assertEqual(actual.nfev, expected.nfev)
        self.assertTrue(callback.called)


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(34)
        self.phase = Phase(name='Cu', point_group='m-3m')
        self.master = kp.signals.EBSDMasterPattern(self.rng.random((2, 31, 31), dtype=np.float32),
            projection='lambert', hemisphere='both', phase=self.phase)
        self.patterns = self.rng.random((6, 12, 16), dtype=np.float32)
        self.xmap = CrystalMap(rotations=Rotation.from_euler(self.rng.random((6, 3))),
            x=np.arange(6), phase_id=np.zeros(6, dtype=int), phase_list=PhaseList(self.phase))
        self.session = core.WorkflowSession()
        self.session.master = SimpleNamespace(mp_signal=self.master, energy_kv=30.)

    def tearDown(self):
        self.session.close()

    def test_public_kikuchipy_matches_with_fixed_varying_pcs_masks_and_budgets(self):
        mask = self.rng.random((12, 16)) < .2
        for varying in (False, True):
            pcs = np.tile([.48, .45, .65], (6, 1))
            if varying:
                pcs[:, 0] += np.arange(6)*.001
            detector = kp.detectors.EBSDDetector(shape=(12, 16), pc=pcs)
            for signal_mask in (None, mask):
                for budget in (3, 5, 15, 25, 100):
                    for dtype in (np.uint8, np.float32):
                        with self.subTest(varying=varying, masked=signal_mask is not None, budget=budget, dtype=dtype):
                            sig = kp.signals.EBSD((self.patterns*255).astype(dtype))
                            self.session._configure_signal_navigation_axis(sig)
                            kwargs = dict(xmap=self.xmap, detector=detector, master_pattern=self.master,
                                energy=30., trust_region=[1.4]*3, signal_mask=signal_mask, method='minimize',
                                method_kwargs=dict(method='Nelder-Mead', options=dict(maxfev=budget)), compute=True)
                            with contextlib.redirect_stdout(io.StringIO()), dask.config.set(num_workers=2):
                                expected = sig.refine_orientation(**kwargs)
                                actual = self.session._run_kikuchipy_refinement(sig, 'refine_orientation', **kwargs)
                            np.testing.assert_array_equal(actual.rotations.data, expected.rotations.data)
                            for name in ('scores', 'num_evals'):
                                np.testing.assert_array_equal(actual.prop[name], expected.prop[name])
                            np.testing.assert_array_equal(detector.pc, pcs)  # No caller mutation.
                            self.assertEqual(kwargs['method_kwargs']['method'], 'Nelder-Mead')

    def test_pc_and_joint_fits_keep_original_solver_and_per_point_geometry(self):
        detector = kp.detectors.EBSDDetector(shape=(12, 16), pc=np.tile([.48, .45, .65], (6, 1)))
        for operation in ('refine_projection_center', 'refine_orientation_projection_center'):
            signal = kp.signals.EBSD(self.patterns)
            backend = Mock(return_value='refined')
            setattr(signal, operation, backend)
            kwargs = dict(detector=detector, method='minimize',
                          method_kwargs=dict(method='Nelder-Mead', options=dict(maxfev=25)))
            with patch.object(self.session, '_materialize_signal_batch', return_value=signal):
                self.session._run_kikuchipy_refinement(signal, operation, **kwargs)
            self.assertIs(backend.call_args.kwargs['detector'], detector)
            self.assertEqual(backend.call_args.kwargs['method_kwargs'], kwargs['method_kwargs'])

    def test_primary_full_batch_and_cancellation_commit_only_completed_points(self):
        session = self.session
        count = 1500
        session.master.kind, session.master.phase = 'kikuchipy', self.phase
        session.data = SimpleNamespace(h=128, w=156, rows=1, cols=count, source_type='up_ang')
        session.current_eulers_rad = np.full((count, 3), .2)
        session.current_pc_bruker = np.full((count, 3), .5)
        session.current_phases = np.zeros(count, dtype=int)
        session.indexed_candidate_eulers_rad = np.full((count, 5, 3), .2)
        session.last_scores_map = np.zeros((1, count))
        session._signal_from_indices = Mock(return_value=object())
        session._kikuchipy_detector_for_indices = Mock(return_value=object())
        session._kikuchipy_refinement_binning_settings = Mock(return_value=(1, (0, 128, 0, 156), None, 'full'))
        session._invalidate_residual_cache = Mock()
        session._invalidate_orientation_cache = Mock()
        def fit(_signal, *, point_indices, **kwargs):
            return SimpleNamespace(rotations=Rotation.from_euler(np.full((len(point_indices), 3), .4)),
                                   prop={'scores': np.full(len(point_indices), .9)})
        session._refine_orientation_signal = Mock(side_effect=fit)
        messages = []
        def cancel_after_batch(value, message):
            messages.append(message)
            if message.startswith('Refined '):
                raise InterruptedError()
        with self.assertRaises(InterruptedError):
            session.refine_orientations_indices(np.arange(count), phase_id=0,
                parallel_cores=6, progress_callback=cancel_after_batch)
        expected_batch = (256*1024**2)//(128*156*4*5)
        self.assertEqual(expected_batch, 672)
        session._signal_from_indices.assert_called_once()
        np.testing.assert_array_equal(session._signal_from_indices.call_args.args[0],
                                      np.repeat(np.arange(expected_batch), 5))
        np.testing.assert_allclose(session.current_eulers_rad[:expected_batch], .4, atol=1e-15)
        np.testing.assert_array_equal(session.current_eulers_rad[expected_batch:], .2)
        self.assertTrue(np.isnan(session.indexed_candidate_eulers_rad[:expected_batch]).all())
        self.assertTrue(np.isfinite(session.indexed_candidate_eulers_rad[expected_batch:]).all())
        np.testing.assert_array_equal(session.last_scores_map[0, :expected_batch], .9)
        np.testing.assert_array_equal(session.last_scores_map[0, expected_batch:], 0)
        self.assertTrue(any('batch 1/3' in message for message in messages))

    def test_iteration_limits_and_unsupported_orientation_options(self):
        detector = kp.detectors.EBSDDetector(shape=(12, 16), pc=[.48, .45, .65])
        sig = kp.signals.EBSD(self.patterns)
        self.session._configure_signal_navigation_axis(sig)
        for options in ({'maxiter': 1}, {'maxiter': 3}, {'maxfev': 5, 'maxiter': 2},
                        {'maxfev': 25, 'adaptive': True}, {'maxfev': 25, 'xatol': 1e-2, 'fatol': 1e-3}):
            kwargs = dict(xmap=self.xmap, detector=detector, master_pattern=self.master,
                energy=30., trust_region=None, method='minimize',
                method_kwargs=dict(method='Nelder-Mead', options=options), compute=True)
            with contextlib.redirect_stdout(io.StringIO()), dask.config.set(num_workers=2):
                expected = sig.refine_orientation(**kwargs)
                actual = self.session._run_kikuchipy_refinement(sig, 'refine_orientation', **kwargs)
            np.testing.assert_array_equal(actual.rotations.data, expected.rotations.data)
            for name in ('scores', 'num_evals'):
                np.testing.assert_array_equal(actual.prop[name], expected.prop[name])


if __name__ == '__main__':
    unittest.main()
