"""Numerical and transport contracts for the CPU ROI workers."""
import unittest
from unittest.mock import patch

import kikuchipy as kp
import numpy as np
from orix.crystal_map import Phase

from multistep_overlap_ebsd import core


class CPUWorkerTests(unittest.TestCase):
    def state(self):
        rng = np.random.default_rng(10)
        mp = kp.signals.EBSDMasterPattern(
            rng.random((2, 31, 31), dtype=np.float32), projection='lambert',
            hemisphere='both', phase=Phase(name='Cu', point_group='m-3m'))
        return dict(master_kind='kikuchipy', mp_signal=mp, h=12, w=16, cols=2,
                    master_energy_kv=30., sample_tilt_deg=-20., detector_tilt_deg=.1,
                    azimuthal_deg=0., twist_deg=0., kikuchipy_frame_active=True,
                    weights=np.ones((12, 16), dtype=np.float32)/192,
                    fit_blur_gain=False, fit_maxiter=2, fit_popsize=4,
                    fit_bounds=None, blur_sigma=.7, fit_method='staged_joint')

    def test_batched_kikuchipy_projection_preserves_fixed_and_varying_pcs(self):
        eulers = np.array([[.2, .3, .4], [1.2, .4, 2.1], [.4, .9, 1.3]])
        for frame in (False, True):
            for varying in (False, True):
                state = self.state()
                state['kikuchipy_frame_active'] = frame
                pcs = np.tile([.48, -.43, .96], (3, 1))
                if varying:
                    pcs[1, 0] += .025
                    pcs[2, 2] += .04
                with self.subTest(frame=frame, varying=varying), patch.object(core, '_RESIDUAL_ROI_WORKER_STATE', state):
                    expected = np.stack([core._residual_roi_worker_simulated_pattern(e, pc) for e, pc in zip(eulers, pcs)])
                    actual = core._residual_roi_worker_simulated_patterns(eulers, pcs)
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_worker_retains_residual_but_does_not_transfer_inspection_images(self):
        state = self.state()
        payload = core.ResidualBatchPayload(
            indices=np.array([0, 1]), experimental=np.arange(384, dtype=np.float32).reshape(2, 12, 16),
            eulers_rad=np.array([[.2, .3, .4], [.4, .5, .6]]),
            pc_bruker=np.tile([.48, -.43, .96], (2, 1)), pc_custom=np.ones((2, 3)))
        with patch.object(core, '_RESIDUAL_ROI_WORKER_STATE', state):
            results = core._compute_residual_roi_batch(payload)
        for result in results:
            self.assertEqual(result.residual.shape, (12, 16))
            self.assertEqual(result.residual.dtype, np.float32)
            self.assertEqual(result.fitted_sigma, .7)
            for name in ('experimental', 'simulated', 'simulated_unfitted', 'blurred_simulated', 'gain_map'):
                self.assertIsNone(getattr(result, name))

    def test_worker_energy_cache_keeps_selected_plane_and_method(self):
        rng = np.random.default_rng(12)
        raw = rng.random((2, 3, 31, 31), dtype=np.float32)
        source = kp.signals.EBSDMasterPattern(raw, projection='lambert', hemisphere='both',
                                            phase=Phase(name='Cu', point_group='m-3m'))
        for axis in source.axes_manager.navigation_axes:
            if axis.index_in_array == 1:
                axis.name, axis.units, axis.offset, axis.scale = 'energy', 'keV', 28., 1.
            else:
                axis.name = 'hemisphere'
        with patch.object(core, '_RESIDUAL_ROI_WORKER_STATE', None), patch('kikuchipy.load', return_value=source.as_lazy()):
            core._init_residual_roi_worker(
                'kikuchipy', 'unused', 29., 'highest', None, None, 12, 16, 2,
                np.eye(3), None, -20., .1, 0., 0., True,
                np.ones((12, 16))/192, True, 2, 4, None, 0., 'staged_joint')
            state = core._RESIDUAL_ROI_WORKER_STATE
            np.testing.assert_array_equal(state['mp_signal'].data, raw[:, 1])
            self.assertEqual(state['fit_method'], 'staged_joint')
            self.assertIsInstance(state['mp_signal'].data, np.ndarray)


if __name__ == '__main__':
    unittest.main()
