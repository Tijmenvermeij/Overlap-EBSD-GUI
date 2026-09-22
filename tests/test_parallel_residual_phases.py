"""Real spawned workers must preserve phase-specific residual results."""
from concurrent.futures import ProcessPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np

from multistep_overlap_ebsd import core
from multistep_overlap_ebsd.phases import PhaseRegistry


def init_synthetic_masters(common_args, masters, signals):
    # Only file loading is replaced; projection, fitting and process transport
    # run normally, with distinct synthetic master patterns for each phase.
    import kikuchipy
    from orix.crystal_map import Phase
    def load(path, **kwargs):
        master = kikuchipy.signals.EBSDMasterPattern(signals[str(path)],
            projection='lambert', hemisphere='both', phase=Phase(name='Cu', point_group='m-3m'))
        for axis in master.axes_manager.navigation_axes:
            if axis.index_in_array == 1:
                axis.name, axis.offset, axis.scale = 'energy', 18., 1.
        return master
    kikuchipy.load = load
    core._init_phase_residual_roi_worker(common_args, masters)


class ParallelResidualPhaseTests(unittest.TestCase):
    def test_spawned_workers_match_sequential_for_one_two_three_phases(self):
        from test_h5_indexed_input import IndexedInputTests
        helper = IndexedInputTests(); helper.setUp()
        s = helper.session
        try:
            helper.write_input(); helper.load_input()
            s.phase_registry = PhaseRegistry()
            signals = {}
            for n in range(3):
                path = helper.root / f'master{n}.h5'
                path.write_bytes(bytes([n]))
                master = helper.master.deepcopy()
                master.data = np.random.default_rng(90+n).random(master.data.shape, dtype=np.float32)
                signals[str(path.resolve())] = master.data.copy()
                with patch('kikuchipy.load', return_value=master):
                    s.attach_phase_master(str(path))
            # Include nonconstant PCs, a nonzero blur, and interleaved phases.
            s.current_pc_bruker[:, 0] += np.arange(6)*.001
            def pool(**kwargs):
                self.assertIs(kwargs['initializer'], core._init_phase_residual_roi_worker)
                self.assertEqual(kwargs['max_workers'], 2)
                return ProcessPoolExecutor(**dict(kwargs, initializer=init_synthetic_masters,
                    initargs=(*kwargs['initargs'], signals)))
            for phases in (1, 2, 3):
                s.current_phases[:] = np.arange(6) % phases + 1
                for fitted in (False, True):
                    with self.subTest(phases=phases, fitted=fitted):
                        options = dict(blur_sigma=.3, fit_blur_gain=fitted, fit_maxiter=2, fit_popsize=2)
                        s.compute_overlap_residual_indices(np.arange(6), parallel_cores=1, **options)
                        reference = [s.get_residual_point_result(i) for i in range(6)]
                        progress = []
                        with patch.object(core, 'ProcessPoolExecutor', side_effect=pool):
                            s.compute_overlap_residual_indices(np.arange(6), parallel_cores=2,
                                progress_callback=lambda value, message: progress.append(message), **options)
                        self.assertFalse(any('falling back' in msg for msg in progress), progress)
                        for i, expected in enumerate(reference):
                            actual = s.get_residual_point_result(i)
                            self.assertEqual(actual.primary_phase_key, expected.primary_phase_key)
                            np.testing.assert_allclose(actual.residual, expected.residual, atol=2e-5)
                            self.assertAlmostEqual(actual.ncc_es, expected.ncc_es, places=5)
        finally:
            helper.tearDown()


if __name__ == '__main__':
    unittest.main()
