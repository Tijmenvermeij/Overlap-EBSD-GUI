"""Scientific equivalence, cache lifetime and cancellation for CPU indexing."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dask
import dask.array as da
import kikuchipy as kp
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from multistep_overlap_ebsd import core
from multistep_overlap_ebsd.cpu_indexing import PreparedDictionary, cpu_job, refinement_chunks, unique_fit_rows


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)

    def test_matches_kikuchipy_with_mask_threads_and_cached_statistics(self):
        dictionary = self.rng.integers(0, 255, (47, 8, 10), dtype=np.uint8)
        patterns = self.rng.integers(0, 255, (260, 8, 10), dtype=np.uint8)
        phase = Phase(name='Cu', point_group='m-3m')
        xmap = CrystalMap(rotations=Rotation.from_euler(self.rng.random((47, 3))),
                         phase_id=np.zeros(47, dtype=int), x=np.arange(47), phase_list=PhaseList(phase))
        reference_signal = kp.signals.EBSD(dictionary)
        reference_signal.xmap = xmap
        for mask in (None, self.rng.random((8, 10)) < .2):
            with self.subTest(mask=mask is not None), contextlib.redirect_stdout(io.StringIO()):
                reference = kp.signals.EBSD(patterns).dictionary_indexing(
                    reference_signal, keep_n=5, signal_mask=mask, n_per_iteration=11)
                cache = PreparedDictionary(da.from_array(dictionary, chunks=(11, 8, 10)), mask)
                for workers in (1, 4):
                    indices, scores = cache.index(patterns, 5, 11, workers)
                    np.testing.assert_array_equal(indices, reference.prop['simulation_indices'])
                    np.testing.assert_allclose(scores, reference.prop['scores'], atol=2e-6, rtol=0)
                self.assertTrue(cache.ready.all())
                before = cache.prepare(0, 11, 1)
                with patch('multistep_overlap_ebsd.cpu_indexing.np.mean', side_effect=AssertionError('Recomputed mean')):
                    np.testing.assert_array_equal(cache.prepare(0, 11, 1), before)
                self.assertEqual(cache.means.nbytes + cache.norms.nbytes + cache.ready.nbytes, 47*9)
                session = core.WorkflowSession()
                app_cache = SimpleNamespace(signal=reference_signal, rotation_count=47, pattern_shape=(8, 10))
                try:
                    result = session._dictionary_index_kikuchipy_signal(kp.signals.EBSD(patterns),
                        cache=app_cache, keep_n=5, signal_mask=mask, n_per_iteration=11)
                    np.testing.assert_array_equal(result[2], reference.rotations.to_euler())
                finally:
                    session.close()

    def test_exact_ties_are_deterministic_across_blocks_and_keep_n_is_capped(self):
        pattern = np.arange(80, dtype=np.float32).reshape(8, 10)
        data = np.repeat(pattern[None], 19, axis=0)
        cache = PreparedDictionary(data, None)
        for size in (3, 7, 19):
            indices, scores = cache.index(pattern[None], 5, size, 2)
            np.testing.assert_array_equal(indices, [[0, 1, 2, 3, 4]])
            np.testing.assert_allclose(scores, 1, atol=2e-6)
        indices, scores = cache.index(pattern[None], 40, 3, 1)
        np.testing.assert_array_equal(indices, np.arange(19)[None])

    def test_degenerate_dictionary_is_excluded_and_invalid_experiment_has_no_score(self):
        patterns = self.rng.random((3, 8, 10), dtype=np.float32)
        data = np.concatenate((np.ones((2, 8, 10), dtype=np.float32), patterns))
        cache = PreparedDictionary(data, None)
        indices, scores = cache.index(patterns[:1], 3, 2, 1)
        self.assertTrue(np.all(indices >= 2))
        self.assertTrue(np.isfinite(scores).all())
        with np.errstate(divide='ignore', invalid='ignore'):
            _, scores = cache.index(np.zeros((1, 8, 10), dtype=np.float32), 3, 2, 1)
        self.assertTrue(np.isnan(scores).all())
        with self.assertRaises(ValueError):
            PreparedDictionary(data, np.ones((8, 10), dtype=bool)).index(patterns, 3, 2, 1)

    def test_cancellation_keeps_only_complete_preparation_and_can_resume(self):
        data = self.rng.random((12, 8, 10), dtype=np.float32)
        cache = PreparedDictionary(data, None)
        def cancel(fraction):
            if fraction > 0:
                raise InterruptedError()
        with self.assertRaises(InterruptedError):
            cache.index(data[:2], 3, 4, 2, cancel)
        np.testing.assert_array_equal(cache.ready, np.arange(12) < 4)
        actual = cache.index(data[:2], 3, 4, 2)
        expected = PreparedDictionary(data, None).index(data[:2], 3, 4, 2)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_dictionary_cache_replaced_on_mask_or_dictionary_change(self):
        data = self.rng.random((12, 8, 10), dtype=np.float32)
        dictionary = kp.signals.EBSD(data)
        dictionary.xmap = CrystalMap(rotations=Rotation.identity(12), x=np.arange(12))
        cache = SimpleNamespace(signal=dictionary, rotation_count=12, pattern_shape=(8, 10))
        session = core.WorkflowSession()
        try:
            def run(mask=None):
                return session._dictionary_index_kikuchipy_signal(kp.signals.EBSD(data[:2]),
                    cache=cache, keep_n=5, signal_mask=mask)
            run()
            first = session._dictionary_preparation
            run()
            self.assertIs(session._dictionary_preparation, first)
            mask = np.zeros((8, 10), dtype=bool); mask[0] = True
            run(mask)
            self.assertIsNot(session._dictionary_preparation, first)
            first = session._dictionary_preparation
            dictionary.data = dictionary.data.copy()
            run(mask)
            self.assertIsNot(session._dictionary_preparation, first)
            session._clear_dictionary_cache()
            self.assertIsNone(session._dictionary_preparation)
        finally:
            session.close()

    def test_worker_limit_restored_on_cancel_and_chunk_size_adapts(self):
        @cpu_job
        def cancelled(*, parallel_cores=0):
            self.assertEqual(dask.config.get('num_workers'), 2)
            raise InterruptedError()
        with dask.config.set(num_workers=7):
            with self.assertRaises(InterruptedError):
                cancelled(parallel_cores=2)
            self.assertEqual(dask.config.get('num_workers'), 7)
        self.assertEqual(refinement_chunks(80, 10), 4)
        self.assertGreater(refinement_chunks(640, 10), 4)
        self.assertEqual(refinement_chunks(6400, 10), 64)
        @cpu_job
        def positional(parallel_cores=0):
            return dask.config.get('num_workers')
        self.assertEqual(positional(2), 2)


class RefinementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.phase = Phase(name='Cu', point_group='m-3m')
        self.master = kp.signals.EBSDMasterPattern(rng.random((2, 31, 31), dtype=np.float32),
            projection='lambert', hemisphere='both', phase=self.phase)
        self.session = core.WorkflowSession()
        self.session.master = SimpleNamespace(mp_signal=self.master, energy_kv=30.)
        self.patterns = rng.random((3, 12, 16), dtype=np.float32)

    def tearDown(self):
        self.session.close()

    def test_repeated_seeds_and_single_point_match_original_fits(self):
        for points in (np.array([0, 0]), np.array([0, 0, 1, 1, 2])):
            eulers = np.array([[.2, .3, .4], [.5, .7, .9], [1.1, .4, .7]])[points]
            if len(points) > 2:
                eulers[3, 0] += .04  # Retain a genuinely distinct candidate.
            xmap = CrystalMap(rotations=Rotation.from_euler(eulers), x=np.arange(len(points)),
                             phase_id=np.zeros(len(points), dtype=int), phase_list=PhaseList(self.phase))
            pcs = np.tile([.48, .45, .65], (len(points), 1)); pcs[:, 0] += points*.01
            detector = kp.detectors.EBSDDetector(shape=(12, 16), pc=pcs)
            sig = kp.signals.EBSD(self.patterns[points])
            kwargs = dict(xmap=xmap, detector=detector, master_pattern=self.master, energy=30.,
                trust_region=[1.4]*3, method='minimize',
                method_kwargs=dict(method='Nelder-Mead', options=dict(maxfev=15)), compute=True)
            with contextlib.redirect_stdout(io.StringIO()), dask.config.set(num_workers=2):
                reference = sig.refine_orientation(**kwargs)
                actual = self.session._refine_orientation_signal(sig, point_indices=points, **kwargs)
            np.testing.assert_array_equal(actual.rotations.data, reference.rotations.data)
            for name in ('scores', 'num_evals'):
                np.testing.assert_array_equal(actual.prop[name], reference.prop[name])

    def test_missing_fallback_seeds_deduplicate_only_within_each_point(self):
        points = np.repeat([8, 9], 5)
        eulers = np.tile([.2, .4, .8], (10, 1))
        rows, inverse = unique_fit_rows(points, eulers)
        np.testing.assert_array_equal(rows, [0, 5])
        np.testing.assert_array_equal(inverse, [0]*5+[1]*5)

    def test_master_cache_reused_and_replaced_with_new_energy_or_data(self):
        source = kp.signals.EBSDMasterPattern(
            np.stack((self.master.data, self.master.data+1), axis=1),
            projection='lambert', hemisphere='both', phase=self.phase)
        for axis in source.axes_manager.navigation_axes:
            if axis.index_in_array == 1:
                axis.name, axis.offset, axis.scale = 'energy', 29., 1.
        self.session.master.mp_signal = source.as_lazy()
        first = self.session._refinement_master()
        self.assertIs(self.session._refinement_master(), first)
        np.testing.assert_array_equal(first.data, source.data[:, 1])
        self.session.master.energy_kv = 29.
        second = self.session._refinement_master()
        self.assertIsNot(second, first)
        np.testing.assert_array_equal(second.data, source.data[:, 0])
        # A collapsed weighted-energy master has no energy axis.
        self.session.master.mp_signal = second
        np.testing.assert_array_equal(self.session._refinement_master().data, second.data)

    def test_batch_projection_matches_single_points_for_varying_pcs_frames_and_energy_models(self):
        session = self.session
        session.master.kind = 'kikuchipy'
        session.data = SimpleNamespace(h=12, w=16, source_type='h5oina',
            sample_tilt_deg=70., detector_tilt_deg=5., azimuthal_deg=2., twist_deg=1.)
        pcs = np.tile([.48, .45, .65], (3, 1))
        session.current_pc_custom = pcs.copy()
        eulers = np.array([[.2, .3, .4], [.5, .7, .9], [1.1, .4, .7]])
        indices = np.array([2, 0, 2, 1])
        # Test a selected energy plane, an already collapsed weighted master,
        # and a source whose dtype triggers Kikuchipy intensity rescaling.
        source = kp.signals.EBSDMasterPattern(
            np.stack((self.master.data, self.master.data+1), axis=1),
            projection='lambert', hemisphere='both', phase=self.phase)
        for axis in source.axes_manager.navigation_axes:
            if axis.index_in_array == 1:
                axis.name, axis.offset, axis.scale = 'energy', 29., 1.
        source64 = source.deepcopy()
        source64.data = source64.data.astype(np.float64)
        weighted = core._master_signal_with_energy_weights(source, np.array([29., 30.]), np.array([.3, .7]))
        for master in (source.as_lazy(), weighted, source64.as_lazy()):
            session.master.mp_signal = master
            for vary_pc in (False, True):
                session.current_pc_bruker = pcs.copy()
                if vary_pc:
                    session.current_pc_bruker[:, 0] += np.arange(3)*.01
                for frame in ('up_ang', 'h5oina'):
                    session.data.source_type = frame
                    with dask.config.set(scheduler='threads', num_workers=2):
                        expected = np.stack([session._simulate_pattern_for_euler(i, eulers[i]) for i in indices])
                        actual = session._simulate_patterns_for_eulers(indices, eulers[indices])
                    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
