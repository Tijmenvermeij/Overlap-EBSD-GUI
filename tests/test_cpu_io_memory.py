from __future__ import annotations

import gc
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dask.array as da
import h5py
import numpy as np

from multistep_overlap_ebsd.core import (
    INSPECTION_PATTERN_CACHE_SIZE,
    OverlapMixtureResult,
    WorkflowSession,
    _H5DatasetArray,
    _read_h5_patterns,
    _residual_to_uint8,
)


class CPUInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="overlap-cpu-input-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "patterns.h5oina"
        rng = np.random.default_rng(41)
        self.patterns = rng.integers(0, 65535, size=(3, 4, 8, 10), dtype=np.uint16)

    def session(self, *, flat: bool = True) -> WorkflowSession:
        with h5py.File(self.path, "w") as h5:
            h5.create_dataset("7/EBSD/Data/Processed Patterns", data=(
                self.patterns.reshape(12, 8, 10) if flat else self.patterns
            ))
        session = WorkflowSession()
        session.data = SimpleNamespace(
            pattern_path=str(self.path), source_type="h5oina", h5_analysis_root="7",
            rows=3, cols=4, count=12, h=8, w=10,
            signal=SimpleNamespace(data=da.from_array(self.patterns, chunks=(1, 1, 8, 10))),
        )
        session._configure_h5_pattern_source()
        self.assertIsNotNone(session._h5_pattern_source)
        return session

    @staticmethod
    def pixels(signal) -> np.ndarray:
        data = signal.data
        return np.asarray(data.compute() if hasattr(data, "compute") else data)

    def test_both_h5_layouts_preserve_roi_order_duplicates_and_full_scan(self) -> None:
        for flat in (False, True):
            session = self.session(flat=flat)
            for indices in ([8], [8, 8, 8], [5, 6, 9, 10], [10, 2, 10, 1]):
                idx = np.asarray(indices)
                actual = self.pixels(session._signal_from_indices(idx))
                np.testing.assert_array_equal(actual, self.patterns.reshape(12, 8, 10)[idx])
                self.assertIsNone(session._flat_pattern_source_cache)
            np.testing.assert_array_equal(session._pattern_at(7), self.patterns.reshape(12, 8, 10)[7])
            full = session._signal_from_indices(np.arange(12))
            self.assertTrue(hasattr(full.data, "chunks"))
            np.testing.assert_array_equal(self.pixels(full), self.patterns.reshape(12, 8, 10))
            cached = session._flat_pattern_source_cache[1]
            self.assertIs(cached, session._flat_h5_pattern_data())

    def test_background_binning_and_crop_match_loaded_signal(self) -> None:
        session = self.session()
        idx = np.array([11, 2, 11, 4])
        session.set_dynamic_background(True, std_px=1.5)
        direct = self.pixels(session._signal_from_indices(
            idx, software_binning=2, crop_extent=(0, 8, 0, 8),
        ))
        session._h5_pattern_source = None
        lazy = self.pixels(session._signal_from_indices(
            idx, software_binning=2, crop_extent=(0, 8, 0, 8),
        ))
        np.testing.assert_array_equal(direct, lazy)

    def test_processed_batch_matches_single_patterns_including_background_dtype(self) -> None:
        session = self.session()
        indices = np.array([11, 2, 11, 4])
        for background in (False, True):
            session.set_dynamic_background(background, std_px=1.5)
            expected = np.stack([session._processed_pattern_at(int(i)) for i in indices])
            actual = session._processed_patterns_from_indices(indices)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.dtype(np.float32))

    def test_transformed_loader_data_disables_direct_access_and_reuses_flat_view(self) -> None:
        session = self.session()
        session.data.signal.data = session.data.signal.data[..., ::-1]
        session._configure_h5_pattern_source()
        self.assertIsNone(session._h5_pattern_source)
        idx = np.array([11, 2, 11, 4])
        first = self.pixels(session._signal_from_indices(idx))
        cached = session._flat_pattern_source_cache[1]
        second = self.pixels(session._signal_from_indices(idx[::-1]))
        self.assertIs(cached, session._flat_pattern_source_cache[1])
        np.testing.assert_array_equal(first, self.patterns.reshape(12, 8, 10)[idx, :, ::-1])
        np.testing.assert_array_equal(second, first[::-1])

    def test_h5_selection_reads_no_unrequested_patterns(self) -> None:
        class RecordingDataset:
            shape = (12, 8, 10)
            ndim = 3
            dtype = np.dtype(np.uint16)

            def __init__(inner):
                inner.keys = []

            def __getitem__(inner, key):
                inner.keys.append(key)
                return self.patterns.reshape(12, 8, 10)[key]

        dataset = RecordingDataset()
        indices = np.array([9, 2, 3, 9])
        actual = _read_h5_patterns(dataset, indices, rows=3, cols=4)
        np.testing.assert_array_equal(actual, self.patterns.reshape(12, 8, 10)[indices])
        self.assertEqual(dataset.keys, [slice(2, 4), slice(9, 10)])


class CPUResidualMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = WorkflowSession()
        self.addCleanup(self.session._clear_residual_pattern_store)
        count = INSPECTION_PATTERN_CACHE_SIZE + 3
        self.session.data = SimpleNamespace(
            rows=1, cols=count, count=count, h=8, w=10, source_type="h5oina",
        )
        self.session.master = SimpleNamespace(kind="legacy")
        self.session.current_eulers_rad = np.zeros((count, 3))
        self.session.current_phases = np.ones(count, dtype=np.int32)
        self.session.current_pc_bruker = np.full((count, 3), 0.5)
        self.session.current_pc_custom = np.full((count, 3), 0.5)
        rng = np.random.default_rng(32)
        self.experimental = rng.normal(size=(8, 10)).astype(np.float32)
        self.simulated = rng.normal(size=(8, 10)).astype(np.float32)
        self.weights = np.ones((8, 10), dtype=np.float32)
        self.weights[:2] = 0
        self.weights[:, :2] = 0
        self.session._overlap_weights = Mock(return_value=self.weights)
        self.session._processed_pattern_at = Mock(return_value=self.experimental)
        self.session._processed_patterns_from_indices = Mock(
            side_effect=lambda indices: np.stack([self.experimental] * len(indices)))
        self.session._simulate_pattern_for_euler = Mock(return_value=self.simulated)
        self.session.dictionary_cache = SimpleNamespace(software_binning=1, crop_extent=(0, 8, 0, 10))

    def result(self, index):
        return self.session.analyze_overlap_point(index, fit_blur_gain=False, blur_sigma=0.8, store_result=False)

    def test_disk_cache_preserves_exact_float32_and_compact_fit_metadata(self) -> None:
        result = self.result(2)
        self.session._store_residual_result(result)
        compact = self.session.residual_point_results[2]
        self.assertIsNone(compact.experimental)
        self.assertIsNone(compact.simulated)
        self.assertIsNone(compact.residual)
        self.assertIsNone(compact.gain_map)
        self.assertEqual(compact.fitted_sigma, result.fitted_sigma)
        self.assertEqual(compact.gain_params, result.gain_params)
        self.assertEqual(compact.ellipse_params, result.ellipse_params)
        store = self.session._residual_pattern_store
        self.assertEqual(store.patterns.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(store.read(np.array([2, 2])), np.stack([result.residual] * 2))
        self.session._simulate_pattern_for_euler.reset_mock()
        signal = self.session._residual_signal_from_indices(np.array([2, 2]))
        np.testing.assert_array_equal(signal.data.compute(), np.stack([result.residual] * 2))
        self.session._simulate_pattern_for_euler.assert_not_called()
        np.testing.assert_array_equal(self.session._residual_pattern_u8(2), _residual_to_uint8(result.residual))
        selected = self.session.get_residual_point_result(2)
        for name in ("experimental", "simulated", "residual", "blurred_simulated", "gain_map"):
            np.testing.assert_allclose(getattr(selected, name), getattr(result, name), atol=1e-6)
            self.assertTrue(np.all(getattr(selected, name)[self.weights == 0] == 0))

    def test_selected_residual_and_mixture_images_are_bounded(self) -> None:
        for index in range(self.session.data.count):
            result = self.result(index)
            self.session._store_residual_result(result, keep_patterns=True)
            mixture = OverlapMixtureResult(
                index=index, row=0, col=index, primary_fraction=0.6, secondary_fraction=0.4,
                primary_coefficient=0.6, secondary_coefficient=0.4, ncc_mixture=0.9,
                residual_rms=0.1, old_primary_ncc=0.6, old_secondary_ncc=0.4,
                experimental=result.experimental, primary_simulated=result.simulated,
                secondary_simulated=result.simulated, combined_simulated=result.simulated,
                residual=result.residual,
            )
            self.session._store_overlap_mixture_result(mixture, keep_patterns=True)
        self.assertEqual(sum(r.experimental is not None for r in self.session.residual_point_results.values()), INSPECTION_PATTERN_CACHE_SIZE)
        self.assertEqual(sum(r.experimental is not None for r in self.session.overlap_mixture_results.values()), INSPECTION_PATTERN_CACHE_SIZE)
        self.assertIsNone(self.session.residual_point_results[0].residual)
        self.assertIsNotNone(self.session.get_residual_point_result(0).experimental)
        self.assertEqual(len(self.session._residual_inspection_indices), INSPECTION_PATTERN_CACHE_SIZE)

    def test_mixed_memory_and_disk_batch_preserves_external_result_precedence(self) -> None:
        first, second = self.result(0), self.result(1)
        self.session._store_residual_result(first)
        self.session._store_residual_result(second, keep_patterns=True)
        store = self.session._residual_pattern_store
        with patch.object(store, "read", wraps=store.read) as read:
            signal = self.session._residual_signal_from_indices(np.array([0, 1, 0]))
        self.assertEqual(read.call_count, 1)
        np.testing.assert_array_equal(read.call_args.args[0], [0, 0])
        np.testing.assert_array_equal(signal.data.compute(), np.stack([first.residual, second.residual, first.residual]))
        external = replace(first, residual=np.full((8, 10), 0.125, dtype=np.float32))
        supplied = {0: external, 1: self.session.residual_point_results[1]}
        signal = self.session._residual_signal_from_indices(np.array([0, 1]), residual_results=supplied)
        np.testing.assert_array_equal(signal.data.compute(), np.stack([external.residual, second.residual]))

    def test_partial_invalidation_preserves_other_float32_rows_and_full_cleans_file(self) -> None:
        for index in range(2):
            self.session._store_residual_result(self.result(index), keep_patterns=True)
        store = self.session._residual_pattern_store
        path = Path(store.path)
        unchanged = store.read(np.array([1])).copy()
        self.session._invalidate_residual_cache(np.array([0]))
        self.assertFalse(store.available[0])
        self.assertTrue(store.available[1])
        self.assertNotIn(0, self.session._residual_inspection_indices)
        np.testing.assert_array_equal(store.read(np.array([1])), unchanged)
        self.session._invalidate_residual_cache()
        self.assertIsNone(self.session._residual_pattern_store)
        self.assertFalse(path.exists())

    def test_restored_metadata_reconstructs_without_retaining_all_inspection_images(self) -> None:
        original = self.result(0)
        self.session.residual_point_results[0] = self.session._strip_residual_point_result(original)
        signal = self.session._residual_signal_from_indices(np.array([0, 0]))
        np.testing.assert_allclose(signal.data.compute(), np.stack([original.residual] * 2), atol=1e-6)
        self.assertIsNone(self.session.residual_point_results[0].experimental)
        self.assertIsNone(self.session.residual_point_results[0].residual)
        self.assertTrue(self.session._residual_pattern_store.available[0])
        self.assertEqual(self.session._processed_patterns_from_indices.call_count, 1)
        np.testing.assert_array_equal(self.session._processed_patterns_from_indices.call_args.args[0], [0])

    def test_partial_cache_reconstructs_only_missing_rows_and_preserves_fit_metadata(self) -> None:
        originals = [self.result(i) for i in range(4)]
        for result in originals:
            self.session.residual_point_results[result.index] = self.session._strip_residual_point_result(result)
        self.session._store_residual_result(originals[0])
        self.session._store_residual_result(originals[1], keep_patterns=True)
        # Include a fitted nontrivial gain, blur and secondary orientation. The
        # secondary simulation is irrelevant to residual dictionary matching.
        result = replace(self.session.residual_point_results[2], fitted_sigma=1.3,
                         gain_params=(.7, 1.2, .9), ellipse_params=(.8, 1.1, .2, -.1),
                         secondary_euler_rad=np.array([.1, .2, .3]), fit_message="saved fit")
        self.session.residual_point_results[2] = result
        originals[2] = self.session._materialize_residual_point_result(result)
        self.session._simulate_pattern_for_euler.reset_mock()
        callbacks = []
        actual = self.session._residual_signal_from_indices(
            np.array([2, 0, 1, 2, 3]), progress_callback=callbacks.append,
        ).data.compute()
        np.testing.assert_array_equal(actual, np.stack([originals[i].residual for i in [2, 0, 1, 2, 3]]))
        np.testing.assert_array_equal(self.session._processed_patterns_from_indices.call_args.args[0], [2, 3])
        self.assertEqual(self.session._simulate_pattern_for_euler.call_count, 2)
        saved = self.session.residual_point_results[2]
        self.assertEqual(saved.gain_params, result.gain_params)
        self.assertEqual(saved.scale, result.scale)
        self.assertEqual(saved.fit_message, "saved fit")
        np.testing.assert_array_equal(saved.secondary_euler_rad, result.secondary_euler_rad)
        self.assertEqual(callbacks[0], 0.)
        self.assertEqual(callbacks[-1], 1.)
        self.assertEqual(callbacks, sorted(callbacks))

    def test_reconstruction_can_cancel_between_bounded_batches_and_resume(self) -> None:
        count = 300
        self.session.data.count = self.session.data.cols = count
        self.session.current_eulers_rad = np.zeros((count, 3))
        original = self.result(0)
        self.session.residual_point_results = {
            i: replace(self.session._strip_residual_point_result(original), index=i, col=i)
            for i in range(count)
        }
        def cancel(fraction):
            if fraction > 0:
                raise InterruptedError()
        with self.assertRaises(InterruptedError):
            self.session._residual_signal_from_indices(np.arange(count), progress_callback=cancel)
        available = self.session._residual_pattern_store.available.copy()
        self.assertGreater(available.sum(), 0)
        self.assertLess(available.sum(), count)
        self.session._processed_patterns_from_indices.reset_mock()
        actual = self.session._residual_signal_from_indices(np.arange(count)).data.compute()
        np.testing.assert_array_equal(actual, np.stack([original.residual] * count))
        read_indices = np.concatenate([call.args[0] for call in self.session._processed_patterns_from_indices.call_args_list])
        np.testing.assert_array_equal(read_indices, np.flatnonzero(~available))

    def test_session_cleanup_removes_owned_temporary_store(self) -> None:
        self.session._store_residual_result(self.result(0))
        store = self.session._residual_pattern_store
        path = Path(store.path)
        session = WorkflowSession()
        session._residual_pattern_store = store
        self.session._residual_pattern_store = None
        del session
        gc.collect()
        self.assertFalse(path.exists())

    def test_close_is_idempotent_and_releases_owned_storage(self) -> None:
        self.session._store_residual_result(self.result(0), keep_patterns=True)
        path = Path(self.session._residual_pattern_store.path)
        with tempfile.TemporaryDirectory(prefix="overlap-owned-dictionary-test-") as directory:
            dictionary_path = Path(directory) / "dictionary.h5"
            dictionary_path.touch()
            signal = SimpleNamespace(data=da.from_array(np.zeros((2, 8, 10))), close_file=Mock())
            self.session.dictionary_cache = SimpleNamespace(
                signal=signal, owns_storage=True, storage_path=str(dictionary_path),
            )
            self.session.data.signal = signal
            self.session.close()
            self.session.close()
            self.assertFalse(dictionary_path.exists())
            signal.close_file.assert_not_called()
        self.assertFalse(path.exists())
        self.assertIsNone(self.session.data)
        self.assertIsNone(self.session.master)
        self.assertIsNone(self.session.dictionary_cache)
        self.assertFalse(self.session.residual_point_results)
        self.assertFalse(self.session._residual_inspection_indices)
        self.assertIsNone(self.session.last_overlap)

    def test_close_only_closes_real_handles_once_without_proxy_or_closed_file_warnings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="overlap-native-handle-test-") as directory:
            path = Path(directory) / "input.h5"
            native = h5py.File(path, "w")
            self.addCleanup(native.close)
            dataset = native.create_dataset("patterns", data=np.ones((2, 8, 10)))
            array = da.from_array(dataset, chunks=(1, 8, 10))
            proxy = da.from_array(_H5DatasetArray(path, "patterns"), chunks=(1, 8, 10))
            self.session.data.signal = SimpleNamespace(data=array)
            self.session.master = SimpleNamespace(
                mp_signal=SimpleNamespace(data=da.from_array(np.zeros((2, 8, 10)))),
                source_mp_signal=SimpleNamespace(data=array[:1]),
            )
            self.session.dictionary_cache = SimpleNamespace(
                signal=SimpleNamespace(data=proxy), owns_storage=False,
            )
            with (
                self.assertNoLogs("rsciio", level="WARNING"),
                self.assertNoLogs("hyperspy", level="WARNING"),
                patch.object(h5py.File, "close", autospec=True, side_effect=h5py.File.close) as close,
            ):
                self.session.close()
                self.assertFalse(native.id.valid)
                self.assertEqual(close.call_count, 1)
                # A previously closed native graph is also a normal case.
                self.session.data = SimpleNamespace(signal=SimpleNamespace(data=array))
                self.session.close()
                self.assertEqual(close.call_count, 1)


if __name__ == "__main__":
    unittest.main()
