from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation

from multistep_overlap_ebsd.core import (
    DICTIONARY_FORMAT_V1,
    DICTIONARY_FORMAT_V2,
    WorkflowSession,
)


class _FakeMasterSignal:
    def get_patterns(self, *, rotations, detector, dtype_out, **_kwargs):
        dtype = np.dtype(dtype_out)
        base = np.arange(np.prod(detector.shape), dtype=np.float32).reshape(detector.shape)
        patterns = np.stack([base + i for i in range(rotations.size)])
        if np.issubdtype(dtype, np.integer):
            patterns = np.stack(
                [np.rint(255 * (p - p.min()) / (p.max() - p.min())).astype(dtype) for p in patterns]
            )
        return SimpleNamespace(data=patterns.astype(dtype, copy=False))


class DictionaryStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="overlap-ebsd-dictionary-test-")
        self.root = Path(self.temp_dir.name)
        self.phase = Phase(name="hexagonal", point_group="6/mmm")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _session(self) -> WorkflowSession:
        session = WorkflowSession()
        session.data = SimpleNamespace(
            h=4,
            w=6,
            detector_px_size=1.0,
            detector_binning=1.0,
            sample_tilt_deg=70.0,
            detector_tilt_deg=0.0,
            azimuthal_deg=0.0,
            twist_deg=0.0,
        )
        session.master = SimpleNamespace(
            kind="kikuchipy",
            phase=self.phase,
            mp_signal=_FakeMasterSignal(),
            energy_kv=20.0,
        )
        return session

    def test_generation_is_uint8_disk_backed_and_save_removes_temp(self) -> None:
        session = self._session()
        rotations = Rotation.identity(3)
        with (
            patch.dict(os.environ, {"OVERLAP_EBSD_DICTIONARY_CACHE_DIR": str(self.root)}),
            patch("orix.sampling.get_sample_fundamental", return_value=rotations),
        ):
            cache = session._get_or_build_kikuchipy_dictionary(
                phase_id=1,
                resolution_deg=2.0,
                pc_bruker=np.array([0.5, 0.5, 0.6]),
                software_binning=1,
            )

        temporary_path = Path(cache.storage_path)
        self.assertTrue(temporary_path.exists())
        self.assertTrue(cache.owns_storage)
        self.assertEqual(cache.pattern_dtype, "uint8")
        self.assertEqual(cache.signal.data.dtype, np.dtype(np.uint8))

        output_path = self.root / "hex_dictionary_bin1_2deg.h5"
        session.save_dictionary(str(output_path))
        self.assertTrue(output_path.exists())
        self.assertFalse(temporary_path.exists())
        self.assertFalse(cache.owns_storage)
        self.assertEqual(cache.storage_path, str(output_path.resolve()))
        with h5py.File(output_path, "r") as h5:
            self.assertEqual(h5.attrs["format"], DICTIONARY_FORMAT_V2)
            self.assertEqual(h5["patterns"].dtype, np.dtype(np.uint8))
            self.assertGreater(h5["patterns"].chunks[0], 1)

    def test_v1_float32_dictionary_loads_lazily(self) -> None:
        session = self._session()
        path = self.root / "legacy.h5"
        patterns = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
        with h5py.File(path, "w") as h5:
            h5.attrs["format"] = DICTIONARY_FORMAT_V1
            h5.attrs["phase_id"] = 1
            h5.attrs["resolution_deg"] = 2.0
            h5.attrs["software_binning"] = 1
            h5.create_dataset("patterns", data=patterns, chunks=(1, 4, 6))
            h5.create_dataset("eulers_rad", data=np.zeros((3, 3)))
            h5.create_dataset("pc_bruker", data=np.array([0.5, 0.5, 0.6]))
            h5.create_dataset("crop_extent", data=np.array([0, 4, 0, 6]))

        session.load_dictionary(str(path))
        cache = session.dictionary_cache
        self.assertIsNotNone(cache)
        self.assertEqual(cache.pattern_dtype, "float32")
        self.assertTrue(hasattr(cache.signal.data, "chunks"))
        np.testing.assert_array_equal(cache.signal.data.compute(), patterns)

    def test_uint8_quantization_preserves_ncc_best_match(self) -> None:
        from kikuchipy.indexing import NormalizedCrossCorrelationMetric

        rng = np.random.default_rng(4)
        dictionary = rng.normal(size=(32, 8, 8)).astype(np.float32)
        experimental = dictionary[17] + rng.normal(scale=0.03, size=(8, 8)).astype(np.float32)
        minimum = dictionary.min(axis=(1, 2), keepdims=True)
        span = dictionary.max(axis=(1, 2), keepdims=True) - minimum
        dictionary_u8 = np.rint(255 * (dictionary - minimum) / span).astype(np.uint8)

        metric = NormalizedCrossCorrelationMetric(
            n_experimental_patterns=1,
            n_dictionary_patterns=dictionary.shape[0],
        )
        scores_float = metric(experimental[np.newaxis], dictionary).compute()[0]
        scores_u8 = metric(experimental[np.newaxis], dictionary_u8).compute()[0]
        self.assertEqual(int(np.argmax(scores_float)), 17)
        self.assertEqual(int(np.argmax(scores_u8)), 17)
        self.assertLess(abs(float(scores_float[17] - scores_u8[17])), 5e-4)


if __name__ == "__main__":
    unittest.main()
