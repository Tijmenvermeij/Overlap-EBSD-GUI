from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import kikuchipy as kp
import numpy as np
from orix.crystal_map import Phase

from multistep_overlap_ebsd.core import (
    DICTIONARY_FORMAT_V2,
    MASTER_ENERGY_MODE_GLOBAL,
    MasterPatternModel,
    WorkflowSession,
    _emsoft_global_energy_weights,
    _master_signal_with_energy_weights,
)


class MasterEnergyWeightingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="overlap-ebsd-energy-test-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_detector_projected_mc_weights_preserve_energy_ratios(self) -> None:
        path = self.root / "synthetic_emsoft.h5"
        energy_ratios = np.asarray([1.0, 2.0, 7.0], dtype=np.float64)
        accum_e = np.ones((9, 9, 3), dtype=np.float64) * energy_ratios.reshape(1, 1, 3)
        with h5py.File(path, "w") as h5:
            h5.create_dataset("EMData/MCOpenCL/accum_e", data=accum_e)
            nml = h5.require_group("NMLparameters/MCCLNameList")
            nml.create_dataset("Ehistmin", data=np.asarray([18.0]))
            nml.create_dataset("Ebinsize", data=np.asarray([1.0]))
            nml.create_dataset("EkeV", data=np.asarray([20.0]))
            nml.create_dataset("numsx", data=np.asarray([9]))

        detector = kp.detectors.EBSDDetector(
            shape=(8, 10),
            pc=(0.46, 0.53, 0.62),
            convention="bruker",
            px_size=1.0,
            binning=1,
            sample_tilt=70.0,
        )
        energies, weights = _emsoft_global_energy_weights(
            str(path),
            detector,
            np.asarray([18.0, 19.0, 20.0]),
            20.0,
        )
        np.testing.assert_array_equal(energies, [18.0, 19.0, 20.0])
        np.testing.assert_allclose(weights, energy_ratios / energy_ratios.sum(), atol=1e-12)

    def test_weighted_master_is_collapsed_once_before_projection(self) -> None:
        data = np.empty((2, 3, 5, 7), dtype=np.float32)
        for hemisphere in range(2):
            for energy_index, value in enumerate((10.0, 20.0, 40.0)):
                data[hemisphere, energy_index] = value + hemisphere
        source = kp.signals.EBSDMasterPattern(
            data,
            projection="lambert",
            hemisphere="both",
        )
        for axis in source.axes_manager.navigation_axes:
            if int(axis.index_in_array) == 0:
                axis.name = "hemisphere"
            elif int(axis.index_in_array) == 1:
                axis.name = "energy"
                axis.units = "keV"
                axis.offset = 18.0
                axis.scale = 1.0

        weighted = _master_signal_with_energy_weights(
            source,
            np.asarray([18.0, 19.0, 20.0]),
            np.asarray([0.2, 0.3, 0.5]),
        )
        expected = 0.2 * data[:, 0] + 0.3 * data[:, 1] + 0.5 * data[:, 2]
        np.testing.assert_allclose(weighted.data, expected, atol=1e-6)
        self.assertEqual(weighted.data.shape, (2, 5, 7))
        self.assertFalse(
            any("energy" in str(getattr(axis, "name", "")).lower() for axis in weighted.axes_manager.navigation_axes)
        )

    def test_loading_dictionary_restores_global_model_on_master(self) -> None:
        data = np.ones((2, 3, 5, 7), dtype=np.float32)
        source = kp.signals.EBSDMasterPattern(
            data,
            projection="lambert",
            hemisphere="both",
            phase=Phase(name="hexagonal", point_group="6/mmm"),
        )
        for axis in source.axes_manager.navigation_axes:
            if int(axis.index_in_array) == 0:
                axis.name = "hemisphere"
            elif int(axis.index_in_array) == 1:
                axis.name = "energy"
                axis.units = "keV"
                axis.offset = 18.0
                axis.scale = 1.0

        session = WorkflowSession()
        session.data = SimpleNamespace(h=4, w=6)
        session.master = MasterPatternModel(
            kind="kikuchipy",
            path=str(self.root / "master.h5"),
            mp_signal=source,
            projector=None,
            phase=source.phase,
            energy_kv=20.0,
            source_mp_signal=source,
        )
        dictionary_path = self.root / "weighted_dictionary.h5"
        with h5py.File(dictionary_path, "w") as h5:
            h5.attrs["format"] = DICTIONARY_FORMAT_V2
            h5.attrs["phase_id"] = 1
            h5.attrs["resolution_deg"] = 2.0
            h5.attrs["software_binning"] = 1
            h5.attrs["master_energy_mode"] = MASTER_ENERGY_MODE_GLOBAL
            h5.create_dataset("patterns", data=np.ones((2, 4, 6), dtype=np.uint8))
            h5.create_dataset("eulers_rad", data=np.zeros((2, 3), dtype=np.float64))
            h5.create_dataset("pc_bruker", data=np.asarray([0.5, 0.5, 0.6]))
            h5.create_dataset("crop_extent", data=np.asarray([0, 4, 0, 6]))
            h5.create_dataset("master_energy_values_kv", data=np.asarray([18.0, 19.0, 20.0]))
            h5.create_dataset("master_energy_weights", data=np.asarray([0.2, 0.3, 0.5]))
            h5.create_dataset("master_energy_reference_pc_bruker", data=np.asarray([0.5, 0.5, 0.6]))

        session.load_dictionary(str(dictionary_path))
        self.assertEqual(session.master.energy_mode, MASTER_ENERGY_MODE_GLOBAL)
        self.assertEqual(session.master.mp_signal.data.shape, (2, 5, 7))
        np.testing.assert_allclose(session.master.energy_weights, [0.2, 0.3, 0.5])
        self.assertEqual(session.dictionary_cache.master_energy_mode, MASTER_ENERGY_MODE_GLOBAL)


if __name__ == "__main__":
    unittest.main()
