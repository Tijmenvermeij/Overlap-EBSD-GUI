"""Fresh-session import of pattern-matched H5OINA as primary indexing."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import kikuchipy as kp
import numpy as np
from orix.crystal_map import Phase

from multistep_overlap_ebsd.core import (
    DICTIONARY_FORMAT_V2, GeometryConfig, H5OINA_NCC_DATASET, H5OINA_EXPORT_DATA_GROUP,
    WorkflowSession,
)
from multistep_overlap_ebsd.gui import MultiStepOverlapGUI


class IndexedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'residual.h5oina'
        self.patterns = np.arange(6*8*10, dtype=np.uint8).reshape(2, 3, 8, 10)
        self.eulers = np.arange(18, dtype=np.float64).reshape(6, 3)/10
        self.phases = np.ones(6, dtype=np.int32)
        self.scores = np.array([.5, .6, .7, .8, .9, .4], dtype=np.float32)
        self.session = WorkflowSession()
        self.master_path = self.root / 'master.h5'
        self.master_path.touch()
        self.master = kp.signals.EBSDMasterPattern(
            np.random.default_rng(1).random((2, 3, 11, 11), dtype=np.float32),
            projection='lambert', hemisphere='both', phase=Phase(name='Cu', point_group='m-3m'))
        for axis in self.master.axes_manager.navigation_axes:
            if axis.index_in_array == 1:
                axis.name, axis.offset, axis.scale = 'energy', 18., 1.

    def tearDown(self):
        self.session.close()
        self.temp.cleanup()

    def write_input(self, *, export=None, ncc=True):
        with h5py.File(self.input, 'w') as h5:
            h5.create_dataset('7/EBSD/Data/Processed Patterns', data=self.patterns)
            h5.create_dataset('7/EBSD/Data/Euler', data=self.eulers)
            h5.create_dataset('7/EBSD/Data/Phase', data=self.phases)
            if ncc:
                h5.create_dataset('7/'+H5OINA_NCC_DATASET, data=self.scores)
            if export:
                group = h5.require_group('7/'+H5OINA_EXPORT_DATA_GROUP)
                group.attrs['Export Type'] = export
                group.create_dataset('ROI Mask', data=np.ones(6, dtype=np.uint8))
                group.create_dataset('Primary NCC', data=np.full(6, .99))
                group.create_dataset('Residual NCC', data=np.full(6, .11))
                group.create_dataset('Residual Pattern Available', data=[1, 1, 0, 1, 1, 1])
                group.create_dataset('Primary Fit Scale', data=np.full(6, 2.))
                group.create_dataset('Mixture NCC', data=np.full(6, .98))

    def load_input(self):
        detector = SimpleNamespace(sample_tilt=70., tilt=0., azimuthal=0., twist=0., px_size=1., binning=1.,
            pc_bruker=lambda: np.full((6, 3), .5), pc_oxford=lambda: np.full((6, 3), .6))
        axis = SimpleNamespace(scale=1., units='um')
        signal = SimpleNamespace(data=self.patterns, detector=detector, xmap=None,
                                 axes_manager=SimpleNamespace(navigation_axes=[axis, axis]))
        with patch('kikuchipy.load', return_value=signal):
            return self.session.load_input(str(self.input), None, GeometryConfig())

    def load_master(self, **kwargs):
        with patch('kikuchipy.load', return_value=self.master):
            self.session.load_master(str(self.master_path), **kwargs)

    def write_dictionary(self, energy=20.):
        path = self.root / 'dictionary.h5'
        with h5py.File(path, 'w') as h5:
            h5.attrs.update(format=DICTIONARY_FORMAT_V2, phase_id=1, resolution_deg=1.4,
                            software_binning=1, master_energy_mode='highest')
            h5.create_dataset('patterns', data=self.patterns.reshape(6, 8, 10)[:2])
            h5.create_dataset('eulers_rad', data=self.eulers[:2])
            h5.create_dataset('pc_bruker', data=np.full(3, .5))
            h5.create_dataset('crop_extent', data=[0, 8, 0, 10])
            h5.create_dataset('master_energy_values_kv', data=[energy])
        return path

    def test_standard_ncc_without_gui_metadata_is_primary_indexing(self):
        self.scores = np.array([-.2, 0., .5, .4, np.nan, np.inf], dtype=np.float32).reshape(2, 3)
        self.phases[2] = 0
        self.eulers[3, 0] = np.nan
        self.write_input()
        message = self.load_input()
        np.testing.assert_array_equal(self.session.indexed_mask, [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(self.session.last_indexed_indices, [0, 1])
        self.assertAlmostEqual(self.session.get_primary_index_ncc(0), -.2)
        self.assertEqual(self.session.get_primary_index_ncc(1), 0.)
        self.assertIsNone(self.session.get_primary_index_ncc(2))
        self.assertTrue(np.isnan(self.session.last_scores_map.reshape(-1)[2:]).all())
        self.assertIn('Pattern Matching NCC', message)
        self.assertIn('can be skipped', message)

    def test_residual_export_is_fresh_primary_with_no_parent_residual_or_mixture_state(self):
        self.write_input(export='residual')
        self.load_input()
        s = self.session
        np.testing.assert_array_equal(s.current_eulers_rad, self.eulers)
        np.testing.assert_array_equal(s._processed_patterns_from_indices(np.array([1, 0])),
                                      self.patterns.reshape(6, 8, 10)[[1, 0]])
        mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(s.indexed_mask, mask)
        # The stored patterns' standard NCC wins over both parent-run NCC maps.
        np.testing.assert_array_equal(s.last_scores_map.reshape(-1)[mask], self.scores[mask])
        self.assertIsNone(s.master)
        self.assertIsNone(s.residual_eulers_rad)
        self.assertIsNone(s.indexed_candidate_eulers_rad)
        self.assertIsNone(s.residual_candidate_eulers_rad)
        self.assertIsNone(s.residual_pattern_output_path)
        self.assertFalse(s.residual_point_results)
        self.assertFalse(s.overlap_mixture_results)
        before = s.last_scores_map.copy()
        self.load_master()
        s.load_dictionary(str(self.write_dictionary()))
        np.testing.assert_array_equal(s.last_scores_map, before)
        np.testing.assert_array_equal(s.indexed_mask, mask)
        gui = SimpleNamespace(session=s, _residual_ncc_threshold=lambda: .55)
        np.testing.assert_array_equal(MultiStepOverlapGUI._primary_threshold_mask(gui),
                                      [[1, 0, 1], [0, 0, 1]])

    def test_roi_and_primary_exports_do_not_use_stale_residual_availability(self):
        self.write_input(export='primary')
        with h5py.File(self.input, 'r+') as h5:
            h5['7/'+H5OINA_EXPORT_DATA_GROUP+'/ROI Mask'][0] = 0
        self.load_input()
        np.testing.assert_array_equal(self.session.indexed_mask, [0, 1, 1, 1, 1, 1])

    def test_missing_or_wrong_size_ncc_leaves_ordinary_input_unindexed(self):
        for scores in (None, np.ones(5), np.ones(7)):
            self.write_input(ncc=False)
            if scores is not None:
                with h5py.File(self.input, 'r+') as h5:
                    h5.create_dataset('7/'+H5OINA_NCC_DATASET, data=scores)
            self.load_input()
            self.assertFalse(self.session.indexed_mask.any())
            self.assertIsNone(self.session.last_indexed_indices)
            self.assertTrue(np.isnan(self.session.last_scores_map).all())

    def test_other_analysis_ncc_is_not_borrowed(self):
        self.write_input(ncc=False)
        with h5py.File(self.input, 'r+') as h5:
            h5.create_dataset('8/'+H5OINA_NCC_DATASET, data=self.scores)
        self.load_input()
        self.assertFalse(self.session.indexed_mask.any())

    def test_root_level_ncc_and_malformed_roi_mask(self):
        self.write_input()
        with h5py.File(self.input, 'r+') as h5:
            h5.move('7/EBSD', 'EBSD')
            h5.move('7/Data Processing', 'Data Processing')
            del h5['7']
        self.load_input()
        self.assertTrue(self.session.indexed_mask.all())
        with h5py.File(self.input, 'r+') as h5:
            group = h5.require_group(H5OINA_EXPORT_DATA_GROUP)
            group.create_dataset('ROI Mask', data=np.ones(5))
        self.load_input()
        self.assertFalse(self.session.indexed_mask.any())

    def test_gui_export_without_standard_ncc_keeps_backward_compatibility(self):
        self.write_input(export='residual', ncc=False)
        self.load_input()
        np.testing.assert_allclose(self.session.last_scores_map.reshape(-1)[[0, 1, 3, 4, 5]], .11)

    def test_initial_weighted_master_keeps_imported_indexing_but_model_changes_invalidate(self):
        self.write_input()
        self.load_input()
        weights = dict(energy_values_kv=np.array([18., 19., 20.]), energy_weights=np.array([.2, .3, .5]),
                       reference_pc_bruker=np.full(3, .5))
        self.load_master(energy_mode='global_weighted', energy_values_kv=weights['energy_values_kv'],
                         energy_weights=weights['energy_weights'], energy_reference_pc_bruker=np.full(3, .5))
        self.session.set_master_energy_model('global_weighted', **weights)
        self.assertTrue(self.session.indexed_mask.all())
        self.session.set_master_energy_model('highest')
        self.assertFalse(self.session.indexed_mask.any())

    def test_changed_dictionary_energy_or_replacement_master_invalidates_scores(self):
        for change in ('energy', 'master', 'conditioning'):
            self.session.close()
            self.session = WorkflowSession()
            self.write_input()
            self.load_input()
            self.load_master()
            self.assertTrue(self.session.indexed_mask.all())
            if change == 'energy':
                self.session.load_dictionary(str(self.write_dictionary(energy=19.)))
            elif change == 'master':
                self.load_master()
            else:
                self.session.set_dynamic_background(True, std_px=3.)
            self.assertFalse(self.session.indexed_mask.any())
            self.assertTrue(np.isnan(self.session.last_scores_map).all())


if __name__ == '__main__':
    unittest.main()
