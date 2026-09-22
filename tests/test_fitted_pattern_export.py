from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from multistep_overlap_ebsd import core
from multistep_overlap_ebsd.fitted_export import export_fitted_patterns, full_range_pattern


class FittedPatternExportTests(unittest.TestCase):
    def setUp(self):
        from test_h5_indexed_input import IndexedInputTests
        self.helper = helper = IndexedInputTests(); helper.setUp()
        helper.write_input(); helper.load_input(); helper.load_master()
        self.s = helper.session
        self.raw = np.random.default_rng(23).random((2, 8, 10), dtype=np.float32)
        self.s._simulate_pattern_for_euler = lambda i, e: self.raw[0]
        self.s._simulate_secondary_pattern = lambda i, e: self.raw[1]
        self.s._phase_context = lambda key, **kwargs: type('View', (), {
            '_simulate_pattern_for_euler': lambda view, i, e: self.raw[int(key == 'second')]})()
        self.s.phase_registry.entries[0].key = 'first'
        entry = self.s.phase_registry.add('Second', output_id=2)
        entry.key = 'second'
        for phase in self.s.phase_registry.entries:
            phase.structure = {'lattice_angstrom_degrees':[3.6,3.6,3.6,90.,90.,90.], 'point_group':'m-3m'}
        # Reuse a test master for catalogue serialization, not projection.
        self.s.phase_masters = {'first': self.s.master, 'second': self.s.master}
        result = core._overlap_mixture_result_from_raw_patterns(2, 0, 2,
            self.s._processed_pattern_at(2), *self.raw, self.s._overlap_weights(),
            primary_euler_rad=np.array([.2,.4,.6]), secondary_euler_rad=np.array([.7,.8,.9]),
            old_primary_ncc=.5, old_secondary_ncc=.4, fit_maxiter=2, fit_popsize=2, fit_bounds=None)
        self.result = replace(result, primary_phase_key='second', secondary_phase_key='first',
                              overlap_accepted=True, primary_coefficient=.8, secondary_coefficient=.3)
        self.s.overlap_mixture_results = {2:self.result}

    def tearDown(self):
        self.helper.tearDown()

    def test_full_scan_exports_use_optimized_solution_and_black_nonoverlap(self):
        s = self.s; r = self.result
        original = hashlib.sha256(self.helper.input.read_bytes()).hexdigest()
        angles = s.current_eulers_rad.copy()
        for secondary in (False, True):
            for subtract_residual in (False, True):
                out = self.helper.root / f'fit-{secondary}-{subtract_residual}.h5oina'
                export_fitted_patterns(s, out, secondary=secondary, accepted_only=True,
                                       primary_subtract_residual=subtract_residual)
                with h5py.File(out) as h5:
                    patterns = h5['7/EBSD/Data/Processed Patterns'][()]
                    phases = h5['7/EBSD/Data/Phase'][()].reshape(-1)
                    eulers = h5['7/EBSD/Data/Euler'][()].reshape(-1,3)
                    self.assertEqual(patterns.shape, (6,8,10))
                    self.assertEqual(int(phases[2]), 1 if secondary else 2)
                    np.testing.assert_allclose(eulers[2], r.secondary_euler_rad if secondary else r.primary_euler_rad, atol=1e-6)
                    if secondary:
                        self.assertFalse(patterns[[0,1,3,4,5]].any())
                        self.assertFalse(phases[[0,1,3,4,5]].any())
                    else:
                        np.testing.assert_array_equal(patterns[0], full_range_pattern(s._pattern_at(0), np.uint8))
                    self.assertEqual(patterns[2].min(),0)
                    self.assertEqual(patterns[2].max(),255)
                    weights=s._overlap_weights(); gain=core._power_gain_map((8,10),r.gain_params,r.ellipse_params)
                    p=core._normalize_weighted(gaussian_filter(self.raw[1],r.fitted_sigma)*gain,weights)
                    q=core._normalize_weighted(gaussian_filter(self.raw[0],r.fitted_sigma)*gain,weights)
                    exp=core._normalize_weighted(s._processed_pattern_at(2),weights)
                    expected=exp-.8*p if secondary else .8*p if subtract_residual else exp-.3*q
                    np.testing.assert_array_equal(patterns[2],full_range_pattern(expected,np.uint8,weights>0))
                    restored = core._h5_primary_indexing_state(h5, roots=['7'], rows=2,cols=3,phases=phases,eulers=eulers)
                    self.assertTrue(restored[1][2])
                    if secondary: self.assertEqual(restored[1].sum(),1)
        self.assertEqual(hashlib.sha256(self.helper.input.read_bytes()).hexdigest(),original)
        np.testing.assert_array_equal(s.current_eulers_rad,angles)

    def test_rejected_fit_and_cancellation(self):
        s=self.s
        s.overlap_mixture_results[2]=replace(self.result, overlap_accepted=False)
        out=self.helper.root/'black.h5oina'
        export_fitted_patterns(s,out,secondary=True,accepted_only=True,primary_subtract_residual=False)
        with h5py.File(out) as h5:
            self.assertFalse(h5['7/EBSD/Data/Processed Patterns'][()].any())
        original=out.read_bytes()
        def cancel(value,message):
            if value >= 10: raise InterruptedError()
        with self.assertRaises(InterruptedError):
            export_fitted_patterns(s,out,secondary=True,accepted_only=False,primary_subtract_residual=False,progress_callback=cancel)
        self.assertEqual(out.read_bytes(),original)
        self.assertEqual(list(out.parent.glob('.black-*.h5oina')),[])
        with self.assertRaises(ValueError):
            export_fitted_patterns(s,self.helper.input,secondary=False,accepted_only=True,primary_subtract_residual=False)

    def test_scaling_constant_nonfinite_and_uint16(self):
        self.assertFalse(full_range_pattern(np.ones((2,2)),np.uint8).any())
        np.testing.assert_array_equal(full_range_pattern([[float('nan'),-2],[0,2]],np.uint16),[[0,0],[32768,65535]])

    def test_emsoft_catalog_converts_nm_to_angstrom_without_changing_provenance(self):
        from types import SimpleNamespace
        from multistep_overlap_ebsd.phase_export import phase_metadata
        path = self.helper.root / 'emsoft.h5'
        with h5py.File(path,'w') as h5:
            h5.create_dataset('CrystalData/LatticeParameters', data=[.411,.411,.411,90,90,90])
        entry=self.s.phase_registry.entries[1]
        before=dict(entry.structure)
        metadata=phase_metadata(entry,SimpleNamespace(path=str(path),phase=self.s.master.phase))
        np.testing.assert_allclose(metadata['Lattice Dimensions'],[4.11,4.11,4.11])
        self.assertEqual(entry.structure,before)

    def test_uint16_source_preserves_full_16bit_range(self):
        with h5py.File(self.helper.input,'r+') as h5:
            name='7/EBSD/Data/Processed Patterns'
            values=h5[name][()].astype(np.uint16)*200
            del h5[name]
            h5.create_dataset(name,data=values)
        out=self.helper.root/'sixteen.h5oina'
        export_fitted_patterns(self.s,out,secondary=False,accepted_only=True,primary_subtract_residual=True)
        with h5py.File(out) as h5:
            patterns=h5['7/EBSD/Data/Processed Patterns']
            self.assertEqual(patterns.dtype,np.dtype(np.uint16))
            self.assertEqual(patterns[0].max(),65535)
            self.assertEqual(patterns[2].max(),65535)
