"""Identity, competition and mixed-master regressions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from orix.crystal_map import Phase

from multistep_overlap_ebsd.core import WorkflowSession, MasterPatternModel, DictionaryCache
from multistep_overlap_ebsd.phases import PhaseRegistry, DictionaryAsset, DictionaryProvenance, file_fingerprint
from multistep_overlap_ebsd.phase_export import write_h5_phase_catalog, ang_phase_header
from multistep_overlap_ebsd.multiphase import phase_structure


class PhaseStructureTests(unittest.TestCase):
    def test_real_lattice_metadata(self):
        from diffpy.structure import Lattice, Structure
        phase = Phase(name="hexagonal", point_group="6/mmm",
                      structure=Structure(lattice=Lattice(2, 2, 4, 90, 90, 120)))
        np.testing.assert_allclose(phase_structure(phase)["lattice_angstrom_degrees"],
                                   [2, 2, 4, 90, 90, 120])

    def test_lattice_without_new_cell_parms_api(self):
        from diffpy.structure import Lattice, Structure
        class LegacyLattice(Lattice):
            def __getattribute__(self, name):
                if name == "cell_parms":
                    raise AttributeError("'Lattice' object has no attribute 'cell_parms'")
                return super().__getattribute__(name)
        structure = Structure(lattice=LegacyLattice(2, 3, 4, 80, 90, 100))
        phase = SimpleNamespace(structure=structure)
        self.assertFalse(hasattr(structure.lattice, "cell_parms"))
        np.testing.assert_allclose(phase_structure(phase)["lattice_angstrom_degrees"],
                                   [2, 3, 4, 80, 90, 100])


class RegistryTests(unittest.TestCase):
    def test_sparse_ids_duplicate_names_and_reordering_roundtrip(self):
        registry = PhaseRegistry()
        keys = [registry.add('same name', output_id=i, input_ids=[i]).key for i in (2, 7, 19)]
        registry.entries.reverse()
        restored = PhaseRegistry.from_json(registry.to_json())
        self.assertEqual([e.key for e in restored.entries], keys[::-1])
        self.assertEqual(restored.by_output_id(7).key, keys[1])
        with self.assertRaises(ValueError):
            restored.add('duplicate ID', output_id=7)
        with self.assertRaises(ValueError):
            restored.add('duplicate mapping', input_ids=[2])

    def test_asset_identity_survives_move_but_not_changed_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'
            a.write_bytes(b'master'); b.write_bytes(b'master')
            r = PhaseRegistry(); e = r.add('A'); r.bind_master(e.key, str(a))
            revision = r.search_revision
            r.bind_master(e.key, str(b), expected_sha256=e.master_sha256)
            self.assertEqual(r.search_revision, revision)
            a.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                r.bind_master(e.key, str(a), expected_sha256=e.master_sha256)
            self.assertEqual(e.master_path, str(b.resolve()))

    def test_snapshot_requires_verified_settings_and_has_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'asset'; p.touch()
            r = PhaseRegistry()
            e = r.add('A', output_id=5); r.bind_master(e.key, str(p))
            provenance = DictionaryProvenance.create(e.master_sha256, {}, {'pc':[.5,.5,.6]})
            a = DictionaryAsset(path=str(p), provenance=provenance)
            e.dictionaries.append(a); e.active_dictionary_key=a.key
            snap = r.snapshot({e.key:provenance})
            e.name = 'renamed'
            self.assertEqual(snap[0].key, e.key)
            with self.assertRaises(ValueError):
                r.snapshot({e.key:replace(provenance, settings_json='{}')})
            a.provenance=None
            self.assertEqual(e.status(provenance), 'Legacy / unverified')


class CompetitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = s = WorkflowSession()
        s.data = SimpleNamespace(count=4, rows=2, cols=2, h=4, w=4,
            source_type='h5oina', phases=np.array([2,7,19,0]), sample_tilt_deg=70., detector_tilt_deg=0.,
            azimuthal_deg=0., twist_deg=0., map_layers={'Phase':np.zeros((2,2))}, phase_symmetries={})
        s.current_eulers_rad = np.zeros((4,3)); s.current_phases=np.zeros(4,dtype=np.int32)
        s.current_pc_bruker=s.current_pc_custom=np.tile([.5,.5,.6],(4,1))
        s.last_scores_map=np.full((2,2),np.nan);s.indexed_mask=np.zeros(4,dtype=bool)
        self.scores={2:[.9,.3,.1,.6],7:[.2,.95,.2,.6],19:[.3,.1,.99,.2]}
        for pid in (2,7,19):
            e=s.phase_registry.add('same name',output_id=pid)
            path=Path(self.temp.name)/str(pid);path.touch();s.phase_registry.bind_master(e.key,str(path))
            mp=MasterPatternModel('kikuchipy',str(path),SimpleNamespace(data=np.ones((2,3))),None,Phase(name='A',point_group='m-3m'),energy_kv=20.)
            s.phase_masters[e.key]=mp
            cache=DictionaryCache(pid,5.,np.array([.5,.5,.6]),1,(0,4,0,4),(4,4),SimpleNamespace(data=np.ones((2,4,4))),2, storage_path=str(path))
            s.phase_dictionaries[e.key]=cache
            a=DictionaryAsset(path=str(path),provenance=s.phase_dictionary_provenance(e.key,cache));e.dictionaries.append(a);e.active_dictionary_key=a.key
        s.master=mp;s.dictionary_cache=cache
        s._signal_from_indices=lambda indices,**kw: np.asarray(indices)
        self.original_master=s.master

    def tearDown(self):
        self.session.close();self.temp.cleanup()

    def matcher(self, view, signal, *, cache, keep_n, **kwargs):
        indices=np.asarray(signal)
        scores=np.asarray(self.scores[cache.phase_id])[indices]
        eulers=np.full((len(indices),3),cache.phase_id/100)
        return eulers,scores,np.repeat(eulers[:,None,:],keep_n,axis=1),np.repeat(scores[:,None],keep_n,axis=1)

    def index(self, **kwargs):
        owner=self
        def match(view,*args,**kw):return owner.matcher(view,*args,**kw)
        with patch.object(WorkflowSession,'_dictionary_index_kikuchipy_signal',match), patch.object(WorkflowSession,'_signal_mask_for_dictionary_cache',return_value=None), patch.object(WorkflowSession,'_dictionary_n_per_iteration',return_value=2):
            return self.session.index_enabled_phases(np.arange(4),**kwargs)

    def test_three_phase_competition_retains_candidates_and_ties_ignore_order(self):
        self.index(keep_n=2)
        np.testing.assert_array_equal(self.session.current_phases,[2,7,19,2])
        self.assertEqual(len(self.session.phase_candidates),3)
        self.assertIs(self.session.master,self.original_master)
        self.session.phase_registry.entries.reverse();self.index(keep_n=2)
        np.testing.assert_array_equal(self.session.current_phases,[2,7,19,2])
        self.assertEqual(self.session.phase_score_gap[3],0.)
        np.testing.assert_array_equal(self.session.get_layer_map('Phase'),[[2,7],[19,2]])

    def test_cancellation_keeps_only_complete_phase_competitions(self):
        def before_complete(value, message):
            if value > 20:
                raise InterruptedError("stop")
        with self.assertRaises(InterruptedError):
            self.index(progress_callback=before_complete)
        self.assertFalse(self.session.indexed_mask.any())
        self.assertFalse(self.session.phase_candidates)
        def after_complete(value, message):
            if value == 100:
                raise InterruptedError("stop")
        with self.assertRaises(InterruptedError):
            self.index(progress_callback=after_complete)
        self.assertTrue(self.session.indexed_mask.all())
        np.testing.assert_array_equal(self.session.current_phases,[2,7,19,2])

    def test_refinement_can_promote_a_weaker_phase_without_mutating_selected_master(self):
        self.index()
        master=self.session.master
        def refine(view, indices, *, phase_id, **kwargs):
            view.last_scores_map.reshape(-1)[indices] = .99 if phase_id == 19 else .8
            return "refined"
        with patch.object(WorkflowSession,'refine_orientations_indices',refine):
            self.session.refine_enabled_phases(np.arange(4))
        np.testing.assert_array_equal(self.session.current_phases,[19]*4)
        self.assertIs(self.session.master,master)

    def test_imported_assignments_restrict_primary_search(self):
        self.scores[2]=[.9,.999,.999,.6]
        self.session.keep_imported_phase_assignments=True
        self.index()
        np.testing.assert_array_equal(self.session.current_phases,[2,7,19,2])

    def test_residual_can_choose_different_phase(self):
        self.session.current_phases[:]=2
        self.session._residual_signal_from_indices=lambda indices,**kw:np.asarray(indices)
        self.index(residual=True)
        np.testing.assert_array_equal(self.session.residual_phases,[2,7,19,2])
        np.testing.assert_array_equal(self.session.current_phases,[2]*4)

    def test_context_close_and_mutations_do_not_touch_parent(self):
        key=self.session.phase_registry.entries[0].key
        view=self.session._phase_context(key)
        view.current_phases[:]=99;view.close()
        np.testing.assert_array_equal(self.session.current_phases,np.zeros(4))
        self.assertIsNotNone(self.session.master)

    def test_mismatch_rejected_before_any_results_are_written(self):
        key=self.session.phase_registry.entries[0].key
        self.session.phase_dictionaries[key].resolution_deg=6.
        with self.assertRaises(ValueError):self.index()
        self.assertFalse(self.session.indexed_mask.any())

    def test_nonfinite_batch_does_not_commit(self):
        for phase in self.scores:self.scores[phase][1]=np.nan
        with self.assertRaises(ValueError):self.index()
        self.assertFalse(self.session.indexed_mask.any())

    def test_explicit_secondary_master_and_primary_batch_dispatch(self):
        self.session.current_phases[:]=[2,7,19,2]
        self.session._ensure_residual_state();self.session.residual_phases[:]=[19,2,7,7]
        entries=self.session.phase_registry.entries
        for e in entries:
            self.session.phase_masters[e.key].kind='legacy'
            self.session.phase_masters[e.key].projector=SimpleNamespace(project=lambda *args,v=e.output_id,**kw:np.full((4,4),v))
        self.session.data.rot_sd=np.eye(3);self.session.data.direction_cosines=None
        result=self.session._simulate_patterns_for_eulers(np.arange(4),np.zeros((4,3)))
        np.testing.assert_array_equal(result[:,0,0],[2,7,19,2])
        self.assertEqual(self.session._simulate_secondary_pattern(0,np.zeros(3))[0,0],19)


class CatalogTests(unittest.TestCase):
    def test_h5_and_ang_catalogs_preserve_sparse_ids_and_units(self):
        registry=PhaseRegistry()
        for pid,lattice,group in ((2,[3,3,3,90,90,90],'m-3m'),(7,[2,2,4,90,90,120],'6/mmm'),(19,[4,4,4,90,90,90],'m-3m')):
            registry.add('duplicate',output_id=pid,structure={'master':{'lattice_angstrom_degrees':lattice,'point_group':group}})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'phases.h5'
            with h5py.File(path,'w') as h5:
                write_h5_phase_catalog(h5,['7'],registry,{})
                np.testing.assert_allclose(h5['7/EBSD/Header/Phases/7/Lattice Angles'][()],np.deg2rad([90,90,120]))
                self.assertEqual(set(h5['7/EBSD/Header/Phases']),{'2','7','19'})
        header=ang_phase_header(['# GRID: SqrGrid','# Phase 1','# MaterialName old'],registry,{})
        self.assertIn('# Phase 19',header);self.assertNotIn('# Phase 1',header)
        self.assertIn('# GRID: SqrGrid',header)

    def test_missing_metadata_is_not_fabricated(self):
        registry=PhaseRegistry();registry.add('unknown')
        with self.assertRaises(ValueError):ang_phase_header([],registry,{})

class RealPipelineTests(unittest.TestCase):
    def test_three_masters_dictionary_index_refine_and_checkpoint(self):
        import kikuchipy as kp
        from orix.quaternion import Rotation
        from test_h5_indexed_input import IndexedInputTests
        helper=IndexedInputTests();helper.setUp()
        s=helper.session
        try:
            helper.write_input();helper.load_input()
            # Remove the imported single-phase placeholder to exercise new phases.
            s.phase_registry=PhaseRegistry()
            rotations=Rotation.from_euler([[.1,.2,.3],[.5,.8,1.1],[1.2,.4,.6]])
            for n in range(3):
                path=helper.root/f'master{n}.h5';path.write_bytes(bytes([n]))
                master=helper.master.deepcopy()
                master.data=np.random.default_rng(10+n).random(master.data.shape,dtype=np.float32)
                with patch('kikuchipy.load',return_value=master):
                    s.attach_phase_master(str(path))
                entry=s.phase_registry.entries[-1]
                with patch('orix.sampling.get_sample_fundamental',return_value=rotations):
                    s.generate_phase_dictionary(entry.key,resolution_deg=10.,software_binning=1)
                s.save_phase_dictionary(entry.key,str(helper.root/f'dictionary{n}.h5'))
            s.current_phases[:]=[1,2,3,1,2,3]
            s.current_eulers_rad[:]=s._eulers_from_kikuchipy_frame(rotations.to_euler())[[0,1,2,2,0,1]]
            simulated=s._simulate_patterns_for_eulers(np.arange(6),s.current_eulers_rad)
            with h5py.File(helper.input,'r+') as h5:
                values=np.stack([np.rint(255*(a-a.min())/(a.max()-a.min())).astype(np.uint8) for a in simulated])
                h5['7/EBSD/Data/Processed Patterns'][...]=values.reshape(2,3,8,10)
            # Selected-point indexing followed by automatic refinement must
            # also work before any batch has warmed up the compiled optimizer.
            s.dictionary_index_indices(np.array([0]),phase_id=1,keep_n=2,parallel_cores=1)
            s.refine_orientations_indices(np.array([0]),phase_id=1,maxfev=5,parallel_cores=1)
            self.assertEqual(s.current_phases[0],1)
            self.assertGreater(s.last_scores_map.flat[0],.99)
            s.dictionary_index_indices(np.arange(6),phase_id=1,keep_n=2,parallel_cores=1)
            np.testing.assert_array_equal(s.current_phases,[1,2,3,1,2,3])
            s.refine_orientations_indices(np.arange(6),phase_id=1,maxfev=5,parallel_cores=1)
            np.testing.assert_array_equal(s.current_phases,[1,2,3,1,2,3])
            self.assertTrue(np.all(s.last_scores_map>.99))
            s.save_workflow_state(str(helper.root/'workflow.npz'))
            with np.load(helper.root/'workflow.npz',allow_pickle=False) as state:
                self.assertEqual(len(PhaseRegistry.from_json(str(state['phase_registry_json'].item())).entries),3)
                self.assertEqual(int(state['workflow_schema_version']),4)
                self.assertEqual(sum(k.endswith('_indices') and 'primary_phase_' in k for k in state.files),3)
            reopened=WorkflowSession()
            sources={str(Path(model.path).resolve()): model.source_mp_signal for model in s.phase_masters.values()}
            def load_saved(path, **kwargs):
                return sources[str(Path(path).resolve())] if str(Path(path).resolve()) in sources else s.data.signal
            try:
                with patch('kikuchipy.load', side_effect=load_saved):
                    note=reopened.restore_workflow_state(str(helper.root/'workflow.npz'))
                self.assertEqual(len(reopened.phase_masters),3,note)
                self.assertEqual(len(reopened.phase_dictionaries),3,note)
                np.testing.assert_allclose(reopened.current_eulers_rad,s.current_eulers_rad)
                np.testing.assert_array_equal(reopened.current_phases,s.current_phases)
                self.assertEqual(set(reopened.phase_candidates),set(s.phase_candidates))
            finally:
                reopened.close()
            # Residual indexing/refinement must use all masters, too.
            s.compute_overlap_residual_indices(np.arange(6),fit_blur_gain=False,parallel_cores=1)
            s.index_overlap_residual_indices(np.arange(6),keep_n=2,parallel_cores=1)
            s.refine_overlap_residual_indices(np.arange(6),maxfev=5,parallel_cores=1)
            self.assertTrue(np.all(np.isfinite(s.last_residual_scores_map)))
        finally:
            helper.tearDown()

class RoundTripTests(unittest.TestCase):
    def test_three_phase_h5_primary_residual_and_map_only_reload(self):
        from test_h5_indexed_input import IndexedInputTests
        from multistep_overlap_ebsd.core import GeometryConfig
        from orix.quaternion import Rotation
        helper=IndexedInputTests();helper.setUp();s=helper.session
        reopened=WorkflowSession()
        try:
            helper.write_input();helper.load_input()
            # Use sparse IDs and duplicate phase names including hexagonal axes.
            s.phase_registry=PhaseRegistry()
            for pid,lattice,group in ((2,[3,3,3,90,90,90],'m-3m'),(7,[2,2,4,90,90,120],'6/mmm'),(19,[4,4,4,90,90,90],'m-3m')):
                e=s.phase_registry.add('duplicate',output_id=pid,structure={'master':{'lattice_angstrom_degrees':lattice,'point_group':group}})
                s.phase_masters[e.key]=MasterPatternModel('kikuchipy','',None,None,Phase(name='duplicate',point_group=group))
            s.master=next(iter(s.phase_masters.values()))
            s.current_phases[:]=[0,2,7,19,7,2]
            s.indexed_mask[:]=[False,True,True,True,True,True]
            s.last_scores_map[:]=.8
            with h5py.File(helper.input,'r+') as h5:
                header=h5.require_group('7/EBSD/Header')
                for axis,name in enumerate(('Pattern Center X','Pattern Center Y','Detector Distance')):
                    h5.create_dataset(f'7/EBSD/Data/{name}',data=s.current_pc_custom[:,axis])
                for key,value in {'X Cells':3,'Y Cells':2,'Pattern Height':8,'Pattern Width':10,'Tilt Angle':np.deg2rad(70.)}.items():
                    header.create_dataset(key,data=value)
            primary=helper.root/'primary.h5oina'
            s.export_primary_roi_results((0,0,2,3),str(primary))
            # No pattern payload: exercise actual loader fallback, not a fake pattern signal.
            reopened.load_input(str(primary),None,GeometryConfig())
            self.assertFalse(reopened.data.patterns_available)
            np.testing.assert_array_equal(reopened.current_phases,s.current_phases)
            np.testing.assert_array_equal(reopened.indexed_mask,s.indexed_mask)
            self.assertEqual([e.output_id for e in reopened.phase_registry.entries],[2,7,19])
            self.assertEqual(reopened.get_ipf_color_map().shape,(2,3,3))
            with self.assertRaisesRegex(RuntimeError,'maps only'):
                reopened._processed_pattern_at(1)
            original=Rotation.from_euler(s.current_eulers_rad[1:])
            restored=Rotation.from_euler(reopened.current_eulers_rad[1:])
            np.testing.assert_allclose(original.angle_with(restored),0,atol=2e-7)
            # Re-export again without masters, preserving catalog and one-time frame conversion.
            repeated=helper.root/'repeated.h5oina'
            reopened.export_primary_roi_results((0,0,2,3),str(repeated))
            reopened.close();reopened=WorkflowSession()
            reopened.load_input(str(repeated),None,GeometryConfig())
            np.testing.assert_allclose(original.angle_with(Rotation.from_euler(reopened.current_eulers_rad[1:])),0,atol=2e-7)
            s._ensure_residual_state()
            s.residual_eulers_rad[:]=s.current_eulers_rad
            s.residual_phases[:]=[0,19,2,7,19,7]
            s.last_residual_scores_map[:]=.7
            # Availability is an independent mask, not inferred from nonzero phase IDs.
            with patch.object(s,'_residual_pattern_indices',return_value=np.arange(1,6)):
                residual=helper.root/'residual_export.h5oina'
                s.export_residual_roi_results((0,0,2,3),str(residual))
            reopened.close();reopened=WorkflowSession()
            reopened.load_input(str(residual),None,GeometryConfig())
            np.testing.assert_array_equal(reopened.current_phases,s.residual_phases)
            self.assertIsNone(reopened.residual_eulers_rad)
            np.testing.assert_allclose(reopened.last_scores_map.reshape(-1)[1:],.7)
            np.testing.assert_allclose(original.angle_with(Rotation.from_euler(reopened.current_eulers_rad[1:])),0,atol=2e-7)
        finally:
            reopened.close();helper.tearDown()

class MixtureTests(unittest.TestCase):
    def test_known_same_and_different_phase_mixtures_and_single_component_control(self):
        from multistep_overlap_ebsd import core
        yy,xx=np.indices((18,20))
        primary=np.sin(xx/2).astype(np.float32)
        secondary=np.cos(yy/3).astype(np.float32)
        weights=np.ones_like(primary)
        params=np.array([0.,1.,1.,1.,1.,1.,0.,0.])
        def mixture_fit(exp,p,s,w,**kwargs):
            fit=core._evaluate_overlap_mixture_pattern(exp,p,s,w,params);fit.success=True
            return fit
        def primary_fit(exp,sim,w,**kwargs):
            return core._overlap_point_result_from_raw_patterns(0,0,0,exp,sim,w,fit_blur_gain=False,
                     fit_maxiter=1,fit_popsize=4,fit_bounds=None)
        for same_phase in (False,True):
            for fraction in (0.,.2,.5,.8):
                with self.subTest(same_phase=same_phase,fraction=fraction):
                    s=WorkflowSession()
                    try:
                        a=s.phase_registry.add('A');b=a if same_phase else s.phase_registry.add('B')
                        s.data=SimpleNamespace(count=1,rows=1,cols=1,h=18,w=20,source_type='h5oina')
                        s.current_phases=np.array([a.output_id]);s.residual_phases=np.array([b.output_id])
                        s.current_eulers_rad=np.zeros((1,3));s.residual_eulers_rad=np.ones((1,3))
                        s.current_pc_bruker=s.current_pc_custom=np.full((1,3),.5)
                        s.master=SimpleNamespace();s.phase_masters={a.key:s.master,b.key:s.master}
                        s.last_scores_map=np.array([[.8]]);s.last_residual_scores_map=np.array([[.8]])
                        s.indexed_mask=np.ones(1,dtype=bool)
                        exp=(1-fraction)*core._normalize_weighted(primary,weights)+fraction*core._normalize_weighted(secondary,weights)
                        s._processed_pattern_at=lambda index:exp
                        s._overlap_weights=lambda:weights
                        s._simulate_pattern_for_euler=lambda i,e:primary if np.allclose(e,0) else secondary
                        s._simulate_secondary_pattern=lambda i,e:secondary
                        s._phase_context=lambda key,**kwargs:SimpleNamespace(_simulate_pattern_for_euler=s._simulate_pattern_for_euler)
                        original=core._overlap_point_result_from_raw_patterns
                        def single(*args,**kw):
                            kw['fit_blur_gain']=False
                            return original(*args,**kw)
                        with patch.object(core,'_fit_overlap_mixture_pattern',mixture_fit), patch.object(core,'_overlap_point_result_from_raw_patterns',single):
                            result=s.fit_overlap_mixture_point(0,store_result=False)
                        self.assertEqual(result.primary_phase_key,a.key)
                        self.assertEqual(result.secondary_phase_key,b.key)
                        self.assertAlmostEqual(result.secondary_fraction,fraction,places=5)
                        self.assertEqual(result.overlap_accepted,fraction>0)
                        restored=s._mixture_result_from_metadata(s._mixture_result_metadata(s._strip_overlap_mixture_result(result)))
                        self.assertEqual(restored.overlap_accepted,result.overlap_accepted)
                        self.assertEqual(restored.secondary_phase_key,b.key)
                    finally:
                        s.phase_masters={};s.close()

class StepFourExportTests(unittest.TestCase):
    def test_fitted_phase_pair_is_exported_separately_from_indexing_assignments(self):
        from multistep_overlap_ebsd.core import OverlapMixtureResult
        from test_workflow_state import WorkflowStateTests
        with tempfile.TemporaryDirectory() as directory:
            s=WorkflowSession()
            try:
                s.data=WorkflowStateTests._data()
                s.data.x_coords=np.tile(np.arange(3),2);s.data.y_coords=np.repeat(np.arange(2),3)
                s.data.scan_unit='um';s.data.step_x=s.data.step_y=1.;s.data.pc_output_convention='oxford'
                a=s.phase_registry.add('A',output_id=2);b=s.phase_registry.add('B',output_id=7);c=s.phase_registry.add('C',output_id=19)
                s.current_phases=np.full(6,2);s.residual_phases=np.full(6,7)
                result=OverlapMixtureResult(index=1,row=0,col=1,primary_fraction=.6,secondary_fraction=.4,
                    primary_coefficient=.6,secondary_coefficient=.4,ncc_mixture=.99,residual_rms=.01,
                    old_primary_ncc=.7,old_secondary_ncc=.6,experimental=None,primary_simulated=None,
                    secondary_simulated=None,combined_simulated=None,residual=None,
                    primary_euler_rad=np.zeros(3),secondary_euler_rad=np.ones(3),primary_phase_key=c.key,
                    secondary_phase_key=a.key,overlap_accepted=True,single_component_ncc=.8,overlap_ncc_improvement=.19)
                s.overlap_mixture_results[1]=result
                path=Path(directory)/'step4.h5'
                s.export_overlap_optimization_results(str(path),(0,0,2,3))
                with h5py.File(path,'r') as h5:
                    self.assertEqual(h5['Maps/primary_phase'][0,1],19)
                    self.assertEqual(h5['Maps/secondary_phase'][0,1],2)
                    self.assertEqual(h5['Maps/indexing_primary_phase'][0,1],2)
                    self.assertEqual(h5['Maps/indexing_secondary_phase'][0,1],7)
                    self.assertEqual(h5['Maps/overlap_accepted'][0,1],1)
                    self.assertEqual(h5['Maps/overlap_accepted'][0,0],-1)
                    self.assertEqual(h5['Point Results/primary_phase_key'].asstr()[0],c.key)
            finally:
                s.close()

class ExternalAngTests(unittest.TestCase):
    def test_orix_reads_generated_cubic_hexagonal_phase_catalog(self):
        import orix.io
        registry=PhaseRegistry()
        for pid,lattice,group in ((2,[3,3,3,90,90,90],'m-3m'),(7,[2,2,4,90,90,120],'6/mmm'),(19,[4,4,4,90,90,90],'m-3m')):
            registry.add('duplicate',output_id=pid,structure={'master':{'lattice_angstrom_degrees':lattice,'point_group':group}})
        header=ang_phase_header(['# GRID: SqrGrid','# XSTEP: 1','# YSTEP: 1','# NCOLS_ODD: 3','# NCOLS_EVEN: 3','# NROWS: 2'],registry,{})
        values=np.zeros((6,10));values[:,3]=np.tile(np.arange(3),2);values[:,4]=np.repeat(np.arange(2),3)
        values[:,5]=100;values[:,6]=.8;values[:,7]=[2,7,19,2,7,19]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'phases.ang'
            with path.open('w') as stream:
                stream.write('\n'.join(header)+'\n');np.savetxt(stream,values)
            xmap=orix.io.load(path)
            np.testing.assert_array_equal(xmap.phase_id,values[:,7])
            self.assertEqual(xmap.phases[7].point_group.name,'622')
            np.testing.assert_allclose(phase_structure(xmap.phases[7])["lattice_angstrom_degrees"],[2,2,4,90,90,120])
