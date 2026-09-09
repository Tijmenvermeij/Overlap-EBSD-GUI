from __future__ import annotations

import threading
import tkinter as tk
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from multistep_overlap_ebsd.gui import MultiStepOverlapGUI
from multistep_overlap_ebsd.cpu_fitting import FIT_METHOD_LABELS


class Value:
    def __init__(self, value):
        self.value = value
        self.owner = threading.get_ident()

    def get(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("Worker accessed a UI variable")
        return self.value

    def set(self, value):
        if threading.get_ident() != self.owner:
            raise AssertionError("Worker mutated a UI variable")
        self.value = value


def bind(stub, *names):
    for name in names:
        setattr(stub, name, MethodType(getattr(MultiStepOverlapGUI, name), stub))
    return stub


class GuiActionTests(unittest.TestCase):
    def point_editor_stub(self):
        interpreter = tk.Tcl()
        eulers = np.array([13.1234567890123, 74.9876543210987, 226.123456789012])
        pc = np.array([0.462345678901234, 1.103456789012345, 0.579876543210987])
        state = {'euler_deg': eulers, 'pc_custom': pc, 'phase': 1, 'pc_convention': 'oxford'}
        jobs = []
        gui = SimpleNamespace(
            busy=False, session=SimpleNamespace(data=object(), get_point_state=Mock(return_value=state),
                                               set_point_state=Mock(return_value='updated')),
            index_var=Value(2), status_var=Value(''), pc_conv_label_var=Value(''),
            _run_threaded=lambda action, **kwargs: jobs.append((action, kwargs)),
            _mark_calibration_unapplied=Mock(),
        )
        for name in ('euler1_deg_var', 'euler2_deg_var', 'euler3_deg_var', 'pcx_var', 'pcy_var', 'pcz_var'):
            setattr(gui, name, tk.DoubleVar(master=interpreter, value=0.0))
        MultiStepOverlapGUI._populate_point_vars(gui)
        return gui, jobs, eulers, pc

    def test_populated_point_editor_preserves_full_precision_and_unchanged_apply_is_noop(self):
        gui, jobs, eulers, pc = self.point_editor_stub()
        np.testing.assert_array_equal(
            [gui.euler1_deg_var.get(), gui.euler2_deg_var.get(), gui.euler3_deg_var.get()], eulers,
        )
        np.testing.assert_array_equal([gui.pcx_var.get(), gui.pcy_var.get(), gui.pcz_var.get()], pc)
        MultiStepOverlapGUI._apply_selected_point_values(gui)
        self.assertFalse(jobs)
        gui.session.set_point_state.assert_not_called()
        self.assertEqual(gui.status_var.get(), 'No changes applied.')

    def test_point_editor_pc_only_edit_does_not_resubmit_or_round_eulers(self):
        gui, jobs, _eulers, pc = self.point_editor_stub()
        edited = pc[0] + 0.0000001234567
        gui.pcx_var.set(edited)
        MultiStepOverlapGUI._apply_selected_point_values(gui)
        gui.pcx_var.set(0.9)
        jobs[0][0]()
        self.assertEqual(gui.session.set_point_state.call_args.args, (2,))
        self.assertEqual(gui.session.set_point_state.call_args.kwargs,
                         {'pc_custom': (edited, pc[1], pc[2])})
        self.assertFalse(jobs[0][1]['sync_conditioning'])
        gui._mark_calibration_unapplied.assert_called_once_with()

    def test_point_editor_euler_only_edit_preserves_refined_pc_and_preview_precision(self):
        gui, jobs, eulers, pc = self.point_editor_stub()
        edited = eulers[1] + 0.000001234567
        gui.euler2_deg_var.set(edited)
        MultiStepOverlapGUI._apply_selected_point_values(gui)
        jobs[0][0]()
        gui._mark_calibration_unapplied.assert_not_called()
        self.assertEqual(gui.session.set_point_state.call_args.kwargs,
                         {'euler_deg': (eulers[0], edited, eulers[2])})
        preview_eulers, preview_pc = MultiStepOverlapGUI._edited_point_overrides(gui)
        np.testing.assert_array_equal(preview_pc, pc)
        np.testing.assert_array_equal(preview_eulers, np.deg2rad([eulers[0], edited, eulers[2]]))

    def test_h5_default_phase_excludes_unindexed_zero_and_uses_metadata_fallback(self):
        for phases, metadata, expected in (
            ([0, 0, 0, 1, 2, 2], {}, 2),
            ([0, 0], {3: object(), 2: object()}, 2),
            ([0, 0], {}, 1),
            ([], {}, 1),
        ):
            data = SimpleNamespace(source_type='h5oina', phases=np.asarray(phases), phase_symmetries=metadata)
            self.assertEqual(MultiStepOverlapGUI._default_input_phase_id(data), expected)

    def test_up_default_phase_preserves_zero_based_conventions(self):
        data = SimpleNamespace(source_type='up_ang', phases=np.asarray([0, 0, 1]), phase_symmetries={1: object()})
        self.assertEqual(MultiStepOverlapGUI._default_input_phase_id(data), 0)

    def indexing_stub(self, *, auto=True):
        actions = []
        session = SimpleNamespace(
            dictionary_cache=SimpleNamespace(resolution_deg=1.2),
            dictionary_index_indices=Mock(return_value="indexed"),
            refine_orientations_indices=Mock(return_value="refined"),
        )
        gui = bind(SimpleNamespace(
            session=session, busy=False,
            phase_id_var=Value(1), di_res_deg_var=Value(2.0), dictionary_keep_n_var=Value(5),
            auto_refine_var=Value(auto), follow_dictionary_trust_var=Value(True),
            trust_euler_var=Value(9.0), maxfev_var=Value(50), refine_full_resolution_var=Value(True),
            _set_reindex_progress=Mock(), _set_refinement_progress=Mock(),
            _check_job_cancelled=Mock(), _post_ui=lambda cb: cb(),
            _sync_pattern_conditioning_settings=Mock(),
            _run_threaded=lambda action: actions.append(action),
        ), '_index_refinement_settings', '_index_primary_indices')
        return gui, actions

    def test_indexing_snapshots_settings_and_automatically_refines_at_dictionary_spacing(self):
        gui, actions = self.indexing_stub()
        gui._index_primary_indices(np.array([2, 3]), label="ROI")
        gui.maxfev_var.set(99)
        gui.auto_refine_var.set(False)
        gui.phase_id_var.set(7)
        gui.trust_euler_var.set(15)
        actions[0]()
        index = gui.session.dictionary_index_indices.call_args.kwargs
        refine = gui.session.refine_orientations_indices.call_args.kwargs
        self.assertEqual(index['phase_id'], 1)
        self.assertEqual(index['keep_n'], 5)
        self.assertEqual(refine['trust_euler_deg'], 1.2)
        self.assertEqual(refine['maxfev'], 50)
        self.assertTrue(refine['use_full_resolution'])

    def test_auto_refinement_can_be_disabled_and_manual_trust_is_retained(self):
        gui, actions = self.indexing_stub(auto=False)
        gui.follow_dictionary_trust_var.set(False)
        self.assertEqual(gui._index_refinement_settings(), (9.0, 50, True))
        gui._index_primary_indices(np.array([2]), label="Point")
        actions[0]()
        gui.session.refine_orientations_indices.assert_not_called()

    def test_busy_actions_cannot_open_restore_or_mutate_calibration(self):
        session = SimpleNamespace(add_calibration_point=Mock())
        gui = SimpleNamespace(busy=True, session=session, _pending_restore_path=None)
        with patch('multistep_overlap_ebsd.gui.filedialog.askopenfilename') as dialog:
            MultiStepOverlapGUI._restore_workflow(gui)
        dialog.assert_not_called()
        MultiStepOverlapGUI._add_calibration_point(gui)
        session.add_calibration_point.assert_not_called()
        self.assertIsNone(gui._pending_restore_path)

    def test_invalid_settings_do_not_start_job(self):
        gui, _actions = self.indexing_stub()
        gui.index_var = Value(0)
        gui.dictionary_keep_n_var.set("invalid")
        with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
            MultiStepOverlapGUI._index_selected_point(gui)
        error.assert_called_once()
        gui.session.dictionary_index_indices.assert_not_called()

    def test_missing_scale_is_ignored_if_correction_off_and_validated_if_on(self):
        actions = []
        gui = bind(SimpleNamespace(
            busy=False, use_scan_pc_shift_var=Value(False), detector_px_size_var=Value(''),
            session=SimpleNamespace(apply_average_calibration_pc=Mock(return_value='applied')),
            _run_threaded=lambda action, **kwargs: actions.append(action),
            _mark_calibration_unapplied=Mock(), _finish_calibration_application=Mock(),
        ), '_calibration_application_settings')
        MultiStepOverlapGUI._apply_average_calibration_pc(gui)
        actions.pop()()
        self.assertEqual(gui.session.apply_average_calibration_pc.call_args.kwargs,
                         {'use_scan_geometry': False, 'effective_detector_px_size_um': None})
        gui.use_scan_pc_shift_var.set(True)
        with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
            MultiStepOverlapGUI._apply_average_calibration_pc(gui)
        error.assert_called_once()
        self.assertFalse(actions)
        gui.detector_px_size_var.set('160')
        MultiStepOverlapGUI._apply_average_calibration_pc(gui)
        gui.detector_px_size_var.set('99')
        actions.pop()()
        self.assertEqual(gui.session.apply_average_calibration_pc.call_args.kwargs['effective_detector_px_size_um'], 160)

    def test_stale_gui_residual_cannot_be_indexed_after_session_invalidation(self):
        gui = SimpleNamespace(
            busy=False, index_var=Value(4), blur_sigma_var=Value(0.0), residual_keep_n_var=Value(5),
            last_overlap=SimpleNamespace(index=4),
            session=SimpleNamespace(get_residual_point_result=Mock(return_value=None)),
            _residual_ncc_threshold=lambda: 0.0, _selected_primary_ncc=lambda _index: 0.8,
            _sync_pattern_conditioning_settings=Mock(),
            _run_threaded=Mock(),
        )
        with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
            MultiStepOverlapGUI._index_overlap_residual(gui)
        error.assert_called_once()
        gui._run_threaded.assert_not_called()
        gui.session.get_residual_point_result.assert_called_once_with(4)

    def test_conditioning_invalidation_between_preflight_and_worker_cannot_reuse_residual(self):
        actions = []
        gui = SimpleNamespace(
            busy=False, index_var=Value(4), blur_sigma_var=Value(0.0), residual_keep_n_var=Value(5),
            session=SimpleNamespace(
                get_residual_point_result=Mock(side_effect=[SimpleNamespace(index=4), None]),
                index_overlap_residual=Mock(),
            ),
            _residual_ncc_threshold=lambda: 0.0, _selected_primary_ncc=lambda _index: 0.8,
            _sync_pattern_conditioning_settings=Mock(),
            _set_overlap_progress=Mock(), _run_threaded=lambda action: actions.append(action),
        )
        MultiStepOverlapGUI._index_overlap_residual(gui)
        with self.assertRaisesRegex(ValueError, "no longer valid"):
            actions[0]()
        gui.session.index_overlap_residual.assert_not_called()

    def test_worker_completion_is_applied_only_by_main_thread_queue(self):
        main = threading.get_ident()
        scheduled = []
        completed = []
        gui = bind(SimpleNamespace(
            busy=False, session=SimpleNamespace(last_action_note=''),
            status_var=Value(''), _set_info_lines=Mock(), _sync_pattern_conditioning_settings=Mock(),
            after=lambda delay, callback: scheduled.append((threading.get_ident(), callback)),
            _on_action_error=Mock(),
        ), '_run_threaded', '_post_ui', '_drain_ui_callbacks', '_check_job_cancelled')
        gui._set_busy = lambda flag: setattr(gui, 'busy', flag)
        def done(message):
            completed.append((threading.get_ident(), message)); gui.busy = False
        gui._on_action_done = done
        worker_ids = []
        def action():
            worker_ids.append(threading.get_ident()); return 'done'
        self.assertTrue(gui._run_threaded(action))
        gui._worker_thread.join(timeout=5)
        self.assertTrue(gui.busy)
        self.assertFalse(completed)
        self.assertTrue(all(owner == main for owner, _ in scheduled))
        gui._drain_ui_callbacks()
        self.assertEqual(completed, [(main, 'done')])
        self.assertNotEqual(worker_ids, [main])
        gui._on_action_error.assert_not_called()

    def test_cancel_request_is_cooperative_and_checked_at_boundary(self):
        gui = bind(SimpleNamespace(
            busy=True, _cancel_requested=threading.Event(), status_var=Value(''), btn_cancel=Mock(),
        ), '_check_job_cancelled')
        gui._check_job_cancelled()
        MultiStepOverlapGUI._cancel_current_action(gui)
        self.assertTrue(gui._cancel_requested.is_set())
        with self.assertRaises(InterruptedError):
            gui._check_job_cancelled()

    def test_residual_roi_pipeline_shares_settings_and_honors_auto_refine(self):
        for auto in (True, False):
            calls = []
            def operation(name):
                def run(*args, **kwargs):
                    calls.append((name, kwargs))
                    kwargs['progress_callback'](100.0, name)
                    return name
                return run
            gui, _unused = self.indexing_stub(auto=auto)
            gui.session.data = object()
            gui.session.roi_indices = lambda *_bounds: np.array([0, 1])
            gui.session.compute_overlap_residual_indices = operation('generate')
            gui.session.index_overlap_residual_indices = operation('index')
            gui.session.refine_overlap_residual_indices = operation('refine')
            gui.index_var = Value(0)
            gui.fit_blur_gain_var = Value(False)
            gui.blur_sigma_var = Value(1.5)
            gui.gain_fit_maxiter_var = Value(40)
            gui.gain_fit_popsize_var = Value(8)
            gui.parallel_cores_var = Value(1)
            gui.write_residual_patterns_var = Value(False)
            gui.residual_pattern_path_var = Value('')
            gui._primary_fit_bounds = lambda: [(0.0, 1.0)]
            gui._roi_bounds = lambda: (0, 0, 1, 2)
            gui._filter_roi_indices_by_threshold = lambda indices: (indices, 0)
            gui._set_overlap_progress = Mock()
            gui._run_threaded = lambda action: action()
            MultiStepOverlapGUI._run_residual_roi_analysis(gui)
            self.assertEqual([name for name, _kw in calls], ['generate', 'index', 'refine'] if auto else ['generate', 'index'])
            self.assertEqual(calls[1][1]['keep_n'], 5)
            self.assertEqual(calls[0][1]['parallel_cores'], 1)
            self.assertEqual(calls[0][1]['blur_sigma'], 1.5)
            if auto:
                self.assertEqual(calls[2][1]['trust_euler_deg'], 1.2)
                self.assertTrue(calls[2][1]['use_full_resolution'])

    def test_failed_restore_retains_existing_session(self):
        previous = SimpleNamespace(close=Mock())
        staged = SimpleNamespace(restore_workflow_state=Mock(side_effect=ValueError('bad file')),
                                 close=Mock())
        jobs = []
        gui = SimpleNamespace(busy=False, session=previous,
                              _run_threaded=lambda fn, **kwargs: jobs.append((fn, kwargs)))
        with patch('multistep_overlap_ebsd.gui.filedialog.askopenfilename', return_value='/tmp/state.npz'), \
             patch('multistep_overlap_ebsd.gui.WorkflowSession', return_value=staged):
            MultiStepOverlapGUI._restore_workflow(gui)
        with self.assertRaises(ValueError):
            jobs[0][0]()
        self.assertIs(gui.session, previous)
        previous.close.assert_not_called()
        staged.close.assert_called_once()
        self.assertFalse(jobs[0][1]['sync_conditioning'])

    def test_successful_restore_releases_replaced_session_after_commit(self):
        previous = SimpleNamespace(close=Mock())
        restored = SimpleNamespace(restore_workflow_state=Mock(return_value='restored'), close=Mock())
        jobs = []
        gui = SimpleNamespace(busy=False, session=previous,
                              _run_threaded=lambda fn, **kwargs: jobs.append((fn, kwargs)))
        with patch('multistep_overlap_ebsd.gui.filedialog.askopenfilename', return_value='/tmp/state.npz'), \
             patch('multistep_overlap_ebsd.gui.WorkflowSession', return_value=restored):
            MultiStepOverlapGUI._restore_workflow(gui)
        message = jobs[0][0]()
        self.assertIs(gui.session, previous)
        previous.close.assert_not_called()
        jobs[0][1]['on_success'](message)
        self.assertIs(gui.session, restored)
        previous.close.assert_called_once()
        restored.close.assert_not_called()
        self.assertEqual(gui._pending_restore_path, restored.restore_workflow_state.call_args.args[0])

    def fitting_stub(self, method='staged_joint'):
        jobs = []
        residual = SimpleNamespace(
            index=0, row=0, col=0, residual=np.zeros((2, 2)),
            ncc_unfitted=0.8, ncc_es=0.9, fitted_sigma=1.0, fit_message='fitted',
        )
        mixture = SimpleNamespace(
            index=0, primary_fraction=0.7, secondary_fraction=0.3,
            ncc_mixture=0.95, residual_rms=0.1,
        )
        session = SimpleNamespace(
            data=object(), dictionary_cache=object(),
            roi_indices=lambda *_bounds: np.array([0, 1]),
            get_primary_index_ncc=lambda _index: 0.9,
            get_residual_point_result=Mock(return_value=residual),
            get_overlap_mixture_result=Mock(return_value=mixture),
            analyze_overlap_point=Mock(return_value=residual),
            fit_overlap_mixture_point=Mock(return_value=mixture),
        )
        for name in (
            'compute_overlap_residual_indices', 'index_overlap_residual_indices',
            'compute_overlap_mixture_indices', 'dictionary_index_indices',
            'export_overlap_optimization_results',
        ):
            setattr(session, name, Mock(return_value='done'))
        gui = SimpleNamespace(
            session=session, busy=False, index_var=Value(0),
            phase_id_var=Value(1), di_res_deg_var=Value(1.0),
            dictionary_keep_n_var=Value(5), auto_refine_var=Value(False),
            fit_method_var=Value(FIT_METHOD_LABELS[method]),
            fit_blur_gain_var=Value(True), blur_sigma_var=Value(0.0),
            gain_fit_maxiter_var=Value(40), gain_fit_popsize_var=Value(8),
            parallel_cores_var=Value(10), step3_parallel_cores_var=Value(10),
            step4_parallel_cores_var=Value(10), write_residual_patterns_var=Value(False),
            residual_pattern_path_var=Value(''),
            overlap_mixture_trust_euler_var=Value(1.0), overlap_mixture_maxfev_var=Value(80),
            _primary_fit_bounds=lambda: [(0.1, 5.0)], _roi_bounds=lambda: (0, 0, 1, 2),
            _residual_ncc_threshold=lambda: 0.0, _selected_primary_ncc=lambda _index: 0.9,
            _overlap_mixture_residual_ncc_threshold=lambda: 0.0,
            _overlap_mixture_residual_ncc_for_index=lambda _index: 0.9,
            _filter_roi_indices_by_threshold=lambda indices: (indices, 0),
            _filter_overlap_mixture_indices_by_residual_threshold=lambda indices: (indices, 0),
            _sync_residual_keep_n_to_dictionary=Mock(),
            _browse_overlap_optimization_export=lambda: '/tmp/results.h5',
            _set_overlap_progress=Mock(), _set_overlap_optimization_progress=Mock(),
            _set_complete_analysis_progress=Mock(), _set_reindex_progress=Mock(),
            _set_refinement_progress=Mock(), _refresh_complete_analysis_maps=Mock(),
            _sync_pattern_conditioning_settings=Mock(),
            _check_job_cancelled=Mock(), _post_ui=lambda callback: callback(), _log=Mock(),
            _run_threaded=lambda action: jobs.append(action),
        )
        return gui, jobs

    def run_fitting_worker(self, jobs):
        self.assertEqual(len(jobs), 1)
        errors = []
        def execute():
            try:
                jobs[0]()
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=execute, daemon=True)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        if errors:
            raise errors[0]

    def test_all_residual_and_mixture_fit_actions_snapshot_selected_method(self):
        for method in FIT_METHOD_LABELS:
            for action, operation in (
                ('_analyze_overlap', 'analyze_overlap_point'),
                ('_compute_overlap_residual_roi', 'compute_overlap_residual_indices'),
                ('_run_residual_roi_analysis', 'compute_overlap_residual_indices'),
                ('_fit_overlap_mixture', 'fit_overlap_mixture_point'),
                ('_fit_overlap_mixture_roi', 'compute_overlap_mixture_indices'),
            ):
                with self.subTest(method=method, action=action):
                    gui, jobs = self.fitting_stub(method)
                    with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
                        getattr(MultiStepOverlapGUI, action)(gui)
                    error.assert_not_called()
                    gui.fit_method_var.set('changed after job was queued')
                    self.run_fitting_worker(jobs)
                    self.assertEqual(getattr(gui.session, operation).call_args.kwargs['fit_method'], method)

    def test_combined_roi_workflow_snapshots_method_for_both_fitting_stages(self):
        for include_step4 in (False, True):
            with self.subTest(include_step4=include_step4):
                gui, jobs = self.fitting_stub()
                with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
                    MultiStepOverlapGUI._run_complete_roi_analysis(gui, include_step4=include_step4)
                error.assert_not_called()
                gui.fit_method_var.set('changed after job was queued')
                self.run_fitting_worker(jobs)
                self.assertEqual(gui.session.compute_overlap_residual_indices.call_args.kwargs['fit_method'], 'staged_joint')
                if include_step4:
                    self.assertEqual(gui.session.compute_overlap_mixture_indices.call_args.kwargs['fit_method'], 'staged_joint')
                else:
                    gui.session.compute_overlap_mixture_indices.assert_not_called()

    def test_step4_export_snapshots_fit_method(self):
        gui, jobs = self.fitting_stub()
        with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
            MultiStepOverlapGUI._export_overlap_optimization_results(gui)
        error.assert_not_called()
        gui.fit_method_var.set('changed after job was queued')
        self.run_fitting_worker(jobs)
        settings = gui.session.export_overlap_optimization_results.call_args.kwargs['settings']
        self.assertEqual(settings['fit_method'], 'staged_joint')

    def test_restore_prefers_shared_primary_settings_and_ignores_legacy_pixel_placeholder(self):
        class SettingsStub(SimpleNamespace):
            def __getattr__(self, name):
                if name.endswith('_var'):
                    value = Value(0)
                    setattr(self, name, value)
                    return value
                raise AttributeError(name)
        gui = SettingsStub(
            session=SimpleNamespace(data=None), workflow_notebook=None, primary_fit_bound_specs=[],
            dictionary_keep_n_var=Value(5), maxfev_var=Value(50), parallel_cores_var=Value(1),
            detector_px_size_var=Value('160'),
        )
        MultiStepOverlapGUI._apply_workflow_ui_state(gui, {
            'dictionary_keep_n': 5, 'residual_keep_n': 9,
            'maxfev': 40, 'residual_maxfev': 90,
            'step3_parallel_cores': 0, 'step4_parallel_cores': 8,
            'detector_px_size': 1.0, 'detector_binning': 8,
        })
        self.assertEqual(gui.dictionary_keep_n_var.get(), 5)
        self.assertEqual(gui.maxfev_var.get(), 40)
        self.assertEqual(gui.parallel_cores_var.get(), 1)
        self.assertEqual(gui.detector_px_size_var.get(), '160')
        MultiStepOverlapGUI._apply_workflow_ui_state(gui, {'effective_detector_px_size_um': '320'})
        self.assertEqual(gui.detector_px_size_var.get(), '320')
        interpreter = tk.Tcl()
        gui.trust_euler_var = tk.DoubleVar(master=interpreter, value=1.0)
        gui.di_res_deg_var = tk.DoubleVar(master=interpreter, value=1.0)
        gui.follow_dictionary_trust_var = tk.BooleanVar(master=interpreter, value=True)
        def follow_spacing(*_args):
            if gui.follow_dictionary_trust_var.get():
                gui.trust_euler_var.set(gui.di_res_deg_var.get())
        gui.di_res_deg_var.trace_add('write', follow_spacing)
        gui.follow_dictionary_trust_var.trace_add('write', follow_spacing)
        MultiStepOverlapGUI._apply_workflow_ui_state(gui, {
            'trust_euler': 4.0, 'dictionary_resolution': 1.2, 'follow_dictionary_trust': False,
        })
        self.assertEqual(gui.trust_euler_var.get(), 4.0)


if __name__ == '__main__':
    unittest.main()
