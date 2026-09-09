from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from multistep_overlap_ebsd.gui import MultiStepOverlapGUI
from test_gui_actions import Value, bind


class CalibrationApplicationTests(unittest.TestCase):
    def gui(self):
        jobs = []
        session = SimpleNamespace(
            data=SimpleNamespace(effective_detector_px_size_um=40.0, pc_output_convention='oxford'),
            calibration_indices=[0, 1], current_pc_custom=np.array([[0.4, 0.5, 0.6], [0.6, 0.5, 0.6]]),
            calibration_point_summary=Mock(return_value='Detailed optimized PCs'),
            apply_average_calibration_pc=Mock(return_value='Applied.'),
        )
        gui = bind(SimpleNamespace(
            session=session, busy=False, _calibration_apply_session=session,
            _calibration_apply_state='none', _applied_calibration_settings=None,
            _completed_calibration_report=None,
            use_scan_pc_shift_var=Value(False), detector_px_size_var=Value('40'),
            calibration_apply_status_var=Value(''), btn_apply_calibration=Mock(),
            _calibration_apply_status_label=Mock(), _calibration_warning_button=Mock(),
            _calibration_warning_banner=Mock(), workflow_notebook=Mock(),
            _activate_plot_view=Mock(), _refresh_plot=Mock(),
            _run_threaded=lambda action, **kwargs: jobs.append((action, kwargs)),
        ), '_calibration_application_settings', '_mark_calibration_unapplied',
            '_update_calibration_application_controls', '_finish_calibration_application',
            '_finish_calibration_optimization')
        gui.workflow_notebook.index.return_value = 0
        return gui, jobs

    def test_optimization_requires_apply_and_switching_tabs_shows_warning(self):
        gui, _jobs = self.gui()
        message = gui._finish_calibration_optimization('Optimized.', np.array([0, 1]))
        self.assertIn('Next:', message)
        self.assertEqual(gui._calibration_apply_state, 'pending')
        self.assertIn('Apply required', gui.calibration_apply_status_var.get())
        self.assertEqual(gui.btn_apply_calibration.configure.call_args.kwargs['style'], 'PendingCalibration.TButton')
        gui._calibration_warning_banner.pack.assert_not_called()
        gui.workflow_notebook.index.return_value = 2
        MultiStepOverlapGUI._on_workspace_changed(gui)
        gui._calibration_warning_banner.pack.assert_called_once()
        self.assertEqual(gui._calibration_apply_state, 'pending')

    def test_only_successful_apply_turns_green_and_preserves_optimization_statistics(self):
        gui, jobs = self.gui()
        gui._finish_calibration_optimization('Optimized.', np.array([0, 1]))
        report = gui._completed_calibration_report
        MultiStepOverlapGUI._apply_average_calibration_pc(gui)
        self.assertEqual(gui._calibration_apply_state, 'pending')
        action, options = jobs[0]
        message = action()
        self.assertEqual(gui._calibration_apply_state, 'pending')
        gui.session.current_pc_custom[:] = [0.5, 0.5, 0.6]
        options['on_success'](message)
        self.assertEqual(gui._calibration_apply_state, 'applied')
        self.assertEqual(gui.btn_apply_calibration.configure.call_args.kwargs['style'], 'AppliedCalibration.TButton')
        self.assertIs(gui._completed_calibration_report, report)
        self.assertIn('0.141421', report[2])
        gui._calibration_warning_banner.pack_forget.assert_called()

    def test_failed_reapply_cannot_leave_a_green_indicator(self):
        gui, jobs = self.gui()
        gui._finish_calibration_application('Applied.', (False, None))
        gui.session.apply_average_calibration_pc.side_effect = RuntimeError('Application failed')
        MultiStepOverlapGUI._apply_average_calibration_pc(gui)
        with self.assertRaisesRegex(RuntimeError, 'Application failed'):
            jobs[0][0]()
        self.assertEqual(gui._calibration_apply_state, 'pending')
        self.assertIn('Apply required', gui.calibration_apply_status_var.get())

    def test_changed_active_geometry_requires_reapply_but_unused_pitch_does_not(self):
        gui, _jobs = self.gui()
        gui._finish_calibration_application('Applied.', (False, None))
        gui.detector_px_size_var.set('unknown')
        gui._update_calibration_application_controls()
        self.assertEqual(gui._calibration_apply_state, 'applied')
        gui.use_scan_pc_shift_var.set(True)
        gui._update_calibration_application_controls()
        self.assertEqual(gui._calibration_apply_state, 'pending')
        gui.detector_px_size_var.set('40.0')
        gui._finish_calibration_application('Applied.', (True, 40.0))
        gui.detector_px_size_var.set('40.000')
        gui._update_calibration_application_controls()
        self.assertEqual(gui._calibration_apply_state, 'applied')
        gui.detector_px_size_var.set('80')
        gui._update_calibration_application_controls()
        self.assertEqual(gui._calibration_apply_state, 'pending')

    def test_new_session_resets_application_status(self):
        gui, _jobs = self.gui()
        gui._mark_calibration_unapplied()
        gui.session = SimpleNamespace(data=None, calibration_indices=[])
        gui._update_calibration_application_controls()
        self.assertEqual(gui._calibration_apply_state, 'none')
        self.assertEqual(gui.btn_apply_calibration.configure.call_args.kwargs['state'], 'disabled')

    def test_busy_tab_change_does_not_reenable_apply_or_banner_button(self):
        gui, _jobs = self.gui()
        gui._mark_calibration_unapplied()
        gui.busy = True
        gui.workflow_notebook.index.return_value = 1
        MultiStepOverlapGUI._on_workspace_changed(gui)
        self.assertEqual(gui.btn_apply_calibration.configure.call_args.kwargs['state'], 'disabled')
        self.assertEqual(gui._calibration_warning_button.configure.call_args.kwargs['state'], 'disabled')
        gui._refresh_plot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
