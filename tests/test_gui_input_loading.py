from __future__ import annotations

import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch

from multistep_overlap_ebsd.gui import MultiStepOverlapGUI


def value(initial):
    variable = Mock()
    variable.get.return_value = initial
    return variable


class GuiInputLoadingTests(unittest.TestCase):
    def input_stub(self):
        jobs = []
        previous = SimpleNamespace(master=object(), _clear_dictionary_cache=Mock())
        staged = SimpleNamespace(
            master=None, load_input=Mock(return_value="Loaded input."),
            set_pattern_mask_option=Mock(), set_dynamic_background=Mock(),
            _clear_dictionary_cache=Mock(),
        )
        gui = SimpleNamespace(
            busy=False, session=previous,
            sample_tilt_var=value(70.0), detector_tilt_var=value(5.0),
            pattern_path_var=value('/tmp/new.up1'), orientation_path_var=value('/tmp/new.ang'),
            pattern_mask_option_var=value(-1), dynamic_bg_enabled_var=value(True),
            dynamic_bg_std_var=value('8'),
            _run_threaded=lambda action, **kwargs: jobs.append((action, kwargs)),
            _finish_input_load=Mock(return_value='Ready.'),
        )
        return gui, previous, staged, jobs

    def test_new_input_uses_fresh_session_and_commits_only_after_success(self):
        gui, previous, staged, jobs = self.input_stub()
        with patch('multistep_overlap_ebsd.gui.WorkflowSession', return_value=staged):
            MultiStepOverlapGUI._load_input(gui)
        action, options = jobs[0]
        self.assertFalse(options['sync_conditioning'])
        self.assertIs(gui.session, previous)
        gui.sample_tilt_var.get.return_value = 20.0
        gui.pattern_path_var.get.return_value = '/tmp/changed.up1'
        message = action()
        self.assertIs(gui.session, previous)
        self.assertEqual(staged.load_input.call_args.kwargs['pattern_path'], '/tmp/new.up1')
        self.assertEqual(staged.load_input.call_args.kwargs['geom'].sample_tilt_deg, 70.0)
        staged.set_pattern_mask_option.assert_called_once_with(-1)
        staged.set_dynamic_background.assert_called_once_with(True, std_px=8.0)
        self.assertEqual(options['on_success'](message), 'Ready.')
        self.assertIs(gui.session, staged)
        self.assertIsNone(gui.session.master)
        previous._clear_dictionary_cache.assert_called_once_with()
        gui._finish_input_load.assert_called_once_with('Loaded input.')

    def test_failed_input_preserves_previous_session_and_master(self):
        gui, previous, staged, jobs = self.input_stub()
        staged.load_input.side_effect = ValueError('Unreadable input')
        with patch('multistep_overlap_ebsd.gui.WorkflowSession', return_value=staged):
            MultiStepOverlapGUI._load_input(gui)
        with self.assertRaisesRegex(ValueError, 'Unreadable input'):
            jobs[0][0]()
        self.assertIs(gui.session, previous)
        self.assertIsNotNone(gui.session.master)
        previous._clear_dictionary_cache.assert_not_called()
        staged._clear_dictionary_cache.assert_called_once_with()
        gui._finish_input_load.assert_not_called()

    def test_invalid_geometry_does_not_replace_input_or_launch_job(self):
        gui, previous, _staged, jobs = self.input_stub()
        gui.sample_tilt_var.get.return_value = float('nan')
        with patch('multistep_overlap_ebsd.gui.messagebox.showerror') as error:
            MultiStepOverlapGUI._load_input(gui)
        error.assert_called_once()
        self.assertFalse(jobs)
        self.assertIs(gui.session, previous)

    def test_browsing_input_exposes_matching_controls(self):
        for path, expected in (('/tmp/a.up1', 'UP + ANG'), ('/tmp/a.h5oina', 'H5OINA')):
            gui = SimpleNamespace(
                pattern_path_var=value(''), source_type_var=value(''),
                _sync_input_type_controls=Mock(), _refresh_default_workflow_path=Mock(),
            )
            with patch('multistep_overlap_ebsd.gui.filedialog.askopenfilename', return_value=path):
                MultiStepOverlapGUI._browse_patterns(gui)
            gui.source_type_var.set.assert_called_once_with(expected)
            gui._sync_input_type_controls.assert_called_once_with(reset_up_tilt=True)

    def test_loaded_ncc_is_presented_as_completed_primary_indexing(self):
        session = SimpleNamespace(
            data=SimpleNamespace(rows=2, cols=3, count=6, phases=np.ones(6), source_type='h5oina'),
            indexed_mask=np.array([True, False, True, False, True, False]),
            available_layers=lambda: ['NCC', 'Phase'],
        )
        gui = SimpleNamespace(
            session=session, _plot_views={}, _sync_index_quality_layer_choices=Mock(),
            _index_quality_layer_choices=lambda: ['NCC'], _default_index_quality_layer=lambda: 'NCC',
            _default_input_phase_id=lambda data: 1, _default_residual_pattern_path=lambda: '/tmp/next.up1',
            _default_roi_export_path=lambda residual: '/tmp/next.h5oina',
            _default_overlap_optimization_export_path=lambda: '/tmp/mixture.h5',
            _refresh_default_workflow_path=Mock(), _set_reindex_progress=Mock(),
            _update_mode_controls=Mock(), _update_calibration_summary=Mock(), _populate_point_vars=Mock(),
            _sync_loaded_geometry_controls=Mock(), _refresh_context_summary=Mock(),
        )
        for name in ('index_var', 'row_var', 'col_var', 'map_layer_var', 'index_quality_layer_var',
                     'roi_r0_var', 'roi_c0_var', 'roi_nrows_var', 'roi_ncols_var', 'phase_id_var',
                     'roi_export_format_var', 'residual_pattern_path_var', 'primary_roi_export_path_var',
                     'residual_roi_export_path_var', 'overlap_optimization_export_path_var'):
            setattr(gui, name, value('NCC'))
        message = MultiStepOverlapGUI._finish_input_load(gui, 'Loaded saved NCC.')
        gui._set_reindex_progress.assert_called_once_with(
            100., 'Loaded 3/6 indexed primary point(s); DI can be skipped.')
        self.assertIn('Load the master pattern', message)
        self.assertIsNone(gui.last_overlap)
        self.assertIsNone(gui.last_overlap_mixture)


if __name__ == '__main__':
    unittest.main()
