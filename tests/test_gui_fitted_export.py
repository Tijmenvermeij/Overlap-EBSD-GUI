import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from multistep_overlap_ebsd.gui import MultiStepOverlapGUI


class GuiFittedExportTests(unittest.TestCase):
    def test_buttons_export_correct_component_and_cancel_leaves_session_untouched(self):
        for secondary in (False, True):
            for selected_path in ('/tmp/result.h5oina', ''):
                gui = SimpleNamespace(session=SimpleNamespace(data=SimpleNamespace(pattern_path='/tmp/scan.h5oina'),overlap_mixture_results={2:object()}),
                    _source_stem=MultiStepOverlapGUI._source_stem, _run_threaded=Mock(),
                    _check_job_cancelled=Mock(), _post_ui=Mock(), _set_overlap_optimization_progress=Mock())
                with patch('multistep_overlap_ebsd.gui.filedialog.asksaveasfilename',return_value=selected_path) as dialog, patch(
                        'multistep_overlap_ebsd.fitted_export.export_fitted_patterns') as export:
                    MultiStepOverlapGUI._export_fitted_component_patterns(gui,secondary=secondary)
                    self.assertIn('H5OINA',dialog.call_args.kwargs['title'])
                    self.assertIn('residual' if secondary else 'primary', dialog.call_args.kwargs['initialfile'])
                    if selected_path:
                        gui._run_threaded.call_args.args[0]()
                        self.assertEqual(export.call_args.args,(gui.session, selected_path))
                        self.assertEqual(export.call_args.kwargs['secondary'], secondary)
                        self.assertTrue(export.call_args.kwargs['accepted_only'])
                        self.assertTrue(export.call_args.kwargs['primary_subtract_residual'])
                    else:
                        gui._run_threaded.assert_not_called()
