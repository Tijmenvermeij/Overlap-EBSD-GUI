import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from multistep_overlap_ebsd.workflow_autosave import save_checkpoint
from multistep_overlap_ebsd.gui import MultiStepOverlapGUI


class WorkflowAutosaveTests(unittest.TestCase):
    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workflow.npz'
            path.write_bytes(b'previous checkpoint')
            def fail(temporary, **kwargs):
                Path(temporary).write_bytes(b'incomplete')
                raise OSError('disk full')
            with self.assertRaisesRegex(OSError, 'disk full'):
                save_checkpoint(SimpleNamespace(save_workflow_state=fail), path, {})
            self.assertEqual(path.read_bytes(), b'previous checkpoint')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_completed_checkpoint_replaces_file_and_preserves_ui_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workflow.npz'
            state = {'selected_workflow_tab': 2}
            def save(temporary, *, ui_state):
                self.assertEqual(ui_state, state)
                self.assertEqual(Path(temporary).parent, path.parent)
                Path(temporary).write_bytes(b'completed state')
            save_checkpoint(SimpleNamespace(save_workflow_state=save), path, state)
            self.assertEqual(path.read_bytes(), b'completed state')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_worker_saves_after_processing_and_reports_failure_without_losing_result(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                main_thread = threading.get_ident()
                path = Path(directory) / 'workflow.npz'
                events, callbacks = [], []
                def save(temporary, *, ui_state):
                    self.assertNotEqual(threading.get_ident(), main_thread)
                    self.assertEqual(events, ['processed'])
                    self.assertEqual(ui_state, {'selected_workflow_tab': 2})
                    if fail:
                        raise OSError('disk full')
                    Path(temporary).write_bytes(b'checkpoint')
                    events.append('saved')
                gui = SimpleNamespace(
                    busy=False, session=SimpleNamespace(last_action_note='', save_workflow_state=save),
                    workflow_path_var=SimpleNamespace(get=lambda: str(path), set=Mock()),
                    _workflow_ui_state=lambda: {'selected_workflow_tab': 2},
                    _sync_pattern_conditioning_settings=Mock(), _set_busy=Mock(),
                    status_var=SimpleNamespace(set=Mock()), _set_info_lines=Mock(),
                    _check_job_cancelled=Mock(), _post_ui=callbacks.append,
                    _log=Mock(), _on_action_done=Mock(), _on_action_error=Mock(),
                    _drain_ui_callbacks=Mock(), after=Mock(),
                )
                def process():
                    events.append('processed')
                    return 'Indexing complete.'
                MultiStepOverlapGUI._run_threaded(gui, process, autosave=True)
                gui._worker_thread.join(timeout=5)
                self.assertFalse(gui._worker_thread.is_alive())
                for callback in callbacks:
                    callback()
                gui._on_action_error.assert_not_called()
                message = gui._on_action_done.call_args.args[0]
                self.assertIn('Indexing complete.', message)
                self.assertIn('FAILED' if fail else 'autosaved', message)
                self.assertEqual(path.exists(), not fail)
