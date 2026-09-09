from __future__ import annotations

import json
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from multistep_overlap_ebsd.gui import MultiStepOverlapGUI
from multistep_overlap_ebsd.cpu_fitting import FIT_METHOD_DEFAULT, FIT_METHOD_LABELS


class _Value:
    def __init__(self, value=0):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Settings(SimpleNamespace):
    def __getattr__(self, name):
        if name.endswith("_var"):
            variable = _Value()
            setattr(self, name, variable)
            return variable
        raise AttributeError(name)


class _Notebook:
    def __init__(self, selected=0):
        self.selected = selected

    def select(self, value=None):
        if value is not None:
            self.selected = value
        return self.selected

    def index(self, value):
        return value


class CalibrationPersistenceTests(unittest.TestCase):
    def settings(self, *, state="none", settings=None):
        session = SimpleNamespace(data=None)
        return _Settings(
            session=session, workflow_notebook=None, primary_fit_bound_specs=[],
            _calibration_apply_state=state, _calibration_apply_session=session,
            _applied_calibration_settings=settings,
            use_scan_pc_shift_var=_Value(False), detector_px_size_var=_Value(""),
            fit_method_var=_Value(FIT_METHOD_LABELS[FIT_METHOD_DEFAULT]),
            _update_calibration_application_controls=Mock(),
        )

    def application_settings(self, *, state="none", settings=None, actual_pitch=160.0):
        gui = self.settings(state=state, settings=settings)
        gui.session.data = SimpleNamespace(rows=2, cols=2, count=4,
                                           effective_detector_px_size_um=actual_pitch)
        gui.session.calibration_indices = [0, 1]
        gui.session.row_col_from_index = lambda index: divmod(index, 2)
        gui.busy = False
        gui.workflow_notebook = _Notebook()
        gui.btn_apply_calibration = Mock()
        gui._calibration_apply_status_label = Mock()
        gui._calibration_warning_button = Mock()
        gui._calibration_warning_banner = Mock()
        for name in ("_calibration_application_settings", "_update_calibration_application_controls",
                     "_roi_bounds"):
            setattr(gui, name, MethodType(getattr(MultiStepOverlapGUI, name), gui))
        return gui

    def saved_applied_state(self):
        gui = self.application_settings(state="applied", settings=(True, 160.0))
        gui.use_scan_pc_shift_var.set(True)
        gui.detector_px_size_var.set("160.0")
        return json.loads(json.dumps(MultiStepOverlapGUI._workflow_ui_state(gui)))

    def test_round_trip_preserves_pending_and_applied_with_json_settings(self):
        for application_state, settings in (
            ("none", None), ("pending", None), ("pending", (False, None)),
            ("applied", (False, None)), ("applied", (True, 160.125)),
        ):
            with self.subTest(application_state=application_state, settings=settings):
                original = self.settings(state=application_state, settings=settings)
                original.use_scan_pc_shift_var.set(bool(settings and settings[0]))
                original.detector_px_size_var.set(str(settings[1]) if settings and settings[0] else "")
                saved = json.loads(json.dumps(MultiStepOverlapGUI._workflow_ui_state(original)))
                restored = self.settings()
                MultiStepOverlapGUI._apply_workflow_ui_state(restored, saved)
                self.assertEqual(restored._calibration_apply_state, application_state)
                self.assertEqual(restored._applied_calibration_settings, settings)
                self.assertIs(restored._calibration_apply_session, restored.session)
                self.assertEqual(restored.use_scan_pc_shift_var.get(), original.use_scan_pc_shift_var.get())
                self.assertEqual(restored.detector_px_size_var.get(), original.detector_px_size_var.get())
                original._update_calibration_application_controls.assert_called_once_with()
                restored._update_calibration_application_controls.assert_called_once_with()

    def test_fit_method_round_trip_stores_stable_code_and_restores_display_label(self):
        for method, label in FIT_METHOD_LABELS.items():
            with self.subTest(method=method):
                original = self.settings()
                original.fit_method_var.set(label)
                saved = json.loads(json.dumps(MultiStepOverlapGUI._workflow_ui_state(original)))
                self.assertEqual(saved['fit_method'], method)
                restored = self.settings()
                MultiStepOverlapGUI._apply_workflow_ui_state(restored, saved)
                self.assertEqual(restored.fit_method_var.get(), label)

    def test_old_or_invalid_workflow_uses_fastest_fit_method(self):
        self.assertEqual(FIT_METHOD_DEFAULT, 'staged')
        for state in ({}, {'fit_method': 'unknown'}, {'fit_method': []}, {'fit_method': None}):
            with self.subTest(state=state):
                gui = self.settings()
                gui.fit_method_var.set(FIT_METHOD_LABELS['joint'])
                MultiStepOverlapGUI._apply_workflow_ui_state(gui, state)
                self.assertEqual(gui.fit_method_var.get(), FIT_METHOD_LABELS[FIT_METHOD_DEFAULT])

    def test_old_workflow_clears_previous_session_application_warning(self):
        gui = self.settings(state="applied", settings=(True, 160.0))
        gui._calibration_apply_session = object()
        MultiStepOverlapGUI._apply_workflow_ui_state(gui, {})
        self.assertEqual(gui._calibration_apply_state, "none")
        self.assertIsNone(gui._applied_calibration_settings)
        self.assertIs(gui._calibration_apply_session, gui.session)

    def test_malformed_saved_application_state_is_unknown(self):
        for state in (
            {"calibration_apply_state": "finished"},
            {"calibration_apply_state": []},
            {"calibration_apply_state": "applied"},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [True, 0]},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [True, float("nan")]},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [True, float("inf")]},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [True, "160"]},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [True, True]},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [1, 160]},
            {"calibration_apply_state": "applied", "applied_calibration_settings": [False, 160]},
            {"calibration_apply_state": "pending", "applied_calibration_settings": "bad"},
        ):
            with self.subTest(state=state):
                gui = self.settings(state="applied", settings=(False, None))
                MultiStepOverlapGUI._apply_workflow_ui_state(gui, state)
                self.assertEqual(gui._calibration_apply_state, "none")
                self.assertIsNone(gui._applied_calibration_settings)

    def test_save_does_not_carry_application_state_from_another_session(self):
        gui = self.settings(state="applied", settings=(True, 160.0))
        gui._calibration_apply_session = object()
        saved = MultiStepOverlapGUI._workflow_ui_state(gui)
        self.assertEqual(saved["calibration_apply_state"], "none")
        self.assertIsNone(saved["applied_calibration_settings"])

    def test_restore_refresh_observes_saved_settings_and_current_session(self):
        gui = self.settings(state="applied", settings=(False, None))
        gui._calibration_apply_session = object()
        observed = []
        gui._update_calibration_application_controls.side_effect = lambda: observed.append((
            gui._calibration_apply_state, gui._applied_calibration_settings,
            gui._calibration_apply_session, gui.use_scan_pc_shift_var.get(),
            gui.detector_px_size_var.get(),
        ))
        MultiStepOverlapGUI._apply_workflow_ui_state(gui, {
            "calibration_apply_state": "applied", "applied_calibration_settings": [True, 160.0],
            "use_scan_pc_shift": True, "effective_detector_px_size_um": "160.0",
        })
        self.assertEqual(observed, [("applied", (True, 160.0), gui.session, True, "160.0")])

    def test_save_refreshes_application_state_before_serializing(self):
        gui = self.settings(state="applied", settings=(True, 160.0))
        gui._update_calibration_application_controls.side_effect = lambda: setattr(
            gui, "_calibration_apply_state", "pending",
        )
        saved = MultiStepOverlapGUI._workflow_ui_state(gui)
        self.assertEqual(saved["calibration_apply_state"], "pending")
        self.assertEqual(saved["applied_calibration_settings"], [True, 160.0])

    def test_restored_applied_requires_matching_ui_and_loaded_detector_scale(self):
        for actual_pitch, correction, ui_pitch, expected in (
            (160.0, True, "160.0", "applied"),
            (160.0, True, "320.0", "pending"),
            (160.0, False, "160.0", "pending"),
            (80.0, True, "160.0", "pending"),
            (None, True, "160.0", "pending"),
            (160.0, True, "invalid", "pending"),
        ):
            with self.subTest(actual_pitch=actual_pitch, correction=correction, ui_pitch=ui_pitch):
                saved = self.saved_applied_state()
                saved["use_scan_pc_shift"] = correction
                saved["effective_detector_px_size_um"] = ui_pitch
                gui = self.application_settings(actual_pitch=actual_pitch)
                MultiStepOverlapGUI._apply_workflow_ui_state(gui, saved)
                self.assertEqual(gui._calibration_apply_state, expected)
                style = gui.btn_apply_calibration.configure.call_args.kwargs["style"]
                self.assertEqual(style, "AppliedCalibration.TButton" if expected == "applied"
                                 else "PendingCalibration.TButton")
                self.assertIs(gui._calibration_apply_session, gui.session)

    def test_saved_pending_restores_warning_in_another_workflow_tab(self):
        original = self.application_settings(state="pending")
        original.workflow_notebook.select(2)
        saved = json.loads(json.dumps(MultiStepOverlapGUI._workflow_ui_state(original)))
        restored = self.application_settings()
        MultiStepOverlapGUI._apply_workflow_ui_state(restored, saved)
        self.assertEqual(restored._calibration_apply_state, "pending")
        self.assertEqual(restored.workflow_notebook.select(), 2)
        self.assertIn("not yet applied", restored.calibration_apply_status_var.get())
        restored._calibration_warning_banner.pack.assert_called_once_with(
            fill="x", before=restored.workflow_notebook,
        )
        restored._calibration_warning_banner.pack_forget.assert_not_called()

    def test_old_workflow_with_calibration_points_does_not_invent_pending_warning(self):
        gui = self.application_settings(state="pending")
        MultiStepOverlapGUI._apply_workflow_ui_state(gui, {"selected_workflow_tab": 2})
        self.assertEqual(gui._calibration_apply_state, "none")
        gui._calibration_warning_banner.pack.assert_not_called()
        gui._calibration_warning_banner.pack_forget.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
