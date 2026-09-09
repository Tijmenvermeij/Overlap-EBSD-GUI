from __future__ import annotations

import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from multistep_overlap_ebsd.core import WorkflowSession
from multistep_overlap_ebsd.gui_controls import GUIControls


class GuiControlsTests(unittest.TestCase):
    def test_displayed_automatic_range_matches_the_loaded_dictionary(self):
        interpreter = tk.Tcl()
        gui = SimpleNamespace(
            session=SimpleNamespace(dictionary_cache=SimpleNamespace(resolution_deg=1.2)),
            follow_dictionary_trust_var=tk.BooleanVar(interpreter, value=True),
            di_res_deg_var=tk.DoubleVar(interpreter, value=6.0),
            trust_euler_var=tk.DoubleVar(interpreter, value=6.0),
        )
        GUIControls._sync_refinement_settings(gui)
        self.assertEqual(gui.trust_euler_var.get(), 1.2)
        gui.session.dictionary_cache = None
        GUIControls._sync_refinement_settings(gui)
        self.assertEqual(gui.trust_euler_var.get(), 6.0)

    def test_mask_choices_match_the_numeric_processing_configuration(self):
        interpreter = tk.Tcl()
        session = WorkflowSession()
        gui = SimpleNamespace(
            session=session, pattern_mask_mode_var=tk.StringVar(interpreter),
            pattern_mask_option_var=tk.IntVar(interpreter), _on_value_commit=Mock(),
        )
        for label, kind, diameter in (
            ('None', 'none', -1), ('Automatic', 'circle', 0), ('Custom diameter', 'circle', 128),
        ):
            gui.pattern_mask_mode_var.set(label)
            GUIControls._on_mask_mode_changed(gui)
            session.set_pattern_mask_option(gui.pattern_mask_option_var.get())
            self.assertEqual(session.pattern_mask_config.kind, kind)
            self.assertEqual(session.pattern_mask_config.diameter_px, diameter)
            GUIControls._sync_mask_mode(gui)
            self.assertEqual(gui.pattern_mask_mode_var.get(), label)

    def test_custom_mask_starts_at_full_fitting_diameter(self):
        interpreter = tk.Tcl()
        gui = SimpleNamespace(
            session=SimpleNamespace(data=SimpleNamespace(h=100, w=120)),
            pattern_mask_mode_var=tk.StringVar(interpreter, value='Custom diameter'),
            pattern_mask_option_var=tk.IntVar(interpreter), _on_value_commit=Mock(),
            _mask_diameter_entry=Mock(),
        )
        GUIControls._on_mask_mode_changed(gui)
        self.assertEqual(gui.pattern_mask_option_var.get(), 100)
        GUIControls._sync_mask_mode(gui)
        gui._mask_diameter_entry.configure.assert_called_with(state='normal')
        gui.pattern_mask_option_var.set(0)
        GUIControls._sync_mask_mode(gui)
        gui._mask_diameter_entry.configure.assert_called_with(state='disabled')


if __name__ == '__main__':
    unittest.main()
