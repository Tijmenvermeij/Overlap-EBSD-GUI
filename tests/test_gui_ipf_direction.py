from __future__ import annotations

import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from multistep_overlap_ebsd.gui import MultiStepOverlapGUI


class IpfDirectionSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = tk.Tcl()
        self.variable = tk.StringVar(master=self.interpreter, value="Z")
        self.gui_stub = SimpleNamespace(ipf_direction_var=self.variable)

    def test_x_y_z_are_normalized_for_core_and_labels(self) -> None:
        for value, expected in (
            ("X", ("x", "IPF-X")),
            ("y", ("y", "IPF-Y")),
            (" Z ", ("z", "IPF-Z")),
        ):
            self.variable.set(value)
            self.assertEqual(MultiStepOverlapGUI._selected_ipf_direction(self.gui_stub), expected)

    def test_invalid_direction_falls_back_to_z(self) -> None:
        self.variable.set("invalid")
        self.assertEqual(MultiStepOverlapGUI._selected_ipf_direction(self.gui_stub), ("z", "IPF-Z"))
        self.assertEqual(self.variable.get(), "Z")

    def test_close_waits_for_active_worker(self) -> None:
        gui_stub = SimpleNamespace(
            busy=True,
            _worker_thread=None,
            session=Mock(),
            destroy=Mock(),
        )
        with patch("multistep_overlap_ebsd.gui.messagebox.showinfo") as showinfo:
            MultiStepOverlapGUI._on_close(gui_stub)
        showinfo.assert_called_once()
        gui_stub.session._clear_dictionary_cache.assert_not_called()
        gui_stub.destroy.assert_not_called()

    def test_idle_close_cleans_temporary_dictionary_cache(self) -> None:
        gui_stub = SimpleNamespace(
            busy=False,
            _worker_thread=None,
            session=Mock(),
            destroy=Mock(),
        )
        MultiStepOverlapGUI._on_close(gui_stub)
        gui_stub.session._clear_dictionary_cache.assert_called_once()
        gui_stub.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
