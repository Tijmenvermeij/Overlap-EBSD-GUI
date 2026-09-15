from __future__ import annotations

import queue
import threading
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
from matplotlib.figure import Figure

from multistep_overlap_ebsd.core import WorkflowSession
from multistep_overlap_ebsd.gui import MultiStepOverlapGUI
from multistep_overlap_ebsd.live_updates import LiveUpdateGate, progress_batches


class LiveUpdateTests(unittest.TestCase):
    def test_worker_waits_for_main_thread_snapshot_before_next_write(self):
        now = [0.0]
        callbacks = queue.Queue()
        gate = LiveUpdateGate(clock=lambda: now[0])
        state = []
        seen = []
        resumed = threading.Event()
        main = threading.get_ident()
        gate.at_boundary(callbacks.put, lambda: self.fail("Redraw too early"))
        now[0] = 5.0

        def worker():
            state.append(1)
            gate.at_boundary(callbacks.put, lambda: seen.append((threading.get_ident(), state.copy())))
            state.append(2)
            resumed.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        render = callbacks.get(timeout=2)
        self.assertFalse(resumed.is_set())
        self.assertEqual(state, [1])
        render()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(seen, [(main, [1])])
        self.assertEqual(state, [1, 2])
        gate.at_boundary(callbacks.put, lambda: self.fail("Repeated unchanged frame"))
        self.assertTrue(callbacks.empty())

    def test_render_failure_releases_worker(self):
        callbacks = queue.Queue()
        gate = LiveUpdateGate(interval=0)
        thread = threading.Thread(
            target=lambda: gate.at_boundary(callbacks.put, Mock(side_effect=ValueError("draw failed"))),
            daemon=True,
        )
        thread.start()
        with self.assertRaisesRegex(ValueError, "draw failed"):
            callbacks.get(timeout=2)()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_gui_queue_applies_progress_before_snapshot_and_stays_on_main_thread(self):
        main = threading.get_ident()
        events = []
        gui = SimpleNamespace(_ui_callback_queue=queue.Queue(), _job_update_gate=LiveUpdateGate(interval=0))
        gui._refresh_job_maps = lambda: events.append(("draw", threading.get_ident()))
        gui._post_ui = MethodType(MultiStepOverlapGUI._post_ui, gui)
        gui._worker_thread = threading.Thread(
            target=lambda: gui._post_ui(lambda: events.append(("progress", threading.get_ident()))),
            daemon=True,
        )
        gui._worker_thread.start()
        gui._ui_callback_queue.get(timeout=2)()
        gui._ui_callback_queue.get(timeout=2)()
        gui._worker_thread.join(timeout=2)
        self.assertFalse(gui._worker_thread.is_alive())
        self.assertEqual(events, [("progress", main), ("draw", main)])

    def test_batches_fill_memory_cap_without_skipping_or_repeating_points(self):
        points = np.arange(123)[::-1]
        batches = []
        for start, batch in progress_batches(points, 32):
            self.assertEqual(start, sum(map(len, batches)))
            batches.append(batch)
        self.assertEqual(list(map(len, batches)), [32, 32, 32, 27])
        np.testing.assert_array_equal(np.concatenate(batches), points)

    def test_partial_final_batch(self):
        batches = [batch for _, batch in progress_batches(np.arange(35), 20)]
        self.assertEqual(list(map(len, batches)), [20, 15])

    def test_expensive_drawing_reduces_refresh_frequency(self):
        now = [0.]
        gate = LiveUpdateGate(clock=lambda: now[0])
        def slow_render():
            now[0] += 2.
        render = Mock(side_effect=slow_render)
        now[0] = 5.
        gate.at_boundary(lambda callback: callback(), render)
        self.assertEqual(render.call_count, 1)
        now[0] += 197.
        gate.at_boundary(lambda callback: callback(), render)
        self.assertEqual(render.call_count, 1)
        now[0] += 1.
        gate.at_boundary(lambda callback: callback(), render)
        self.assertEqual(render.call_count, 2)

    def test_live_redraw_preserves_tab_and_scan_zoom(self):
        views = {}
        for view in range(4):
            axes = Figure().subplots(1, 2)
            axes[0].imshow(np.zeros((6, 8)))
            axes[0]._overlap_ebsd_scan_map = True
            axes[0].set_xlim(2, 5)
            axes[0].set_ylim(4, 1)
            views[view] = {"axes": axes, "canvas": Mock()}
        drawn = []

        def redraw(*, view_index):
            drawn.append(view_index)
            axes = views[view_index]["axes"]
            for ax in axes:
                ax.clear()
                ax.imshow(np.ones((6, 8)))

        gui = SimpleNamespace(
            workflow_notebook=SimpleNamespace(select=lambda: "mixture", index=lambda _: 3),
            _plot_views=views, _refresh_plot=redraw, _activate_plot_view=Mock(),
        )
        MultiStepOverlapGUI._refresh_complete_analysis_maps(gui, 1)
        self.assertEqual(drawn, [3, 1])
        for view in drawn:
            self.assertEqual(views[view]["axes"][0].get_xlim(), (2, 5))
            self.assertEqual(views[view]["axes"][0].get_ylim(), (4, 1))
        gui._activate_plot_view.assert_called_once_with(3)

    def test_live_refresh_draws_visible_tab_once_for_multiple_changed_stages(self):
        gui = SimpleNamespace(
            busy=True, session=SimpleNamespace(data=object()), _job_result_views={1, 2, 3},
            workflow_notebook=SimpleNamespace(select=lambda: "residual", index=lambda _: 2),
            _populate_point_vars=Mock(), _refresh_complete_analysis_maps=Mock(),
            live_update_status_var=Mock(), _activate_plot_view=Mock(), _log=Mock(),
        )
        MultiStepOverlapGUI._refresh_job_maps(gui)
        gui._refresh_complete_analysis_maps.assert_called_once_with(2)
        self.assertEqual(gui._job_result_views, set())

    def test_live_redraw_does_not_restore_previous_roi_on_residual_or_mixture_tab(self):
        for active in (2, 3):
            with self.subTest(active=active):
                ax = Figure().subplots()
                ax.imshow(np.zeros((12, 14)))
                ax._overlap_ebsd_scan_map = True
                ax.set_xlim(-.5, 2.5)
                ax.set_ylim(2.5, -.5)
                view = dict(axes=[ax], canvas=Mock(), roi_bounds=(0, 0, 3, 3))
                def redraw(*, view_index):
                    ax.clear()
                    ax.imshow(np.ones((12, 14)))
                    ax.set_xlim(4.5, 10.5)
                    ax.set_ylim(8.5, 3.5)
                    view['roi_bounds'] = (4, 5, 5, 6)
                gui = SimpleNamespace(
                    workflow_notebook=SimpleNamespace(select=lambda: active, index=lambda x: x),
                    _plot_views={active: view}, _refresh_plot=redraw, _activate_plot_view=Mock(),
                )
                MultiStepOverlapGUI._refresh_complete_analysis_maps(gui, active)
                self.assertEqual(ax.get_xlim(), (4.5, 10.5))
                self.assertEqual(ax.get_ylim(), (8.5, 3.5))

    def test_each_progress_kind_marks_its_relevant_view(self):
        for method, view in (("_set_reindex_progress", 1), ("_set_refinement_progress", 1),
                             ("_set_overlap_progress", 2), ("_set_overlap_optimization_progress", 3)):
            gui = SimpleNamespace(_job_result_views=set(), _set_progress_state=Mock())
            for name in ("reindex_progress_var", "reindex_progress_status_var", "refinement_progress_var",
                         "refinement_progress_status_var", "overlap_progress_var", "overlap_progress_status_var",
                         "overlap_optimization_progress_var", "overlap_optimization_status_var"):
                setattr(gui, name, Mock())
            getattr(MultiStepOverlapGUI, method)(gui, 50, "Half complete")
            self.assertEqual(gui._job_result_views, {view})

    def test_calibration_batching_preserves_small_selection_retry_policy(self):
        for count in (3, 20):
            session = WorkflowSession()
            session.data = SimpleNamespace(h=128, w=156)
            session.master = SimpleNamespace(kind="kikuchipy")
            session.current_phases = np.ones(count, dtype=int)
            session.current_eulers_rad = np.zeros((count, 3))
            session.current_pc_bruker = np.ones((count, 3))
            session.current_pc_custom = np.ones((count, 3))
            session._refine_indices_kikuchipy = Mock(return_value="Refined")
            progress = Mock()
            session.refine_indices(np.arange(count), 1, progress_callback=progress)
            calls = session._refine_indices_kikuchipy.call_args_list
            np.testing.assert_array_equal(np.concatenate([c.kwargs["indices"] for c in calls]), np.arange(count))
            self.assertTrue(all(c.kwargs["allow_pc_retry"] == (count <= 10) for c in calls))
            self.assertEqual(progress.call_args.args[0], 100)


if __name__ == "__main__":
    unittest.main()
