"""Read-only refinement scaling experiment against a saved workflow.

No session state is changed. The reported kernel time includes Dask scheduling
but excludes graph construction, input loading and master-pattern loading;
these costs are recorded separately. Source patterns are never written out.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 80, 640])
    parser.add_argument("--maxfev", type=int, default=25)
    parser.add_argument("--processes", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--binning", type=int, default=1)
    args = parser.parse_args()
    import dask
    import kikuchipy as kp
    import numba
    import scipy
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation
    from threadpoolctl import threadpool_info, threadpool_limits

    session = core.WorkflowSession()
    report = {
        "versions": dict(python=sys.version.split()[0], numpy=np.__version__,
                         scipy=scipy.__version__, kikuchipy=kp.__version__,
                         numba=numba.__version__, dask=dask.__version__),
        "maxfev": args.maxfev,
        "binning": args.binning,
        "logical_cpus": os.cpu_count(),
        "threadpools": threadpool_info(),
        "measurement_notes": [
            "Single timing per configuration; these are sampled-fit measurements, not a whole-map benchmark.",
            "Map-point throughput with distinct keep_n candidates must account for each candidate fit.",
            "Input loading and selected master materialization are reported separately from graph and kernel times.",
            "Process CPU time covers only the coordinator for multiprocessing; it is not total worker utilization.",
            "Generic Dask processes serialize task data and master arrays; a dedicated persistent worker may behave differently.",
        ],
        "records": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    start = perf_counter()
    with patch.object(session, "load_dictionary", return_value="Skipped dictionary for refinement benchmark."):
        session.restore_workflow_state(str(args.workflow))
    report["restore_seconds"] = perf_counter() - start
    report["saved_settings"] = {key: session.restored_ui_state.get(key) for key in
                                 ("trust_euler", "trust_pc", "maxfev", "dictionary_keep_n")}
    saved_candidates = session.indexed_candidate_eulers_rad
    report["saved_primary_candidates"] = None if saved_candidates is None else dict(
        shape=list(saved_candidates.shape),
        finite_seed_count=int(np.all(np.isfinite(saved_candidates), axis=-1).sum()))
    eligible = np.flatnonzero(np.all(np.isfinite(session.current_eulers_rad), axis=1)
                              & (session.current_phases >= 0))
    indices = eligible[np.linspace(0, len(eligible)-1, max(args.sizes), dtype=int)]
    phase_id = int(session.current_phases[indices[0]])
    assert np.all(session.current_phases[indices] == phase_id)
    phase_list = session._phase_list_for_current_master(phase_id)
    crop = None if args.binning == 1 else session._binning_crop_extent(args.binning)
    start = perf_counter()
    signal = session._refinement_signal_from_indices(indices, software_binning=args.binning,
                                                     crop_extent=crop)
    all_patterns = np.asarray(signal.data)
    report["input_seconds"] = perf_counter()-start
    report["pattern_shape"] = list(all_patterns.shape[1:])
    report["pattern_dtype"] = str(all_patterns.dtype)
    report["indices"] = indices.tolist()
    eulers = session._eulers_to_kikuchipy_frame(session.current_eulers_rad[indices])
    trust = float(session.restored_ui_state.get("trust_euler", 1.4))
    trust_pc = float(session.restored_ui_state.get("trust_pc", .005))
    mask = session._signal_mask_for_full_pattern() if args.binning == 1 else None
    source_mp = session.master.mp_signal

    def graph(n, mode="orientation", chunk=None, materialized=False, candidates=1):
        sig = kp.signals.EBSD(np.repeat(all_patterns[:n], candidates, axis=0)
                             if candidates > 1 else all_patterns[:n])
        sig.axes_manager.navigation_axes[0].name = "x"
        xmap = CrystalMap(rotations=Rotation.from_euler(np.repeat(eulers[:n], candidates, axis=0)),
                          phase_id=np.full(n*candidates, phase_id, dtype=np.int32),
                          x=np.arange(n*candidates, dtype=float), phase_list=phase_list)
        detector = session._kikuchipy_detector_for_indices(np.repeat(indices[:n], candidates), software_binning=args.binning,
                                                           crop_extent=crop)
        kwargs = dict(xmap=xmap, detector=detector,
                      master_pattern=eager_mp if materialized else source_mp,
                      energy=float(session.master.energy_kv), signal_mask=mask,
                      method="minimize", method_kwargs=dict(method="Nelder-Mead",
                      options=dict(maxfev=args.maxfev, disp=False)), compute=False,
                      rechunk=chunk != "serial")
        if isinstance(chunk, int):
            kwargs["chunk_kwargs"] = {"chunk_shape": chunk}
        if mode == "orientation":
            return sig.refine_orientation(trust_region=[trust]*3, **kwargs)
        if mode == "pc":
            return sig.refine_projection_center(trust_region=[trust_pc]*3, **kwargs)
        return sig.refine_orientation_projection_center(trust_region=[trust]*3+[trust_pc]*3, **kwargs)

    references = {}

    def run(n, mode, workers, chunk, *, materialized=True, scheduler="threads", label=None, candidates=1, pool=None):
        with dask.config.set(scheduler="threads", num_workers=workers):
            start = perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                result = graph(n, mode, chunk, materialized, candidates)
            build_seconds = perf_counter()-start
            cpu_start = process_time()
            start = perf_counter()
            compute_kwargs = dict(scheduler=scheduler, num_workers=workers)
            if pool is not None:
                compute_kwargs.update(pool=pool, chunksize=1)
            values = result.compute(**compute_kwargs)
            seconds = perf_counter()-start
            cpu_seconds = process_time()-cpu_start
        values = values.reshape(n, candidates, -1)
        duplicates_identical = bool(np.all(values == values[:, :1]))
        values = values[:, 0]
        reference_key = (n, mode)
        if reference_key not in references:
            references[reference_key] = values.copy()
        baseline = references[reference_key]
        record = dict(mode=mode, map_points=n, candidate_fits=n*candidates, candidates_per_point=candidates,
                      workers=workers, chunk=chunk,
                      scheduler=scheduler, master_materialized=materialized,
                      graph_seconds=build_seconds, kernel_seconds=seconds,
                      total_seconds=build_seconds+seconds,
                      map_points_per_second=n/seconds,
                      candidate_fits_per_second=n*candidates/seconds,
                      process_cpu_seconds=cpu_seconds,
                      mean_cpu_cores=cpu_seconds/seconds,
                      cpu_scope="coordinator_only" if scheduler == "processes" else "all_threads_in_current_process",
                      result_chunks=[list(c) for c in result.chunks],
                      graph_tasks=len(result.dask),
                      identical_to_reference=bool(np.array_equal(values, baseline)),
                      duplicate_seed_results_identical=duplicates_identical,
                      max_abs_difference=float(np.max(np.abs(values-baseline))))
        if label:
            record["label"] = label
        report["records"].append(record)
        print(json.dumps(record), flush=True)
        save()

    try:
        with threadpool_limits(limits=1):
            numba.set_num_threads(1)
            # Warm the exact projection and optimizer kernels before timing.
            with contextlib.redirect_stdout(io.StringIO()):
                graph(2, "orientation", "serial", False).compute(scheduler="single-threaded")
                graph(2, "pc", "serial", False).compute(scheduler="single-threaded")
                graph(2, "orientation_pc", "serial", False).compute(scheduler="single-threaded")
            start = perf_counter()
            energy_axis, energies = core._energy_axis_from_master_signal(source_mp)
            selected_energy = float(energies[np.argmin(abs(energies-float(session.master.energy_kv)))])
            eager_mp = core._master_signal_with_energy_weights(source_mp, np.array([selected_energy]),
                                                               np.array([1.0]))
            report["master_materialization_seconds"] = perf_counter()-start
            report["master_materialized_bytes"] = int(eager_mp.data.nbytes)
            if args.process_only:
                from concurrent.futures import ProcessPoolExecutor
                from multiprocessing import get_context
                n = max(args.sizes)
                run(n, "orientation", 10, 64, label="thread_reference")
                with ProcessPoolExecutor(max_workers=10, mp_context=get_context("spawn")) as pool:
                    run(n, "orientation", 10, 64, scheduler="processes", pool=pool,
                        label="generic_dask_cold_process_pool")
                    run(n, "orientation", 10, 64, scheduler="processes", pool=pool,
                        label="generic_dask_reused_process_pool")
                return
            for n in args.sizes:
                for workers, chunk in ((1, None), (10, None), (10, 4), (10, 16)):
                    run(n, "orientation", workers, chunk)
            for mode in ("pc", "orientation_pc"):
                for n in [min(args.sizes), min(80, max(args.sizes))]:
                    run(n, mode, 1, "serial")
                    run(n, mode, 10, 4)
            if saved_candidates is not None and not np.any(np.isfinite(saved_candidates)):
                for n in [min(args.sizes), min(80, max(args.sizes))]:
                    run(n, "orientation", 10, None, candidates=saved_candidates.shape[1],
                        label="current_all_missing_candidate_fallback")
                    run(n, "orientation", 10, 4, candidates=saved_candidates.shape[1],
                        label="same_fallback_with_small_chunks")
            for n in [min(args.sizes), max(args.sizes)]:
                run(n, "orientation", 10, 4, materialized=False, label="lazy_master")
            if args.processes:
                for n in [min(args.sizes), max(args.sizes)]:
                    run(n, "orientation", 10, 16, scheduler="processes")
    finally:
        session.close()
        save()


if __name__ == "__main__":
    main()
