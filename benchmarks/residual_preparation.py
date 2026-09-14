"""Measure restored-workflow residual preparation without changing source files.

Compares the serial reconstruction in f315c75 with the current implementation.
Workflow loading and the initial projection/JIT warm-up are excluded from timing.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import dask
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd import core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--offset", type=int, default=3000)
    parser.add_argument("--cores", type=int, default=10)
    parser.add_argument("--reference", default="f315c75")
    parser.add_argument("--match", action="store_true")
    args = parser.parse_args()
    source = subprocess.run(
        ["git", "show", f"{args.reference}:multistep_overlap_ebsd/core.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "WorkflowSession")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_residual_signal_from_indices")
    namespace = dict(vars(core))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<reference>", "exec"), namespace)
    reference = namespace[method.name]
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def measure(label, action):
        start = perf_counter()
        result = action()
        record = dict(label=label, seconds=perf_counter()-start)
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        return result, record

    session = core.WorkflowSession()
    try:
        print(session.restore_workflow_state(args.workflow), flush=True)
        indices = np.array(sorted(session.residual_point_results))[args.offset:args.offset+args.count]
        if indices.size != args.count:
            raise ValueError("Insufficient stored fits for the requested batch.")
        # Force reconstruction from fit metadata, even if an exported source exists.
        session.residual_pattern_output_path = None
        metadata = {i: session._strip_residual_point_result(r) for i, r in session.residual_point_results.items()}
        records.append(dict(count=int(indices.size), cores=args.cores, reference=args.reference,
                            map_count=session.data.count, binning=session.dictionary_cache.software_binning))
        with dask.config.set(scheduler="threads", num_workers=args.cores):
            session._simulate_pattern_for_euler(int(indices[0]), session.current_eulers_rad[indices[0]])
            expected, _ = measure("serial_restored_preparation", lambda: session._materialize_signal_batch(
                reference(session, indices)))
            session._clear_residual_pattern_store()
            session.residual_point_results = dict(metadata)
            callbacks = []
            actual, record = measure("batched_restored_preparation", lambda: session._materialize_signal_batch(
                session._residual_signal_from_indices(indices, progress_callback=callbacks.append)))
            record["max_pixel_difference"] = float(np.max(np.abs(actual.data-expected.data)))
            record["pixels_equal"] = bool(np.array_equal(actual.data, expected.data))
            record["progress_callbacks"] = len(callbacks)
            np.testing.assert_array_equal(actual.data, expected.data)
            cached, _ = measure("cached_residual_preparation", lambda: session._materialize_signal_batch(
                session._residual_signal_from_indices(indices)))
            np.testing.assert_array_equal(cached.data, expected.data)
            cache = session.dictionary_cache
            measure("primary_preparation", lambda: session._materialize_signal_batch(
                session._signal_from_indices(indices, software_binning=cache.software_binning, crop_extent=cache.crop_extent)))
            if args.match:
                _, record = measure("dictionary_matching_first_pass", lambda: session._dictionary_index_kikuchipy_signal(
                    actual, cache=cache, keep_n=5, signal_mask=session._signal_mask_for_dictionary_cache(cache)))
            args.output.write_text(json.dumps(records, indent=2)+"\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
