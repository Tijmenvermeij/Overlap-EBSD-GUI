"""Measure the existing residual-construction pipeline on a workflow sample.

No source/workflow output is written. Does not index or refine the residuals.
The clock includes worker startup, input reading, projection, fitting, transfer
and storage, but excludes restoring the workflow before starting the run.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd.core import WorkflowSession
from multistep_overlap_ebsd.cpu_fitting import FIT_METHOD_DEFAULT, FIT_METHOD_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', type=Path)
    parser.add_argument('--cores', type=int, default=10)
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--fit-method', choices=tuple(FIT_METHOD_LABELS), default=FIT_METHOD_DEFAULT)
    parser.add_argument('--direct-h5-dataset', help='Investigation only: read batches directly from this verified pattern dataset.')
    parser.add_argument('--output', type=Path, default=Path('/tmp/step3_pipeline_results.json'))
    args = parser.parse_args()
    session = WorkflowSession()
    with patch.object(session, 'load_dictionary', return_value='Skipped dictionary for residual-only benchmark.'):
        print(session.restore_workflow_state(str(args.workflow)), flush=True)
    ui = session.restored_ui_state
    scores = session.last_scores_map.reshape(-1)
    eligible = np.flatnonzero(np.isfinite(scores) & (scores >= float(ui['overlap_min_ncc'])))
    indices = np.sort(np.random.default_rng(20260909).choice(eligible, min(args.count, len(eligible)), replace=False))
    progress = []
    start = perf_counter()

    def callback(percent, message):
        elapsed = perf_counter()-start
        progress.append(dict(seconds=elapsed, percent=percent, message=message))
        print(f'{elapsed:.3f}s {percent:.1f}% {message}', flush=True)

    with ExitStack() as resources:
        if args.direct_h5_dataset:
            if session.data.source_type != 'h5oina' or session.dynamic_bg_config.enabled:
                raise ValueError('Direct-input prototype requires H5OINA without dynamic background.')
            source = resources.enter_context(h5py.File(session.data.pattern_path, 'r'))
            dataset = source[args.direct_h5_dataset]
            if dataset.shape != (session.data.count, session.data.h, session.data.w):
                raise ValueError('Unexpected dataset shape.')

            def direct_input(batch_indices, *, software_binning, crop_extent):
                if software_binning != 1 or crop_extent != (0, session.data.h, 0, session.data.w):
                    raise ValueError('Direct-input prototype requires full-resolution full-frame selection.')
                ordered, inverse = np.unique(batch_indices, return_inverse=True)
                return SimpleNamespace(data=np.asarray(dataset[ordered], dtype=np.float32)[inverse])

            resources.enter_context(patch.object(session, '_signal_from_indices', side_effect=direct_input))
        session.compute_overlap_residual_indices(
            indices, fit_blur_gain=bool(ui['fit_blur_gain']),
            fit_maxiter=int(ui['gain_fit_maxiter']), fit_popsize=int(ui['gain_fit_popsize']),
            fit_method=args.fit_method,
            fit_bounds=ui['primary_fit_bounds'], parallel_cores=args.cores,
            write_patterns=False, selected_index=None, progress_callback=callback)
    seconds = perf_counter()-start
    image_fields = ('experimental', 'simulated', 'residual', 'simulated_unfitted', 'blurred_simulated', 'gain_map')
    image_bytes = sum(value.nbytes for i in indices for field in image_fields
                      if (value := getattr(session.residual_point_results[int(i)], field)) is not None)
    result = dict(count=len(indices), cores=args.cores, seconds=seconds,
                  direct_h5_dataset=args.direct_h5_dataset,
                  fit_method=args.fit_method,
                  residual_cache_bytes=Path(session._residual_pattern_store.path).stat().st_size,
                  patterns_per_second=len(indices)/seconds, image_bytes=image_bytes,
                  indices=indices.tolist(), progress=progress)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('progress','indices')}), flush=True)


if __name__ == '__main__':
    main()
