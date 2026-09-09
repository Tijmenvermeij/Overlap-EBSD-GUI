"""Extract reproducible Step 3 input pairs from a saved workflow (read only).

Dictionary loading is skipped because residual construction only requires the
saved primary orientations, PCs, conditioning, and master pattern.
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multistep_overlap_ebsd.core import WorkflowSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/tmp/step3_workflow_pairs.npz'))
    args = parser.parse_args()
    session = WorkflowSession()
    with patch.object(session, 'load_dictionary', return_value='Skipped dictionary for residual-only benchmark.'):
        print(session.restore_workflow_state(str(args.workflow)), flush=True)
    scores = np.asarray(session.last_scores_map).ravel()
    threshold = float(session.restored_ui_state.get('overlap_min_ncc', .15))
    eligible = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
    ordered = eligible[np.argsort(scores[eligible], kind='stable')]
    indices = [int(ordered[round(q*(len(ordered)-1))]) for q in (.1, .3, .5, .7, .9)]
    indices = list(dict.fromkeys(indices + [int(session.restored_ui_state['selected_index'])]))
    experimental, simulated, elapsed = [], [], []
    for idx in indices:
        start = perf_counter()
        exp = session._processed_pattern_at(idx)
        split = perf_counter()
        sim = session._simulate_pattern_for_euler(idx, session.current_eulers_rad[idx])
        finish = perf_counter()
        experimental.append(exp)
        simulated.append(sim)
        elapsed.append([split-start, finish-split])
        print(dict(index=idx, primary_ncc=float(scores[idx]), shape=exp.shape,
                   prepare_seconds=elapsed[-1]), flush=True)
    np.savez(args.output, experimental=np.stack(experimental), simulated=np.stack(simulated),
             weights=session._overlap_weights(), indices=indices, prepare_seconds=elapsed,
             primary_scores=scores[indices], workflow=str(args.workflow),
             maxiter=session.restored_ui_state['gain_fit_maxiter'],
             popsize=session.restored_ui_state['gain_fit_popsize'],
             bounds=session.restored_ui_state['primary_fit_bounds'])


if __name__ == '__main__':
    main()
