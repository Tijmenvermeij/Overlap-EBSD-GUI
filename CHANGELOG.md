# Release notes

## Unreleased

- Recognize H5OINA Pattern Matching NCC as existing primary indexing on fresh input load, including files without GUI export metadata. Previously residual patterns and their saved orientations become the new primary input; no parent residual/mixture state is imported. Preserve imported indexing when attaching the first master and loading a dictionary with the same energy settings, so Step 3 can extract/index a further overlap without primary DI. Validate score counts, phases, orientations and export masks, and show the imported point count in Step 2.
- Prioritize computation over map refresh in all steps: use full memory-bounded batches, throttle expensive previews toward 1% overhead, draw only the visible tab, and remove forced redraws at combined-workflow stage transitions. Refinement batches no longer shrink toward a five-second refresh target.
- Compile the primary/residual orientation Nelder–Mead loop and reuse detector direction cosines when all PCs are equal. Keep Kikuchipy's projection, NCC and public refinement API, all candidate seeds, trust regions and evaluation limits. A 1,536-point primary sample with six cores improved from 24.61 to 18.62 seconds at full resolution; binned refinement improved from 11.38 to 4.52 seconds (two-run means). Compared orientations and scores were exactly equal.
- Speed up Step 3 residual refinement by generating inspection simulations only for the selected point. On 320 residuals with five candidates and ten cores, that earlier improvement reduced refinement from 11.83 to 2.92 seconds with exactly equal orientations, scores and selected-point images.
- Show current/total batches throughout primary/residual dictionary indexing and orientation refinement, plus current point ranges/total. Calibration, residual fitting, mixture fitting and dictionary generation retain completed/total counts.
- Extend the global CPU worker limit to primary/residual dictionary indexing and orientation/PC refinement, including selected points and combined workflows.
- Index with aligned dictionary blocks, reusable float32 NCC normalization statistics, prepared experimental batches and a single top-candidate selection. Preserve progress/cancellation within dictionary scans without shrinking batches to a few patterns. Exact score ties use the lowest dictionary index; undefined NCC values remain NaN.
- Deduplicate repeated refinement seeds, reuse the selected master energy and adapt navigation chunks to the number of candidate fits and workers. Preserve all distinct candidate seeds and the existing optimization settings.
- Speed up Step 3 batch preparation after workflow loading: reconstruct residuals with batched input reads and projections, reuse the selected master, and skip unused inspection images. Preserve existing cached rows and report progress/cancellation between reconstruction batches. A 3,000-point batch on the supplied workflow improved from 82.45 to 2.10 seconds with exactly equal residual patterns.
- Add a shared Steps 3–4 fitting selector: blur then gain (default), blur then gain plus joint refinement, and joint search. Preserve the selection in workflows and Step 4 export settings.
- Reduce primary gain fitting analytically where bounds permit; cache Gaussian blurs, coordinates and weighted moments. Step 4 retains its shared gain map and nonnegative mixture coefficients. Optional joint refinement retains the staged result if the final reconstructed residual would worsen.
- Read H5OINA batches directly from verified pattern datasets, preserving conditioning and selection order. Batch worker projections with individual PCs and retain the selected master energy in each worker.
- Store exact float32 residuals in a temporary disk cache and bound inspection images in both steps. Transfer compact worker results, read cached residuals in batches, and release temporary files when sessions are replaced or closed.
- On the supplied Cu workflow, an 80-point residual-construction sample on 10 cores improved from 6.16 to 19.66 patterns/s including startup. Six Step 4 fits improved from 7.30 to 1.28 seconds with the default staged method. These are sample timings; see [validation details](docs/cpu_performance.md).

## 0.1 — 2026-09-09

First usable GUI release for indexing overlapping EBSD/TKD patterns within a single crystal phase. Primary and secondary orientations use the same master-pattern phase; this release does not provide a simultaneous search across different phases.

### Workflow

- Four tabs cover loading and pattern-center calibration, dictionary indexing, residual indexing, and mixture optimization.
- H5OINA and UP + ANG inputs, reusable dictionaries, workflow save/restore, and primary/residual exports support the full workflow.
- Primary and residual indexing share refinement settings. Automatic refinement defaults to full-resolution patterns, five retained matches, and a search range following the dictionary spacing.
- Combined actions run steps 2–3 or 2–4 over the selected ROI. Shared conditioning, worker-core controls for residual/mixture fits, and cooperative cancellation are available throughout the GUI.
- Maps refresh during indexing, refinement, calibration optimization and mixture fitting, targeting five seconds between updates. Completed-batch boundaries keep displayed results consistent; a slow batch or individual solver can delay a refresh. The selected tab, inspection point and map zoom are preserved during live refreshes.

### Calibration and result integrity

- Separate calibration-point optimization from applying the average pattern center to the scan. Pending changes are visible and survive workflow save/restore.
- Preserve per-point pattern centers and detector geometry in workflow files, H5OINA exports and ANG companion files.
- Infer effective detector pixel size for recognized Oxford camera modes; require an explicit scale where metadata are insufficient. Scan-position correction checks supported geometry before applying it.
- Display H5OINA tilt angles to millidegrees to remove floating-point conversion noise, while retaining full precision for calculations.
- Recompute legacy detector rays when the pattern center changes. Preserve blur settings in parallel residual processing.
- Invalidate affected scores, residual orientations and mixture fits when their inputs change. Keep completed indexing/refinement results when cancellation stops a run, and discard incomplete residual output files.
- Keep Tk updates on the GUI thread and limit queued residual-pattern data to control memory use.

### Validation and scope

Tested on macOS with Python 3.12.13, kikuchipy 0.12.1, orix 0.14.2, NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0 and Matplotlib 3.10.9.

- 113 automated checks pass, covering workflow persistence and exports, geometry and calibration, result invalidation, cancellation, GUI actions, and live-update coordination.
- A running Tk GUI check with a synthetic scan verified intermediate indexing and refinement map updates before completion.
- These checks establish software behavior; they are not an experimental accuracy benchmark. Indexing quality, overlap fractions and refinement accuracy still require assessment against representative measured EBSD/TKD data and independent reference orientations.
- The Oxford pixel-size assumptions and supported scan geometry are described in [calibration and detector geometry](docs/calibration.md). Mixture coefficients depend on the fit and pattern conditioning; they are not calibrated physical volume fractions.
