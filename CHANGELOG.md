# Release notes

## 0.2 — 2026-09-23

- Multi-phase primary and residual indexing with linked master patterns and dictionaries in tab 1; two-component fits can use the same or different phases.
- Phase maps, per-phase IPF keys and phase-labelled pattern previews. Saved workflows and H5OINA exports retain phase identities.
- Faster dictionary indexing and refinement; phase-aware parallel residual generation and mixture fitting. Default worker count is now 6.
- Tab 4 exports fitted primary/residual H5OINA solutions with full-range patterns. Non-overlap residual pixels are black.
- Fixes for workflow save paths, dictionary compatibility warnings, residual refinement seeds and exported lattice units.

Development and testing assisted by OpenAI Codex. See the [multi-phase audit](docs/multiphase_gui_audit_2026-09-22.md) for validation and limits. Mixtures remain limited to two components per pixel.

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
