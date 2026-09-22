# Multi-phase GUI audit — 22 September 2026

## Scope and evidence

Reviewed the four active workspaces, phase/master/dictionary associations,
primary and residual indexing/refinement, mixture fitting, phase-aware previews,
ROI handling, workflow persistence, worker dispatch, and component exports.
The automated suite passed (228 tests), followed by the added fitted-export GUI
routing test. A native Tk smoke test initialized the GUI and verified both new
export buttons; their requested widths were 224 and 226 pixels. This is source,
regression and targeted native-GUI validation, not certification of every
interactive sequence or experimental phase discrimination.

## Findings and fixes

- **Multi-phase mixture fitting bypassed multiprocessing.** It now uses the
  selected process-worker count, loads masters on demand per worker, and sends
  only bounded pattern batches and their candidate metadata. Workers execute
  the same alternative phase-pair comparison, single-component baselines,
  ambiguity checks and overlap acceptance logic as sequential fitting.
  Real spawned-worker comparisons cover one, two and three phases, including
  alternative candidates; component IDs, orientations, coefficients, NCC and
  acceptance agree with sequential results.
- **Single-candidate residual refinement could use the previous phase's seed.**
  The copied per-point result now receives the current phase's candidate and
  phase key before refinement. A regression test exercises three phases.
- **EMsoft lattice lengths could be exported in nm as if they were angstroms.**
  Phase export now reads the explicit EMsoft crystal dataset and converts
  lengths to angstroms. It leaves dictionary provenance untouched. This also
  avoids retaining a surrogate imported lattice when a master has been linked.
  EMsoft's units are documented in its [crystallography examples](https://github.com/EMsoft-org/EMsoft/blob/develop/Source/pyEMsoft/docs/pyEMsoft.rst).
- **Tab 4 lacked fitted component pattern exports.** Both new buttons produce
  full-map H5OINA solutions/patterns using optimized component phase IDs and
  orientations. Tests cover component arithmetic, full dynamic range, black
  non-overlap residuals, constant/uint16 scaling, input protection, cancellation,
  repeated exports without changing live results, and reading the exported NCC
  and phase maps as new primary indexing.

## Checked existing behaviour

| Area | Assessment |
| --- | --- |
| Input and workflow loading | Fresh inputs reset the workflow destination. Restores retain their explicit path. Tests cover transaction boundaries, old state and saved multi-phase assets. |
| PC/master/dictionary binding | Content/structure checks remain strict. Only explicitly approved PC/tilt differences are reusable; acceptance is scoped to current geometry and persisted. |
| Dictionary generation/indexing | Worker limits propagate. Shared batching bounds input and score memory; input patterns are prepared once across phases. |
| Primary/residual refinement | Phase-local seeds and master dispatch, candidate retention, cancellation and worker scoping are covered. Refinement scheduling was measured separately to use 12 concurrent tasks. |
| Residual generation | Phase-aware process workers verified for one, two and three phases; correct PCs and blur/gain settings are retained. |
| Mixture fitting | Same-phase and different-phase pairs use their own masters. Alternative-pair search and acceptance are preserved in parallel. Only complete returned batches are committed. |
| Maps and inspection | Shared ROI bounds apply to tabs 2–4. Phase/IPF selectors, legends, per-result simulation phase labels and full-map/ROI phase window are covered by existing GUI tests. Step 4 results remain separate from Steps 2–3. |
| Saved results and exports | Component keys and optimized orientations are preserved. New exports are full scan, include the phase catalogue/current PCs, and do not overwrite the source. H5OINA export uses existing hexagonal Euler correction handling. |
| UI actions | New export actions reject concurrent work, synchronize conditioning first, handle cancelled save dialogs, and report cancellable progress through the Step 4 status area. |

## Export definition and limits

The GUI uses accepted overlap fits. In fit-normalized units, residual is
`E - a1*S1`; primary is `E - residual = a1*S1`. Non-overlap primary images are
original measured patterns; non-overlap residual images and solutions are zero.
Each image is scaled independently only after subtraction. See
[multiphase usage](multiphase_usage.md) for details.

Overlap acceptance remains a provisional scientific diagnostic. Separate
orientation-only mixture refinement marks acceptance unassessed until mixture
fitting is rerun; those points therefore use the non-overlap export fallback.
A mixture still contains two components, selected among any number of enabled
phases; this does not implement three simultaneous components at a pixel.
Native Oxford AZtec re-import was not available for validation. Large production
ROI throughput and RAM usage with many master patterns were not benchmarked in
this audit; each worker caches the masters it actually uses.
