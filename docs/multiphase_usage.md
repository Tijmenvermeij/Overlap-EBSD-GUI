# Multi-phase indexing

In Tab 1, imported phases appear as separate rows, including phases without
pixels in the selected ROI. Select a row and use **Link master to selected
phase…**, or use **Add phase / master…** for a new material. Multiple master files
can be selected at once. The selected row owns its dictionary; matching names
or numeric IDs alone do not establish dictionary compatibility.

Set the shared orientation spacing and software binning, then generate missing
or incompatible dictionaries. Save each dictionary from its selected row to keep
it across sessions. A saved dictionary records its master fingerprint, crystal
metadata, geometry, energy selection and generation settings. Dictionaries from
older versions lack that provenance: regenerate them for multi-phase searches.
Different phases can have different dictionary lengths because of their symmetry.

Tab 2 searches every enabled phase, retaining orientation candidates separately
for each phase and comparing them again after refinement. **Keep imported phase
assignments** restricts primary matching to the input's assignments. Residual
matching in Tab 3 still searches all enabled phases, including the primary phase.
The existing Steps 2–3, 2–4 and 3–4 buttons use these same processing routes.

The number of candidate phases is unrestricted. Each fitted pixel still has at
most two components, which may have the same phase or different phases. Mixture
fitting compares a bounded shortlist (up to four phase pairs); it is not an
exhaustive search over every pair and orientation. Each component uses its own
master, including when inspecting saved results. Step 4 orientation refinement
keeps its results separate from Steps 2–3 in multi-phase sessions.

**Phase maps / IPF keys…**, available in the workflow tabs, shows primary and
residual phases, fitted phase pairs, the dominant fitted phase, per-phase fitted
contributions, score gaps and overlap diagnostics. Select a phase to filter the
map and display its symmetry-specific IPF key. Names include output IDs so
identically named phases remain distinguishable. Colors and names can be edited
without changing scientific identity. Contributions are fitted pattern
intensities, not calibrated material volume fractions.

Overlap diagnostics use provisional thresholds: at least 0.01 NCC improvement
against the better single-component fit, at least 5% contribution from each
component, component correlation no greater than 0.98 in magnitude, and a
converged optimizer. Competing phase pairs within 0.005 NCC are marked ambiguous.
These thresholds are not calibrated detection limits. Rejected and ambiguous
fits remain available for inspection; consult the acceptance layer before
interpreting fitted pair or contribution maps. After a separate Step 4 orientation
refinement, rerun mixture fitting to reassess acceptance.

## Saving and reopening

Use the existing primary ROI, residual ROI and Step 4 export controls. No extra
export modes are introduced. Component exports contain ordinary phase IDs and a
complete phase catalog, never a synthetic A+B crystallographic phase. Step 4
HDF5 stores both component keys, their catalog, contributions and acceptance
diagnostics. ANG retains the existing per-point PC sidecar; its confidence column
contains NCC rather than native EDAX confidence.

Loading an exported H5OINA starts a fresh analysis with its stored component as
primary indexing. A residual export with patterns can therefore feed a further
analysis pass. Available, fingerprint-verified master/dictionary links are
restored; missing assets remain unresolved rows. Files without pattern payloads
support map viewing, but cannot be indexed or fitted. Opening a workflow instead
restores both components, candidates, results and saved asset links. Temporary
unsaved dictionaries must be regenerated after closing the original session.

Changing the enabled phases marks the previous comparison as using an older
phase selection. Re-index to compare the new set. Changing the ROI preserves
results outside the ROI. Dictionary matching commits complete batches before
cancellation; a partially compared batch is not committed.

## Validation and current limits

Automated checks include three actual Kikuchipy master signals, dictionary
creation, primary and residual indexing/refinement, three-asset workflow restore,
known same-phase/different-phase mixture contributions and single-component
controls, and primary/residual H5OINA round trips with sparse phase IDs, duplicate
names and cubic/hexagonal metadata. Repeated export/reload checks guard against
double hexagonal axis conversion. The GUI has a live smoke test for control
placement and phase rows.

Multi-phase residual generation and mixture fitting currently run points
sequentially to avoid the former one-master process-worker assumption. Dictionary
matching and orientation refinement use the configured worker limit. More phases
increase runtime; experimental accuracy and performance on representative large
multi-phase scans still need measurement. Synthetic tests do not establish
experimental phase discrimination or Oxford AZtec re-import compatibility.

When loading a saved dictionary, a difference only in the pattern center can be
accepted explicitly in the confirmation dialog. The phase table then shows
“Ready (PC accepted)”. Dictionary matching uses the saved patterns as an
approximate starting point; orientation refinement uses the current scan PC.
The original dictionary metadata is preserved. Acceptance is recorded in the
workflow for that dictionary and current PC; a subsequent PC change requires
acceptance again. Differences in master, crystal structure, or other simulation
settings are not covered by this option.
