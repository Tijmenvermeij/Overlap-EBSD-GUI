# Multi-phase overlap indexing implementation plan

Implementation started 21 September 2026. See [usage and validation notes](multiphase_usage.md) for the implemented workflow and current limits. This document retains the design and acceptance targets; experimental validation and multi-phase process-worker acceleration remain follow-up work.

## Recommended scope

Support an arbitrary list of candidate phases, with **up to two fitted components per pixel** in the first release. Both components may belong to the same phase or different phases. Retain the four-tab workflow and all existing single-phase workflows. More than two simultaneous components is a separate extension, not an automatic consequence of adding more master patterns.

The central concept is a **phase entry** that owns a master pattern and its compatible dictionaries. A dictionary is a derived asset of a master and simulation configuration, not an independently selected global file.

## 1. User workflow

Replace the single master-pattern controls with a compact phase table, visible in tab 1 and accessible from tab 2:

| Enabled | Phase | Color | Master pattern | Dictionary | Status |
|---|---|---|---|---|---|
| Yes | Austenite | Blue | Loaded | Ready | Ready |
| Yes | Ferrite | Orange | Loaded | Missing | Generate dictionary |

Use **Add phase / master…**, **Load dictionary…**, **Generate missing dictionaries**, and **Save dictionaries…** as the main actions. Selecting a row reveals its details; numeric output IDs, energy configuration, alternate dictionary versions and phase restrictions belong in advanced controls. Loading multiple master files can add several rows in one operation. Existing single-master controls become the one-row version of this table.

Keep all master and dictionary management in tab 1. Tab 2 shows a compact readiness summary and an **Edit phases/dictionaries…** shortcut. Provide one shared dictionary-settings section: consistent geometry, PC, comparison pixels, preprocessing and energy policy, with the same angular spacing by default. Phase symmetry and the resulting dictionary orientation counts can differ. Generate missing/incompatible dictionaries without regenerating compatible ones.

Read phase names and crystallography from master metadata. Match to imported phases using structure, lattice and symmetry metadata, not a name alone. If ambiguous, show a one-time mapping choice. Missing crystallographic metadata needs explicit completion; never silently assume cubic symmetry or fabricate lattice parameters. Keep imported phases without a loaded master visible as “Master missing.” Their original results remain viewable, but simulation-dependent actions for them are unavailable until resolved.

Tab 2 offers **Search enabled phases** by default, plus **Keep imported phase assignments** for trusted existing indexing. Existing primary results can still be used directly by the Steps 3–4 button. For imported results, resolve their phase-to-master association before analysis.

Tab 3 searches all enabled phases for the residual, including the primary phase. Optional advanced restrictions: same phase only, different phases only, or selected phase combinations. Do not equate “second component” with “second phase.”

Retain Run steps 2–3, Run steps 2–4, and Run steps 3–4. Before a run, display the ROI, point count, active phases, and missing/incompatible assets. Shared settings are configured once rather than repeated for each phase.

## 2. Link masters and dictionaries reliably

Introduce a stable internal phase key, independent of table order and export phase IDs. Maintain one explicit mapping among input IDs, internal keys and output IDs. Renaming/reordering phases must not change scientific identity or invalidate fits.

Each dictionary records its master content fingerprint, phase structure fingerprint, energy selection/weighting, orientation convention and crystal-axis convention, detector geometry, reference PC, pattern shape/crop/binning, orientation sampling, preprocessing actually baked into the dictionary, storage dtype and generator/schema version. Record source paths as relocation hints, not proof of identity.

Allow several dictionaries under one phase, but select only one compatible dictionary per phase for a run. Show simple states: Ready, Missing, Incompatible, or Legacy/unverified. Generate missing dictionaries with common settings; reuse compatible dictionaries without loading them all into RAM. A moved but identical master should reconnect by fingerprint. Changing a master or generation settings must never silently reuse a stale dictionary.

Older dictionary files currently lack sufficient master provenance for automatic verification. Offer explicit legacy linking with clear unverified status, or regeneration; do not declare compatibility from matching numeric phase IDs alone. Old workflows migrate to one phase entry. Save registry, bindings, fingerprints and active versions in a versioned workflow schema, retaining external large files and relocation support.

## 3. Scientific pipeline

There is no prerequisite same-phase-only overlap search. From the first multi-phase release, both primary and residual searches consider all enabled phases, and A+A, A+B and B+B are valid outcomes. Separate per-phase dictionary matching is an implementation strategy, not a restriction that both components must share a phase.

Kikuchipy provides dictionary matching, phase-specific orientation refinement and `merge_crystal_maps` for score-based selection between phase maps. The installed version 0.12.1 already includes this merge API. It chooses a winning phase/orientation per point; it does not jointly decompose two overlapping components. Reuse these facilities where appropriate, retain per-phase alternatives for overlap fitting, and extend the application's existing residual subtraction and mixture model to use different masters. The existing optimized matcher should remain usable and be checked against Kikuchipy results. [Kikuchipy merge API](https://kikuchipy.org/en/stable/reference/generated/kikuchipy.indexing.merge_crystal_maps.html).

**Primary indexing:** match every selected experimental pattern against each enabled phase dictionary. Retain the best orientation candidates *per phase*, then refine and compare them using consistent pixels, masks, geometry, resolution and objective settings. A single global top-K list can discard a weaker phase before refinement. Keep per-phase best scores and the best competing-phase score gap. The gap is an ambiguity indicator, not a calibrated probability. Provide an explicit ambiguous/unindexed outcome rather than forcing every pixel into a confident phase.

Use equivalent angular sampling and refinement budgets across phases where possible; differing symmetry means dictionary sizes need not be equal. Validate selection bias with known mixtures rather than inventing a correction based on dictionary size. Energy availability/configuration differences must be visible and validated. Initial cross-phase comparisons should use a common supported energy policy.

**Residual indexing:** subtract the selected primary simulation generated with that primary phase's master. Search the residual across enabled phases and retain phase-tagged candidates. Refinement dispatches each candidate to its own master, crystal symmetry and coordinate convention.

**Mixture optimization:** simulate the primary and secondary components from their respective masters and fit them to the original pattern using shared detector geometry. For the initial version, hold phase labels fixed within each continuous orientation refinement, but evaluate a bounded shortlist of alternative phase/orientation pairs for ambiguous pixels. Preserve the existing same-phase/two-orientation route. Phase-specific independent PC fitting should not become the default: it could absorb model mismatch into geometry.

The sequential primary-first search can miss a better joint solution. A later, separate high-accuracy mode should try alternate primary seeds, reversed extraction order, and additional phase pairs. Report the first release as a bounded candidate search, not a globally exhaustive fit.

Do not always force two components: compare against the single-component baseline and require validated fit improvement, adequate contribution and distinguishable simulations before accepting an overlap. Record weak/degenerate alternatives and optimizer failures. Near-identical orientations of the same phase are deduplicated with that phase's symmetry. Physically similar patterns from different phases should be flagged as ambiguous rather than merged by orientation alone.

Keep primary/secondary identity stable through the pipeline. A separate dominant-component map may choose the larger fitted contribution; do not silently swap all component labels when fractions cross 50%. Fractions remain **fitted pattern contributions**, not calibrated phase volume fractions, particularly across different materials.

## 4. Results and performance architecture

Current limitations found in the code: `WorkflowSession` owns one master and dictionary; residual and mixture point records lack explicit phase identity for both components; parallel payloads assume one master; IPF symmetry lookup uses input metadata with a fallback; export headers often reuse source phase metadata. Therefore this cannot safely be implemented as a UI-only change.

Add a phase registry and asset manager, then give every primary/residual candidate and fitted component an explicit phase key. Carry those keys through worker payloads, caches, materialized inspection images, score maps, exports and workflow checkpoints. Prefer explicit immutable run configuration over temporarily switching `session.master` in a loop, especially with parallel workers.

Bound total memory and worker use across phases. Process dictionaries in blocks, reuse prepared experimental data, and keep master/projection caches per asset. Batch refinement by phase or phase pair. Progress should identify phase and batch and cancellation must retain complete point results.

Use dependency-aware invalidation. Changing a master invalidates its dictionary and dependent simulations/fits. Changing the enabled search set invalidates the claim that existing winners are best across the current set, including points won by other phases; retain old results as an explicitly older analysis revision rather than silently presenting them as current. Editing a color or display name only refreshes presentation. Selecting a different ROI preserves existing results outside it with their validity masks and provenance.

## 5. Maps and inspection

Add layers to existing plots rather than multiplying tabs:

- Primary phase and residual phase maps, with a persistent categorical legend.
- Phase-pair map: A+A, A+B, B+B; use an unordered pair for the map legend while retaining ordered component identities in the data.
- Dominant phase after mixture fitting, distinct from primary phase.
- Existing IPF maps with the correct symmetry for each component's phase. Offer a phase filter and appropriate phase-specific IPF keys.
- Per-phase contribution maps: sum contributions of components belonging to that phase, including both components for A+A.
- Score-gap/ambiguity, overlap acceptance, NCC improvement and existing fit-quality layers.

Use separate appearances for not processed, unindexed, rejected and ambiguous results. Low-confidence coloring should not be mistaken for a new crystallographic phase. A selected-point panel names both phases, orientations, contributions and scores. A “Why this phase?” expansion can show the per-phase candidate comparison without cluttering the default view.

## 6. Export design

Keep the existing export controls and H5OINA/ANG choices. **Export primary ROI** writes the primary phase/orientation; **Export residual ROI** writes the secondary phase/orientation. Extend the existing **Step 4 results export** with both component phase identities and their metadata alongside its orientations, contributions and fit metrics. No extra export buttons, dominant-component export, or new result-package selector are required.

Use one consistent phase catalog and ID mapping across outputs. Never encode a pair such as A+B as a crystallographic phase. Standard component maps contain one phase/orientation per pixel; the existing Step 4 result file carries the complete overlap model. Preserve the distinction between Step 2/3 indexing estimates and Step 4 refined orientations in the existing outputs and metadata.

**H5OINA:** update the chosen component's Euler and Phase arrays and every relevant phase header, including new phases absent from the input. Preserve scan coordinates, detector geometry, PCs and source metadata. Respect crystal-axis conventions, lattice units and radians for angular fields. Phase metadata requirements are defined by the [Oxford specification](https://github.com/oinanoanalysis/h5oina/blob/master/H5OINAFile.md). Keep application metadata versioned and verify ordinary maps remain readable externally; file-format compliance alone does not establish AZtec re-import support.

Keep existing optional-pattern behavior, with explicit availability and provenance: original experimental patterns and computed residual patterns are different. Residual exports intended for another analysis pass must pair residual patterns with their secondary phase/orientation results. Do not silently treat original patterns at unavailable residual positions as valid residual data.

**ANG:** generate a complete multiphase header from the registry rather than only rewriting the original phase column. Use the [orix writer](https://orix.readthedocs.io/en/stable/tutorials/crystal_map.html#ang-format) where compatible, handling its phase-ID remapping explicitly and consistently between the two outputs. Retain the existing per-point PC sidecars. Preserve scan shape and tested invalid/outside-ROI conventions. Document any NCC-to-CI mapping as NCC, not native EDAX confidence. Validate round trips, especially new phases, lattice units and hexagonal crystal-axis conversion.

### Reloading multi-phase H5OINA files

This is required in the first multi-phase release, including files exported by this application and compatible external H5OINA files. Populate the phase registry from the selected analysis's phase metadata and preserve each pixel's phase, orientation, valid NCC, geometry and availability masks. Support three or more phases, nonconsecutive IDs and phases without pixels in the current ROI. Do not collapse imported indexing into whichever master happens to load first, or silently merge phases with identical names or symmetry.

Display imported phase/IPF/NCC maps before masters and dictionaries are loaded. Select a valid indexed inspection point when available. Link each imported phase to its own master and compatible dictionary; restore verified application-recorded bindings when files are available, otherwise show unresolved rows and allow relinking. External files need not contain application-specific provenance. Attaching the first compatible master/dictionary for a phase preserves imported indexing; replacing simulation assets or changing conditioning follows explicit invalidation rules.

Opening a primary or residual H5OINA through **Load input data…** starts a fresh analysis: the indexing corresponding to the stored patterns becomes primary indexing, with its multiple phase identities preserved. A residual export can therefore feed another overlap-analysis pass. Do not import its previous run's residual/mixture state as current results. Full-session continuation remains **Open workflow…**, which restores both components and their linked assets.

Files without pattern payloads should still support viewing their maps; simulation comparison, indexing and fitting require locating the matching pattern data first. Missing or malformed NCC should not erase otherwise valid phase/orientation data, but must not be presented as valid scored primary indexing.

Required round trips: export primary and residual H5OINA containing at least three phases (including cubic and hexagonal), reopen in a fresh session, verify phase IDs/metadata, symmetry-equivalent orientations, NCC values, masks and PC geometry, attach the matching assets, then run Steps 3–4 without repeating Step 2. Include nonconsecutive IDs, newly added phases, duplicate names, absent masters, external files without application metadata, map-only files and an unindexed first pixel. Verify crystal-axis conversions are applied exactly once across repeated exports/reloads.

## 7. Implementation sequence and acceptance gates

1. **Phase registry and migration — medium effort.** Stable IDs, master/dictionary binding, schema versions, explicit run configuration and phase-tagged results. Gate: old workflows and single-phase numerical results remain equivalent; mismatched assets are detected; moved assets reconnect.
2. **Multi-phase primary indexing — substantial.** Phase table, per-phase search/refinement, phase maps, ambiguity and imported-phase mapping. Gate: synthetic and measured known phases classify correctly, changing table order does not alter results, and missing metadata is never silently substituted.
3. **Multi-phase residual and mixture processing — largest scientific effort.** Phase-specific workers, same/different-phase candidates, bounded pair comparison, one-versus-two-component acceptance, and Steps 3–4 automation. Gate: known A+A and A+B mixtures recover both components across contribution/noise ranges; single-phase controls reject false overlaps; ambiguous pairs remain flagged.
4. **Exports and complete restoration — substantial and required before release.** Implement an export schema/round-trip spike during stage 1, then finish H5OINA/ANG writers and full-result restore here. Gate: new phases absent from the input header round-trip correctly; IDs, symmetry, units, component ordering, PC sidecars and ROI masks survive; external readers show correct phase maps.
5. **Usability and performance validation.** A two-phase example project, mixed-symmetry example, cancellation/resume, missing-file recovery and bounded-memory benchmarks. Show measured scaling with phase count; avoid promising unchanged runtime.

Candidate validation sets: FCC/BCC for distinct structures, a cubic/hexagonal case for frame/symmetry handling, same-phase orientation overlap, and a difficult similar-pattern phase pair. Use real measurements as well as synthetic mixtures; simulated recovery alone does not establish experimental phase discrimination.

Implement stages 1–2 first as a reviewable foundation, but call the feature complete only after stage 4. Defer three-or-more-component fitting, exhaustive phase-pair search and automatic crystallographic structure acquisition to subsequent work.
