# Dictionary indexing and refinement: CPU evaluation

Evaluated and implemented on 9–14 September 2026 using the supplied Cu reference workflow on the M4 Pro. The original investigation below used application commit `f315c75`; the implementation and its validation are described first.

## Implemented behavior and validation

### Primary refinement and speed-first refresh policy, 14 September

Primary and residual refinement now use their full memory-bounded batches from the start. The five-second batch-size controller has been removed, including for calibration. Full-resolution 128 × 156 patterns with five retained candidates permit **672 map points / 3,360 candidate fits** per batch under the existing 256 MiB candidate-pattern allowance. Binning 2 permits the existing maximum of 1,024 map points. Dask splits these batches across the selected worker cores; the displayed batch size is not the per-core workload. Cancellation is checked before/after outer refinement batches; a running batch completes before cancellation takes effect.

All steps share a preview policy that measures the coordinator's pause for drawing and queueing, then waits at least 99 times that duration before another preview (1% overhead target, with a five-second minimum interval). Numerical batches never change size to satisfy this interval. Only the visible tab is redrawn, once per preview, and combined workflows no longer force extra redraws at every stage transition. The final result is still drawn when a job ends. This is a measured-cost target, not a hard bound on the first redraw or an unexpectedly expensive later redraw.

Two numerical overheads were removed. Kikuchipy regards an array of identical PCs as varying geometry, so orientation-only fits now collapse identical PCs to one shared detector geometry. The Nelder–Mead iteration loop now runs in compiled Numba code with the Python lock released, using Kikuchipy's existing rotation, projection and NCC kernels via its public custom-SciPy-method API. The simplex arithmetic preserves SciPy's non-adaptive update rules, bounds, stopping tolerances and partial-evaluation-budget behavior, without enabling fastmath in that loop. All distinct candidates, detector pixels, trust regions and evaluation limits are retained. PC-only and joint orientation/PC refinement retain SciPy. Unsupported solver options also fall back to SciPy. The adapter uses internal Kikuchipy kernel imports; if those imports move, the application keeps the public SciPy path. SciPy's license is retained in [third_party/SCIPY_LICENSE.txt](../third_party/SCIPY_LICENSE.txt).

Paired primary measurements on 1,536 freshly indexed points, **six worker cores**, five candidates, 1.4° trust region and `maxfev=25`:

| Configuration | Full 128 × 156 | Binned 64 × 78 (mean of two runs) |
|---|---:|---:|
| Before this follow-up, with earlier CPU improvements | 24.61 s | 11.38 s |
| Shared detector geometry + full batches, SciPy solver | 23.60 s | 11.21 s |
| Shared geometry + full batches + compiled solver | **18.62 s** | **4.52 s** |

This is **1.32× full-resolution throughput** and **2.52× binned throughput**. The binned comparison uses the same binned input in both versions; the improvement does not come from changing resolution. Euler angles and scores were exactly equal in every paired run. Average process CPU usage rose from 5.24 to 5.56 cores for full resolution and from about 3.32 to 4.95 cores for binned refinement. Full-resolution projection remains the dominant computation, so it benefits less from removing Python optimizer overhead.

The timings include input preparation and progress callbacks, but exclude loading, prior dictionary matching, solver warm-up and actual Tk redraw time. Initial compiled-solver setup was measured separately at 4.50 seconds for two full-resolution points; code is cached for subsequent use. These are sample measurements on a machine whose load can change, not full-map guarantees. Reproduce with [primary_refinement.py](../benchmarks/primary_refinement.py), providing a `core.py` snapshot from before this follow-up via `--reference-source`; `--binned` selects binned refinement and `--residual` exercises Step 3. Raw results: [full resolution](../benchmarks/results/refinement_20260914/primary_1536.json), [binned](../benchmarks/results/refinement_20260914/primary_binned_1536.json).

The same adapter was also checked on **672 freshly indexed residuals**, full resolution, six cores and the same five-candidate/25-evaluation settings. Refinement improved from **11.26 to 7.51 seconds (1.50× throughput)** with exactly equal Euler angles and scores; the geometry/batching-only variant took 10.20 seconds. All 672 points ran as one memory-bounded batch. See [residual measurements](../benchmarks/results/refinement_20260914/residual_compiled_672.json).

Numerical tests compare full simplex states, evaluation counts and iteration counts against SciPy over 1,512 bounded/unbounded searches, including ties, contraction, shrink, convergence and exhausted budgets. Public Kikuchipy comparisons cover fixed/varying PCs, masks, uint8/float32 inputs, evaluation/iteration limits and unsupported-option fallbacks. GUI tests verify reduced refresh frequency, visible-tab-only drawing and stage transitions without forced rendering; cancellation checks verify that a completed 672-point batch is retained and untouched points keep their previous state. The full test log is [primary_tests.txt](../benchmarks/results/refinement_20260914/primary_tests.txt).

### Interleaving residual indexing and refinement

The current session already shares a lossless float32 residual cache between indexing and refinement. After loading a workflow, indexing reconstructs missing residuals once and stores them in a temporary HDF5 file. Refinement reads that cache; it does not repeat the primary simulation or blur/gain fit. The cache can hold all computed points on disk: 90,000 residuals at 128 × 156 pixels occupy about 7.2 GB before HDF5 overhead. RAM holds working batches and a bounded set of inspection images. The file is deleted on normal session closure.

Measured before the compiled-solver follow-up on 3,000 restored residuals with ten cores, five dictionary candidates, dictionary binning 2, 1.4° trust region and `maxfev=25`:

| Refinement resolution | Indexing + refinement | Refinement input preparation | Optimistic saving from reusing that input |
|---|---:|---:|---:|
| Full 128 × 156 | 54.11 s | 0.30 s | 0.56% |
| Binned 64 × 78 | 39.17 s | 0.41 s | 1.06% |

Both runs recorded zero residual reconstruction/projection calls during refinement. Input preparation includes the cache read, signal construction and materialization/downsampling. Its entire duration is an optimistic ceiling on the saving from retaining input between the two stages; a combined implementation would still need some of that preparation. These instrumented single-run measurements exclude workflow loading and GUI redraw time and are not timings of an implemented combined pipeline.

This input-reuse saving is too small to justify changing the pipeline for throughput. The stages remain separate, and the independent residual-refinement action remains available. Interleaving could deliver refined results earlier and, with a separate cache-eviction policy, reduce temporary disk use. Interleaving alone would not stop the persistent session cache from growing; evicted residuals would need reconstruction if the user repeats refinement or inspects them later.

Benchmark: [residual_pipeline_reuse.py](../benchmarks/residual_pipeline_reuse.py). Results: [full resolution](../benchmarks/results/refinement_20260914/pipeline_reuse_full_3000.json), [binned refinement](../benchmarks/results/refinement_20260914/pipeline_reuse_binned_3000.json).

### Residual refinement batches, 14 September follow-up

Profiling 320 residuals after dictionary indexing with five retained candidates showed that only 3.10 of 11.86 seconds were spent fitting orientations. Another 8.61 seconds went into serially projecting 320 inspection images, most of which the ROI result store immediately discarded. The adaptive batch controller included this overhead in its timings, which kept batches smaller as well.

ROI residual refinement now generates a secondary inspection image only for the selected point. Other images are reconstructed on selection through the existing inspection cache. Refined orientations, candidate scores, full-resolution residuals and all fitting settings are unchanged. This earlier fix started both primary and residual refinement with core/candidate-aware batches: 128 points for ten cores and five candidates, instead of 16. The later speed-first policy above supersedes that adaptive outer batching; the inspection-image optimization remains active.

Paired measurements on the supplied workflow, 320 newly dictionary-indexed residuals, full resolution, five candidates, ten cores, 1.4° trust region and `maxfev=25`:

| Configuration | Batch sizes | Time |
|---|---|---:|
| Before this fix, including earlier CPU improvements | 16, 64, 140, 100 | 11.83 s |
| Inspection-image fix with original batch policy | 16, 64, 240 | 2.93 s |
| Inspection-image fix with larger starting batch | 128, 192 | 2.92 s |
| Forcing one maximum-size batch | 320 | 3.11 s |

The largest saving came from avoiding unused projections. Simply maximizing batch size was slightly slower in this small sample; the subsequent larger primary benchmark and the user's throughput priority motivated removing refresh-driven adaptation everywhere. The final orientations, NCC scores, selected secondary image and selected residual were exactly equal across all four configurations. Times include cached residual reads, refinement, result storage, progress callbacks and selected-point inspection; workflow loading and prior dictionary indexing are excluded. These are single-run samples, not full-map throughput guarantees.

All 163 tests passed, including selected-image handling for one/multiple candidates, best-candidate selection, batch adaptation under slow fits, memory caps and completed-result retention on cancellation. See [residual_refinement.py](../benchmarks/residual_refinement.py) for the benchmark, which accepts a reference `core.py` captured before the fix, and [raw measurements](../benchmarks/results/refinement_20260914/residual_refinement_320.json).

### Residual preparation after loading a workflow

The larger indexing batches exposed another bottleneck in Step 3. Saved workflows retain fit parameters rather than full residual images. Without a residual image cache or verified exported pattern source, preparation reconstructed every point independently, creating a master-projection graph and loading the selected master plane repeatedly. It also rebuilt unused inspection images, including secondary simulations when a secondary orientation was present. A 64-point profile attributed almost all preparation time to this repeated projection setup.

Reconstruction now reads and projects bounded batches using Kikuchipy, reuses the selected master plane, and applies the saved blur/gain/scale only to produce the residual. Existing in-memory and disk-cached rows are retained when other rows need reconstruction. Full-resolution residuals are cached as before; preparation reports completed-point counts and checks cancellation between reconstruction batches. A separate 64 MiB estimate for projection working arrays limits each reconstruction batch to at most 128 points (93 for these patterns), independently of the larger dictionary matching batch. Fixed-PC projection uses the shared CPU limit. Varying-PC projection retains one navigation chunk per bounded call because Kikuchipy's varying-PC graph fails with multiple navigation chunks; individual PCs are preserved.

Measured on the restored 95,340-position Cu workflow, 128 × 156 input patterns, software binning 2 and ten workers:

| Operation, 3,000 points | Time |
|---|---:|
| Original residual reconstruction and binning | 82.45 s |
| Batched reconstruction and binning | 2.10 s |
| Preparing those residuals again from the session cache | 0.31 s |
| Dictionary matching, first pass | 7.91 s |
| Primary input preparation and binning | 0.11 s |

All pixels in the 3,000 prepared residual patterns were exactly equal. The first reconstruction run includes selected-master preparation and residual-cache writes; workflow loading and a common initial projection/JIT warm-up are excluded. These are single-run measurements, not a full-map throughput guarantee. Tests also compare batched projections against single-point projections for fixed/varying PCs, both input coordinate frames, selected and weighted energies, and float64 master intensity rescaling. Input tests cover float32 background removal, masks, saved gain/blur parameters, mixed caches, duplicate selections and cancellation/resumption.

All 161 tests passed after this fix, alongside compilation and whitespace checks. Reproduce the benchmark with `MPLCONFIGDIR=.mplconfig .venv/bin/python benchmarks/residual_preparation.py WORKFLOW --count 3000 --cores 10 --match --output REPORT.json`. Raw measurements: [residual_preparation_3000.json](../benchmarks/results/indexing_20260909/residual_preparation_3000.json).

### Dictionary matching and refinement

The global worker setting now covers primary and residual dictionary indexing, orientation refinement and calibration's orientation/PC refinements, as well as the existing residual/mixture fits. Settings are captured on the GUI thread before work starts. Dask worker limits are scoped to the job and restored on completion or cancellation; native math-library threading remains backend dependent.

Indexing uses aligned dictionary blocks and substantial memory-capped experimental batches. Experimental patterns are prepared once per batch. Exact float32 dictionary means and norms are reused across batches and primary/residual indexing. This cache uses nine bytes per dictionary pattern (2.74 MB for this dictionary), and is replaced on dictionary/mask changes or cleared with the session. It does not allocate a complete 6 GB normalized dictionary. Score/partition arrays and normalization blocks have separate memory bounds. Candidate selection is performed once; exact ties are ordered by ascending dictionary index. Undefined NCC values, including constant experimental patterns, are reported as NaN; invalid dictionary patterns do not outrank valid matches.

Progress and cancellation are checked inside the dictionary scan. Completed batches are committed before their completion callback; cancellation inside a scan leaves the current batch's orientations and scores untouched. The GUI continues to refresh completed results. Dictionary blocks and completed refinement batches are interruptible boundaries; a running optimizer is not interrupted midway.

Refinement sizes navigation chunks from actual candidate-fit counts and available workers. Repeated seeds for one map point are fitted once, while every distinct seed is retained and results are expanded back into the existing candidate layout. Kikuchipy's two-row workaround is retained when a job reduces to just one unique fit, because Orix and HyperSpy disagree on the shape of a single-point map. The selected master-energy plane is cached, including masters already collapsed with global energy weights. Scientific settings, masks, PCs and optimizer evaluation limits are preserved.

Paired measurements from the integrated implementation, with ten workers and the same inputs per comparison:

| Workload | Reference `f315c75` | Implemented |
|---|---:|---:|
| 512-pattern dictionary batch, first pass | 7.33 s | 4.01 s |
| Same 512-pattern batch, reusable statistics available | 7.33 s | 2.79 s |
| Actual GUI-style indexing of 24 points | 12.41 s | 1.81 s with statistics already cached |
| Repeat refinement of 80 points with cleared candidate slots | 1.59 s | 0.37 s |

The compared candidate Euler values, NCC scores and refined orientations were exactly equal. The small GUI case retains progress callbacks in both versions. Timings exclude workflow loading; dictionary-batch comparisons also exclude experimental input preparation. Repeat refinement includes input preparation and GUI-style progress callbacks. These are single-run sample measurements, not a full-map guarantee; machine load and sampled orientations explain differences from the earlier investigation below.

The actual workflow also completed residual construction, residual indexing/refinement, selected residual refinement, mixture fitting and calibration. All 156 automated tests passed, including checks for masks, lazy dictionaries, thread limits, cache reuse/invalidation, single-point handling, exact ties, degenerate patterns and cancellation before/after a batch commit. Reproduce with [indexing_cpu_validation.py](../benchmarks/indexing_cpu_validation.py); results are in [implemented_validation.json](../benchmarks/results/indexing_20260909/implemented_validation.json).

The following sections retain the investigation and rationale. Persistent process refinement and Metal remain future options; the implemented scheduler uses shared-memory CPU threads.

There are worthwhile improvements for small selections, medium ROIs and full maps. Refinement already uses some parallel processing, while dictionary indexing is chiefly limited by repeated dictionary preparation and small experimental batches. Simply applying the Step 3 process pool to both operations is unlikely to be the best first change.

## Baseline execution at `f315c75`

The GUI core-count setting is forwarded to residual and mixture fitting, but not to primary/residual dictionary indexing or orientation refinement. These operations use Kikuchipy and Dask's default threaded scheduler independently. Installed NumPy 2.4.6 uses Apple Accelerate BLAS; `threadpoolctl` does not report or control this backend in this environment. Consequently, a one-worker Dask benchmark is **not** a guarantee of one CPU core. The benchmarks also record process CPU time divided by wall time.

Primary and residual orientation refinement pass `rechunk=True`. Installed Kikuchipy 0.12.1 normally divides an eager input into chunks of 64 candidate fits, or retains existing larger chunks. The GUI initially submits 16 map points: with five candidates, that is 80 fits and usually only two concurrent tasks. Without candidates it can be a single task. Calibration's orientation+PC and subsequent PC-only calls explicitly use `rechunk=False`; typical small/medium calibration selections therefore remain one task.

Kikuchipy provides public chunk controls for this purpose; Dask documents both threaded scheduling and the need to coordinate inner BLAS threads with outer tasks. See [refinement API](https://kikuchipy.org/en/stable/reference/generated/kikuchipy.signals.EBSD.refine_orientation.html), [Dask scheduling](https://docs.dask.org/en/stable/scheduling.html), and [Dask array best practices](https://docs.dask.org/en/stable/array-best-practices.html). Version-specific behavior above was checked against the installed source, rather than inferred from newer online documentation.

## Dictionary indexing measurements

The dictionary contains 304,053 orientations at 64 × 78 pixels (software binning 2). All tests retain five candidates. Timings include dictionary reads, normalization, matching and candidate selection; workflow restore and experimental input preparation are excluded. Experimental arrays are materialized for these controlled comparisons. These are single runs, not full-map throughput guarantees.

| Experimental patterns in one batch | Current, 1 Dask worker | Current, 10 Dask workers | Streamed NCC prototype |
|---:|---:|---:|---:|
| 16 | 12.29 s | 11.13 s | 4.37 s |
| 128 | 11.86 s | 10.29 s | 4.82 s |
| 512 | 15.98 s | 12.11 s | 6.62 s |

All five candidate indices, their ordering and their scores matched exactly across these comparisons. Equality on this sample does not replace future regression coverage for masks, ties, degenerate patterns and other input formats.

The prototype retains Kikuchipy's NCC normalization and NumPy matrix matching, reads aligned 8,192-pattern dictionary blocks, prepares experimental patterns once and computes candidate indices once before gathering their scores. It keeps one prepared block and a bounded score matrix in memory, without a full normalized dictionary cache. At 512 patterns its time was approximately 0.78 s reading, 3.45 s normalizing, 0.82 s matching and 1.57 s selecting candidates. Current 10-worker indexing averaged only 1.42 CPU cores for this batch, compared with 1.03 for the faster prototype. Reduced work is the primary gain demonstrated here.

Two app-level issues amplify the difference on maps:

* Each experimental batch rescans and normalizes the complete dictionary. The same applies to primary and residual indexing.
* `progress_batches()` starts at 16 patterns and aims for five seconds. When the dictionary scan alone exceeds five seconds, smaller batches cannot meet the target: the controller can shrink toward a single point, repeatedly paying the same fixed cost. Progress/cancellation must be separated from the size of the matrix batch.

An additional test exercised the actual application method on the same 24 points with ten Dask workers. With the GUI-style progress callback it took **15.88 s**, submitting batches of 16 and 8 points. Without the callback it processed one batch in **8.01 s**. This isolates the current batching penalty; the proposed fix must retain GUI progress and cancellation while avoiding unnecessary full dictionary passes. The callback test does not include Tk redraw time.

Prepared dictionary reuse has potential, but needs a memory policy. The source uint8 dictionary is 1.52 GB; fully normalized float32 data is 6.07 GB. Only about 6 GB was available during this investigation, so a full resident normalized cache was not tested. Prefer reuse of each prepared dictionary block across experimental tiles, or an optional complete cache when comfortably within the memory budget. A partial LRU cache smaller than the dictionary would thrash during sequential rescans. Cached per-pattern normalization statistics are another small-memory option to benchmark.

Ten independent dictionary-indexing processes would duplicate preparation and increase memory/read traffic. Shared blocks, aligned reads, sufficiently large matrix batches and controlled compute concurrency are better starting points. HDF5 reads themselves cannot be made parallel just by adding h5py threads; its calls are serialized by a library lock. See [h5py threading](https://docs.h5py.org/en/stable/threads.html) and the [Kikuchipy dictionary-indexing API](https://kikuchipy.org/en/stable/reference/generated/kikuchipy.signals.EBSD.dictionary_indexing.html).

## Refinement measurements

Full 128 × 156 patterns, saved trust settings, SciPy Nelder–Mead and `maxfev=25`. The table includes graph construction and fitting, with inputs and the selected master energy already loaded. Each point in these rows has one candidate. Five distinct candidates mean five fits per map point and must still be searched if the same scientific behavior is required.

| Workload | 1 worker | 10 workers, default chunks | 10 workers, chunks of 4 fits |
|---|---:|---:|---:|
| 8 orientation fits | 0.162 s | 0.185 s | 0.115 s |
| 80 orientation fits | 1.397 s | 1.231 s | 0.745 s |
| 640 orientation fits | 10.998 s | 5.405 s | 6.298 s |
| 80 PC-only fits | 3.233 s | — | 1.420 s |
| 80 orientation+PC fits | 3.037 s | — | 1.532 s |

Results were bit-for-bit identical. Small chunks improve medium workloads, but become counterproductive when enough work already exists. Choose chunks from the candidate count and worker count, with enough work per task, rather than hard-coding four fits. The observed speedup falls short of tenfold. Python optimizer/callback overhead and scheduling are likely contributors, although the Numba projection and NCC kernels release the GIL; their separate contributions were not profiled here.

There is an additional exact-work reduction: this saved workflow retains a `(95340, 5, 3)` primary candidate array whose entries are all NaN. After refinement, candidate rows are cleared but their storage remains allocated. Refining again sees the five-column shape and replaces every missing seed with the current orientation, running five identical fits per point. The same fallback pattern exists for residual candidates. In an 80-point experiment, this took 4.623 s; one fit per point with appropriate chunks took 0.745 s. All duplicated results were identical. Deduplicate missing/repeated seeds per point while retaining every distinct valid candidate.

Materializing and reusing the selected master energy is inexpensive here: approximately 8 MB and 0.040 s. It avoids repeated lazy reads and is useful for both orientation and PC refinement. In a separate 640-fit comparison, threads took 6.14 s, ten fresh processes 9.71 s, and a reused process pool 4.66 s. These are generic Dask processes, with serialization overhead; a dedicated persistent pool might improve further but was not implemented. Threads are the simpler initial option. Processes merit a separate large-workload mode only after accounting for startup, memory and transfer costs.

## Original implementation priorities

1. Fix duplicate/missing refinement candidates and cache the selected master energy. These remove redundant work without changing the intended optimization.
2. Extend the GUI CPU setting to primary/residual orientation and calibration refinement. Use an explicit worker limit and adaptive navigation chunk sizes; keep small jobs inexpensive.
3. Fix dictionary batching independently of the GUI refresh interval. Report progress and check cancellation within dictionary iterations, while committing only complete results for a point/batch.
4. Integrate aligned dictionary reads, prepared experimental reuse and one candidate selection pass. Reuse dictionary preparation across substantial experimental tiles with a conservative memory budget.
5. Benchmark sustained process refinement and controlled BLAS/Dask concurrency before choosing any large-map default.

For a few selected points, dictionary setup/cache and duplicate elimination matter most; extra workers have little work to share. Tens to hundreds of points already benefit from batching and suitable refinement chunks. Full maps gain most from sustained dictionary reuse and balanced workers. There is no universal map-size threshold: dictionary size, pixel count, retained candidates and optimizer evaluations determine the workload.

More aggressive options—coarser dictionaries, more binning, refining candidates at low resolution before a final full-resolution pass, or fewer candidates—could save additional time but can change orientation selection. They need accuracy comparisons and should be separate choices, not silently enabled CPU optimizations. GPU/Metal matrix matching remains feasible future work; this evaluation requires no GPU backend.

Reproduction scripts and raw results: [dictionary_parallel.py](../benchmarks/dictionary_parallel.py), [refinement_parallel.py](../benchmarks/refinement_parallel.py), and [indexing_20260909](../benchmarks/results/indexing_20260909/). Input patterns, dictionary and the saved source workflow were read without modification.
