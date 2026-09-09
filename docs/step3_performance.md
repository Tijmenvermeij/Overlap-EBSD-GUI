# Step 3 residual performance investigation

This report records the investigation before implementation. The resulting CPU changes and current GUI options are described in [CPU fitting and performance](cpu_performance.md).

Measured 9 September 2026. Step 3 can be accelerated on the CPU, and Metal is technically viable. The immediate priorities are eliminating repeated input-graph construction and reducing work inside the fitter. GPU work should follow these changes and a comparison of secondary-indexing results.

The supplied workflow contains 95,340 patterns of 128 × 156 pixels, uses the Cu 30 kV master, and has no pattern mask or dynamic background subtraction. Its fitting settings are 80 maximum differential-evolution generations and population multiplier 15. Hardware is an M4 Pro with 14 CPU cores, 20 GPU cores and 24 GB unified memory. The environment has Python 3.12.13, NumPy 2.4.6, SciPy 1.17.1 and Kikuchipy 0.12.1.

An 80-pattern sample on 10 workers completed residual construction in **12.988 seconds: 6.16 patterns/second**, consistent with the reported approximately five. This includes worker startup, input loading, simulation, fitting, interprocess transfer and result storage. Workflow restoration, residual indexing and disk export are excluded. Six individual fitting examples cover five primary-score quantiles (10%, 30%, 50%, 70%, 90%) plus the selected point; they are a small diagnostic sample.

The largest avoidable input overhead is in [`_signal_from_h5_selection`](../multistep_overlap_ebsd/core.py#L4137): each batch first reshapes the entire lazy 95,340-pattern array. Profiling four batches of four patterns took 1.976 seconds, including 1.349 seconds in reshape and 0.921 seconds creating 381,492 task objects. These cumulative timings overlap. Direct HDF5 reading of the same patterns took about 1.6 milliseconds per batch and produced exactly identical float32 pixels for this workflow. Profiling inflates the current-path timing, so those numbers alone do not establish the speedup.

Without profiling, the four input batches took **272–421 ms each** through the current path and **1.57–1.72 ms each** through direct HDF5 reads. All 16 patterns matched exactly. In the full 80-pattern, 10-worker run, replacing only input loading reduced elapsed time to **10.393 seconds: 7.70 patterns/second**, a **1.25× throughput improvement** with the current fitter. Worker startup remains included. These are individual runs on the same machine, not repeated trials with confidence intervals.

A production input improvement should select required rows directly or retain a reusable flattened view, preserve pattern ordering and duplicates, and apply the existing conditioning afterward. Keep Kikuchipy for signal handling and projection. This change can preserve residual results because it changes how the same experimental pixels reach the fitter.

The current [`primary fitter`](../multistep_overlap_ebsd/core.py#L1139) searches eight parameters: blur, two gain values, gain power, ellipse scales and offsets. Each objective evaluation applies Gaussian filtering, constructs the gain image, normalizes repeatedly and creates a residual image. Default settings permit 9,720 evaluations before polishing, although the measured cases stopped after 882–2,469 evaluations including polishing. Reducing only the maximum generation count therefore may do little. This budget and the default L-BFGS-B polishing behavior follow [SciPy's differential-evolution implementation](https://github.com/scipy/scipy/blob/main/scipy/optimize/_differentialevolution.py).

For normalized experimental and simulated patterns, the objective is `sum(weights) × (1 − NCC²)`. Computing it from weighted moments avoids repeated normalization and residual construction. The prototype was **1.40–1.48× faster per objective evaluation**, with maximum objective differences below 7.6 × 10⁻⁸ on the six actual patterns. Production integration must preserve zero-weight and low-variance behavior; this prototype deliberately covers ordinary nonconstant patterns.

There is also an exact reduction of the gain fit. Write the gain as `a + (b − a)q`, where `q` depends on the ellipse and power. For a fixed blur and gain shape, the centered fitted image is a linear combination of two image columns. A weighted two-column least-squares solve obtains both gain coefficients analytically. Final normalization makes overall gain magnitude unidentifiable. Eliminating it reduces the numerical search from eight to six dimensions without changing the nondegenerate residual family under the default gain bounds. Custom bounds can restrict gain directions and require a constrained solve or fallback.

The following timings sum six single-process fits; they exclude input loading, simulation and multiprocessing:

| Fitting strategy | Six-pattern time | Speedup |
|---|---:|---:|
| Current eight-variable search | 4.210 s | 1.00× |
| Six-variable search with analytic gain | 1.936 s | 2.17× |
| Fit blur, then gain with blur fixed | 0.554 s | 7.60× |
| Fit blur, then gain, then jointly refine | 1.043 s | 4.04× |

Fitting blur first saves filtering work because the gain search reuses one blurred image. A final joint refinement accommodates coupling between blur and gain. It improved fitted NCC in five of six cases; one decreased from 0.46657 to 0.46393. Residual NCC against the current solver ranged 0.99122–0.99925, but RMS differences were **3.9–13.2% of the current residual RMS**. Similar fit scores therefore do not guarantee equivalent secondary information. A controlled example with known blur/gain and 40% secondary amplitude gave a 7.53× fitter speedup with joint refinement and a slightly better objective. These are exploratory results, not validated improvements to secondary indexing.

Memory should also be bounded. Each point currently retains six float32 image planes. At this detector size, retaining them for the whole scan would require **45.7 GB**, before other data. Return residuals plus compact fit metadata, reconstruct inspection images on selection, and use bounded caching or streamed storage. Disk-writing mode currently strips inspection images only after all six have crossed the process boundary.

Metal can run the image operations through [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) or MLX, whose arrays support [shared CPU/GPU memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) and [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). Neither backend is installed here; no GPU speedup was measured. The current NumPy/SciPy implementation does not automatically use Metal. A useful implementation would batch candidate populations and several patterns through one persistent GPU engine, retaining arrays and returning score vectors. Small scalar calls can repeatedly force execution, as described in [MLX's evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html). Preserve Gaussian boundary handling, truncation, masks and normalization, and time completed GPU work.

Recommended implementation order: fix input selection and bound result memory; simplify the CPU objective; validate analytic gain fitting with staged initialization and joint refinement; then benchmark batched Metal against that improved CPU baseline. Preserve the current solver for comparison while validating residual indexing on representative overlaps.

Reproduction uses [`step3_prepare.py`](../benchmarks/step3_prepare.py), [`step3_objective.py`](../benchmarks/step3_objective.py), [`step3_reduced.py`](../benchmarks/step3_reduced.py) and [`step3_pipeline.py`](../benchmarks/step3_pipeline.py). From the repository root, replace the workflow placeholder:

```sh
MPLCONFIGDIR=.mplconfig .venv/bin/python benchmarks/step3_prepare.py "/path/to/saved_workflow.npz"
MPLCONFIGDIR=.mplconfig .venv/bin/python benchmarks/step3_objective.py /tmp/step3_workflow_pairs.npz
MPLCONFIGDIR=.mplconfig .venv/bin/python benchmarks/step3_reduced.py --pairs /tmp/step3_workflow_pairs.npz --limit 6
MPLCONFIGDIR=.mplconfig .venv/bin/python benchmarks/step3_pipeline.py "/path/to/saved_workflow.npz" --cores 10 --count 80
MPLCONFIGDIR=.mplconfig .venv/bin/python benchmarks/step3_pipeline.py "/path/to/saved_workflow.npz" --cores 10 --count 80 --direct-h5-dataset "1/EBSD/Data/Processed Patterns" --output /tmp/step3_pipeline_direct_results.json
```

The direct-input command is specific to the verified dataset path in this workflow and requires dynamic background subtraction to be disabled. Raw measurements, fitted parameters, residual comparisons and profiles are retained in [`benchmarks/results/step3_20260909`](../benchmarks/results/step3_20260909/). The application reference was commit `301ba4d`.

These scripts perform investigation only. No production fitter, saved workflow, source data, dictionary, residual export or application settings were changed.
