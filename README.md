# Overlap EBSD/TKD Indexing

**Version 0.1** — a GUI for resolving overlapping EBSD/TKD patterns within a **single crystal phase**, using calibration, dictionary indexing, residual analysis and mixture fitting. Built around [kikuchipy](https://github.com/pyxem/kikuchipy).

## Install and run

Requires Python 3.10+ and `tkinter`. Tested with Python 3.12 and kikuchipy 0.12.1; see [release notes](CHANGELOG.md) for the tested environment.

```bash
python -m pip install -r requirements_gui.txt
python multistep_overlap_ebsd_gui.py
```

## Workflow

1. **Load and calibrate:** click **Load input data…** to select and open Oxford `.h5oina`, or EDAX `.up1`/`.up2` followed by its companion `.ang`. Click **Load master pattern…** to select and open the matching master. Imported indexing selects the first indexed point when the first map pixel is unindexed, so its simulation appears after master loading. If recalibrating, optimize calibration points and then **Apply average PC to map**.
2. **Dictionary indexing:** generate or load a dictionary, select a region of interest (ROI), and index it. Automatic orientation refinement is enabled by default.
3. **Residual indexing:** fit and subtract the primary simulated pattern, then index and refine the residual to find a second orientation of the same phase.
4. **Mixture optimization:** fit the two orientations together and inspect the NCC and contribution maps.

**Run steps 2–3** and **Run steps 2–4** automate the corresponding stages. Workflow Open/Save, shared pattern conditioning, Cancel, and worker-core controls are available across tabs. The worker limit applies to indexing, orientation/PC refinement, residual fitting and mixture fitting; it defaults to **1** and is saved with the workflow.

Tab 3 also offers **Run steps 3–4 · residual + mixture fit** to start from existing primary indexing. It computes and indexes residuals, optionally refines them, then fits mixtures using the tab 4 NCC threshold. **Save dictionary…** is directly visible in tab 2 beside the dictionary controls.

Maps update between completed batches, preserving the selected tab, inspection point and zoom. Refreshes are at least five seconds apart and become less frequent when drawing is expensive. Updates and cancellation wait for completed work; slow batches or individual optimizations can take longer.

## Defaults and results

- Dictionary spacing **1.5°**, software binning **2**, and **5 retained matches**. Primary and residual indexing share refinement settings.
- Refinement uses full-resolution patterns and a search range following the dictionary spacing; both can be adjusted. New master loads use the highest available energy.
- Steps 3 and 4 share a **Blur / gain fitting method** selector. **Blur then gain (fastest)** is the default; choose **Blur then gain + joint refinement** to further optimize both together, or **Joint blur and gain** for a joint search. The choice is saved with the workflow.
- Residuals are cached at full float32 precision in temporary disk storage, with only a few selected-point inspection images kept in memory. The cache is removed on normal shutdown; loaded workflows reconstruct images from saved fits in bounded batches, reusing the master pattern and reporting preparation progress. See [CPU fitting and performance](docs/cpu_performance.md).
- The global worker limit also controls primary/residual dictionary indexing and orientation/PC refinement. Indexing reuses dictionary normalization statistics. Primary/residual orientation refinement uses a compiled Nelder–Mead loop with Kikuchipy's projection and NCC kernels, reuses identical detector geometry and avoids fitting repeated candidates. Residual refinement builds inspection images only for the selected point. See [indexing and refinement performance](docs/indexing_refinement_performance.md).
- Speed takes priority over live previews throughout the workflow. Numerical batches fill their memory allowance; map refresh timing never shrinks them. The visible map refreshes between batches, with longer intervals when drawing is expensive (a 1% preview-overhead target). Progress and cancellation remain available at safe boundaries. First use of the compiled solver can take a few extra seconds to initialize its cached code.
- Save dictionaries explicitly to retain them beyond the session. Workflow save/restore preserves settings, calibration and results.
- Workflow autosave runs after indexing, refinement, residual generation and mixture fitting, including completed stages within automated runs. It updates the workflow path shown at the top (by default beside the input) using an atomic file replacement. Save failures are reported in the log; completed results remain in memory. Reopen the `.npz` using **Open workflow…**. Pattern data and unsaved dictionaries are not embedded.
- To index a third overlap component, open an exported residual **H5OINA with patterns** as a new input in a fresh GUI. Saved Pattern Matching NCC and orientations are loaded as already indexed **primary** data; Step 3 starts empty. Load the master and a matching dictionary, then use Step 3 to extract and index the next residual without rerunning primary DI. Valid phase/orientation/NCC values and any export ROI/availability masks determine which points are ready. Changing the master, energy model, conditioning or geometry afterwards still invalidates affected matching results.
- Export primary/residual results as H5OINA or ANG, with optional patterns. ROI exports retain the full scan dimensions and mark the ROI results. Per-point PCs are preserved; keep ANG `.pc_map.npz` companions with their ANG files.

## Geometry and scope

H5OINA geometry is imported automatically. UP + ANG starts at a **70° sample tilt**; check it for your acquisition, especially TKD. Scan-position PC correction is off by default and requires a valid physical detector-pixel scale. See [calibration and detector geometry](docs/calibration.md) for assumptions and supported geometry.

This release searches one phase at a time. Mixture coefficients are fitted pattern contributions, not calibrated volume fractions. Experimental accuracy requires validation on representative data; [release notes](CHANGELOG.md) describe the software checks and limitations.

Based on Cios et al., [Resolving Overlapping EBSD Patterns by Experiment-Simulation Residuals Analysis](https://arxiv.org/abs/2601.14155).
