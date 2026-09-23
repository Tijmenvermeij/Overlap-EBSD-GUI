# Overlap EBSD/TKD Indexing

**Version 0.2** — a GUI for resolving overlapping EBSD/TKD patterns across **multiple crystal phases**, built around [kikuchipy](https://github.com/pyxem/kikuchipy). Each phase has a linked master pattern and dictionary. A pixel can contain two fitted components from the same or different phases.

## Install and run

Requires Python 3.10+ and `tkinter`. Tested with Python 3.12 and kikuchipy 0.12.1.

```bash
python -m pip install -r requirements_gui.txt
python multistep_overlap_ebsd_gui.py
```

## Workflow

1. **Load & PC calibration:** open Oxford `.h5oina`, or EDAX `.up1`/`.up2` with its `.ang` file. Add phases and link their master patterns and dictionaries in tab 1. Refine and apply the pattern center before generating dictionaries where possible.
2. **Dictionary indexing:** select an ROI and search the enabled phases. Automatic orientation refinement is on by default.
3. **Residual indexing:** fit and subtract the primary pattern, then search the residual across the enabled phases.
4. **Mixture optimization:** fit both components, compare phase pairs, and inspect contribution maps and overlap diagnostics.

Combined buttons run steps 2–3, 2–4 or 3–4. Tabs 2–4 share the ROI and can display IPF or phase maps. The phase-map window offers full-map and ROI views.

## Settings and saving

- Defaults: **1.5° dictionary spacing**, **binning 2**, **5 retained matches per phase**, full-resolution refinement and **6 worker cores** (capped by available cores).
- Save dictionaries explicitly. Workflow `.npz` files preserve settings, calibration, results and asset links; input patterns and unsaved dictionaries are not embedded. Autosave runs after completed processing stages.
- Dictionary reuse with a different PC or detector tilt requires confirmation. Other compatibility checks remain enforced.
- Export primary/residual maps as H5OINA or ANG. Keep ANG `.pc_map.npz` companions with their files.
- Tab 4 exports full-map fitted primary or residual solutions with individually scaled patterns. At accepted overlaps, residual = measured − fitted primary, and primary = measured − that residual. Elsewhere, primary patterns are original and residual patterns are black.

See [multi-phase usage](docs/multiphase_usage.md), [calibration](docs/calibration.md) and [release notes](CHANGELOG.md).

## Scope and credits

Any number of candidate phases can be enabled; mixture fitting supports **two components per pixel**. Contributions describe fitted pattern intensity, not calibrated volume fractions. Overlap acceptance checks are provisional and need validation on representative experimental data.

Based on Cios et al., [Resolving Overlapping EBSD Patterns by Experiment-Simulation Residuals Analysis](https://arxiv.org/abs/2601.14155). Development and testing assisted by **OpenAI Codex**.
