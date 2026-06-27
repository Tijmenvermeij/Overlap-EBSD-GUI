# Overlap EBSD Indexing

This repository implements the multi-step overlap-EBSD workflow described by Grzegorz Cios, Aimo Winkelmann, Tomasz Tokarski, Wiktor Bednarczyk, and Piotr Bała in the article [Resolving Overlapping EBSD Patterns by Experiment-Simulation Residuals Analysis](https://arxiv.org/abs/2601.14155).

The paper's core idea is preserved here: fit the simulated pattern with a blur and gain model, normalize it, subtract the NCC-scaled simulation from the measured pattern, and use the residual for follow-up indexing, refinement, and overlap-mixture analysis.

The example code referenced by the paper is available on Zenodo as [Cu_residuals.py](https://zenodo.org/api/records/17079414/files/Cu_residuals.py/content), within the Zenodo record [10.5281/zenodo.17079414](https://zenodo.org/records/17079414).

## Current Workflow

The GUI is launched from `multistep_overlap_ebsd_gui.py` and is organized into four stages:

1. Load and PC Calibration
2. Dictionary Indexing
3. Overlap Indexing
4. Overlap Optimization

In practice, the application can:

- Load Oxford `.h5oina` data or EDAX `.up1` / `.up2` patterns with a companion `.ang` file
- Calibrate or edit pattern centers
- Build, save, load, and reuse Kikuchipy dictionaries
- Run dictionary indexing and post-index orientation refinement
- Fit primary overlap residuals, index residuals, and refine residual matches
- Fit overlap-mixture models for selected points or ROIs
- Export reindexed maps, primary or residual ROI maps, workflow state, and optional residual patterns
- Use Kikuchipy where possible, with a legacy projector fallback for older master-pattern formats

The GUI starts with local example file paths filled in. Replace them with your own data or browse to matching files on disk.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements_gui.txt`
- `tkinter` available in your Python installation

Install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_gui.txt
```

## Run

Launch the GUI with:

```bash
python multistep_overlap_ebsd_gui.py
```

## Repository Layout

- `multistep_overlap_ebsd/` main package code
- `multistep_overlap_ebsd_gui.py` GUI launcher
- `requirements_gui.txt` runtime dependencies

## Notes

- Large EBSD datasets, generated dictionaries, residual exports, logs, caches, and local virtual environments are ignored by git.
- `ReferenceCodes/`, `EMSphInx Studio/`, `_Depr/`, the local paper copies, and the deprecated launcher are ignored by git and are not part of the current workflow.
