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
- Build, save, load, and reuse [kikuchipy](https://github.com/pyxem/kikuchipy) dictionaries
- Select either the highest MP energy or EMsoft-style globally MC-weighted energies when loading an MP
- Run dictionary indexing and post-index orientation refinement
- Fit primary overlap residuals, index residuals, and refine residual matches
- Fit overlap-mixture models for selected points or ROIs
- Export primary or residual ROI maps as H5OINA or ANG (including the applied pattern-center map),
  with optional primary/residual patterns; UP + ANG imports can be converted directly to H5OINA
- Use a legacy projector fallback for older master-pattern formats

The GUI starts with local example file paths filled in. Replace them with your own data or browse to matching files on disk.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements_gui.txt`
- `tkinter` available in your Python installation

Install the Python dependencies with:

```bash
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

- Large EBSD datasets, generated dictionaries, residual exports, logs, and caches are ignored by git.
- Generated dictionaries use disk-backed, chunked `uint8` patterns. Generation first uses a temporary cache;
  **Save Dictionary** keeps it under the suggested master-pattern/binning/resolution filename. Existing v1
  `float32` dictionaries remain loadable and are read lazily.
- The selected master-pattern energy model is shared by dictionary generation, pattern display, refinement,
  residual analysis, and overlap-mixture fitting. Global weighting projects the EMsoft Monte Carlo energy
  histogram onto one representative detector and collapses the MP once, so subsequent simulations have the
  same cost as single-energy simulations. It requires `EMData/MCOpenCL/accum_e` in the MP HDF5 file.
  Dictionaries and workflow files store the weights and reference PC.
- `ReferenceCodes/`, `EMSphInx Studio/`, `_Depr/`, the local paper copies, and the deprecated launcher are ignored by git and are not part of the current workflow.
