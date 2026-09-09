# Calibration and detector geometry

Loading data preserves the imported PC map. The scan-position correction checkbox is off by default. The adjacent pixel-size input is the physical detector-plane size represented by **one stored pattern pixel**, in µm; it is independent of dictionary software binning.

PC recalibration has two explicit steps: **1. Optimize calibration points**, then **2. Apply average PC to map**. Optimization updates selected points only. A red **Apply required** indicator remains until applying succeeds, when it turns green. Other tabs show a persistent warning with a shortcut back to Apply while calibration changes are pending. Changes to PCs, calibration-point selection or active scan-correction settings require reapplication. This status is saved with the workflow. The detailed calibration report starts collapsed; compact mean/std statistics appear after completed optimization and retain that result after applying the average.

H5OINA uses the configured Oxford full-resolution pitch of **20 µm**:

| Acquisition mode | Effective pitch before export resizing |
|---|---:|
| Resolution | 20 µm/pixel |
| Sensitivity / Speed 1 | 40 µm/pixel |
| Speed 2 | 160 µm/pixel |

Recognized Speed 3/4 modes also use 160 µm/pixel, accounting for the known Speed 3 acquisition crop. Clear additional stored-pattern reductions increase the effective pitch accordingly. The GUI reports the assumption and acquisition mode; unknown modes or ambiguous resizing/cropping leave the scale unknown. UP inputs also leave it empty unless a previously saved calibration is restored. Supply an effective pitch before enabling scan correction.

The simple correction supports an aligned rectangular raster. Known nonzero scan rotation or unsupported detector geometry is rejected with an explanation. Corrected PC maps and physical calibration metadata are preserved in workflows, H5 exports and ANG `.pc_map.npz` companions; these companions are read automatically when the ANG file is loaded again. Dictionary generation uses the current representative PC; point refinement uses the current per-point PC map.

Changing orientation, PC, master pattern or conditioning invalidates affected downstream results. Recomputing residuals clears their old residual orientations/scores and mixture fits, so outdated results cannot silently remain attached to new inputs.

[Back to the README](../README.md).
