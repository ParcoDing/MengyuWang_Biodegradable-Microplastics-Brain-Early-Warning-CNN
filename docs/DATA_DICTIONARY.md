# Data dictionary

## Samples and units

Brain time points are Day 5 and Day 14. Folder examples are `day5 0CT` and `day14 500ppb`. `0CT` is the control group. Legacy `ppb` labels in filenames identify exposure groups described in the manuscript as 50, 200 and 500 µg/L; analytical measurements have not been converted.

Images are supplied processed PNGs, not original Zeiss instrument acquisitions. Each `i-j.png` is view `j` (1–4) for fish `i` within a day/dose group. The same fish number in another group does not identify the same animal. The 742 filename-defined groups must not be described as 2,968 independent fish. Exposure-tank and experimental-batch mappings are unavailable.

`data/sample_manifest.csv` records group, fish, view, zero-based feature column, image and mask path, and corresponding feature/logit file. It is reconstructed from the explicit numeric loops in the supplied feature extraction code. Positional agreement of the stored CNN logits with this ordering is an assumption pending author confirmation, not an independently observed identifier join.

## Images and masks

- `data/images/<day> <dose>/<fish>-<view>.png`: 256 × 256 processed RGB image.
- `data/segmentation/<day> <dose>/<fish>-<view>_labels.npy`: 256 × 256 integer instance labels. Zero denotes background; positive integers identify segmented objects.
- `data/segmentation/segmentation_index.csv`: source index with folder, day, dose, class index, image filename, mask path, and cell count. Class indices are source metadata and should not be confused with the binary endpoint labels of individual analysis scripts.

The source extraction script filters cells below 20 pixels. Spatial measurements are on the supplied image grid unless explicitly transformed in a script; instrument pixel sizes and per-sample expansion factors are not included.

## Feature arrays

`data/features/<day>_<dose>_arr.npy` has shape **(16, N)**, with features on axis 0 and image views on axis 1. `feature_names.npy` stores the authoritative feature order. To obtain observations × features:

```python
import numpy as np
X = np.load("data/features/day5_0CT_arr.npy", allow_pickle=False).T
names = np.load("data/features/feature_names.npy", allow_pickle=False)
```

| Position (zero-based) | Stored feature name | Interpretation |
|---:|---|---|
| 0 | cell_density | Segmented cell abundance per image area, as scaled in extraction code |
| 1 | cell_roundness | Mean cell circularity derived from area and perimeter |
| 2 | nearest_neighbor_distance | Nearest-centroid distance |
| 3 | local_density_variance | Variation in local cell density |
| 4 | channel_colocalization | Green/magenta intensity association |
| 5 | spatial_heterogeneity_increase | Spatial heterogeneity descriptor |
| 6 | green_mean_cell_area | Mean segmented cell area |
| 7 | green_mean_eccentricity | Mean segmented cell eccentricity |
| 8 | green_neighbor_degree | Mean neighbor count |
| 9 | green_close_neighbor_fraction | Fraction with nearby neighbors |
| 10 | green_dilation_connectivity | Connectivity after dilation |
| 11 | texture_glcm_contrast_std | SD of GLCM contrast |
| 12 | texture_glcm_energy_std | SD of GLCM energy |
| 13 | texture_glcm_dissimilarity_std | SD of GLCM dissimilarity |
| 14 | texture_glcm_correlation_std | SD of GLCM correlation |
| 15 | texture_glcm_homogeneity_mean | Mean GLCM homogeneity |

Exact formulas, scaling constants, thresholds and channel handling are defined in `scripts/extract_features.py`. Missing or undefined measurements can appear as NaN; downstream pipelines retain their supplied imputation/handling logic. Exploratory visualization tables may contain additional descriptors and do not define the fixed 16-feature panel.

## CNN logits and results

`results/cnn/<day>_<dose>_deep_learning_logit.npy` is a one-dimensional array with one stored raw logit per image. These are not calibrated probabilities of future injury. The supplied downstream analyses use them as continuous scores and normalize them using the corresponding controls.

CSV and XLSX tables under `results/` preserve supplied numerical values. Only day labels and path strings were harmonized. Some tables may come from different analysis revisions and do not necessarily match a fresh run of the selected scripts. In particular, a grouped fish-level analysis and a standalone image-level analysis have different sampling units. See `REPRODUCIBILITY.md`.

`docs/provenance.csv` maps every included source file to its new location, with hashes before and after preparation. `docs/excluded_files.csv` records omitted source material. `docs/checksums.sha256` covers the final package files except itself.
