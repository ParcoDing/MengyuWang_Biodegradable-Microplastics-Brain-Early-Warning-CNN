# Biodegradable microplastics and early brain image phenotypes

Data and analysis scripts accompanying **Deep learning reveals hidden early brain phenotypes induced by biodegradable microplastics**, by Mengyu Wang and Wen-Xiong Wang (manuscript prepared for submission; no publication DOI assigned in this package).

The study combines expansion microscopy, predefined image features, and stored convolutional neural network (CNN) scores to study the developing zebrafish brain after exposure to aged polylactic acid (PLA) microplastics at 0, 50, 200, and 500 µg/L. Endpoint groups (0 and 500 µg/L) define the main models; 50 and 200 µg/L are held out for intermediate-dose evaluation.

**Scope:** this package contains processed 256 × 256 images, supplied segmentation masks, 16-feature arrays, stored CNN logits, analysis tables, and downstream analysis scripts. It supports analysis from the supplied features and logits. It does **not** contain a complete CNN training/inference implementation, CNN weights, or original instrument acquisitions. See [reproducibility and outstanding items](docs/REPRODUCIBILITY.md) before treating this as a complete study archive.

## Time points and observations

All brain-image data use **Day 5** and **Day 14**. Source labels `day10`, `10天`, and `Day 10` in this brain-image archive were naming errors; they were corrected following the author's explicit confirmation on 23 October 2026. Numerical measurements and image/mask pixels were preserved. The manuscript's separate whole-body fluorescence experiment also includes an actual Day 10 time point; those acquisitions are not provided here.

| Brain time point | Exposure (µg/L) | Image views | Fish groups inferred from filenames |
|---|---:|---:|---:|
| Day 5 | 0 | 336 | 84 |
| Day 5 | 50 | 336 | 84 |
| Day 5 | 200 | 356 | 89 |
| Day 5 | 500 | 364 | 91 |
| Day 14 | 0 | 404 | 101 |
| Day 14 | 50 | 508 | 127 |
| Day 14 | 200 | 368 | 92 |
| Day 14 | 500 | 296 | 74 |
| Total | | 2,968 | 742 |

Each filename `fish-view.png` has four views per fish, with matching `fish-view_labels.npy` segmentation masks. Fish identifiers are local to each day/dose group. They are not longitudinal identifiers or exposure-tank identifiers. The supplied extraction script establishes this ordering; CNN logit ordering is assumed to follow the matching feature-array columns and still requires author confirmation. See [data documentation](docs/DATA_DICTIONARY.md) and [sample_manifest.csv](data/sample_manifest.csv).

## Repository layout

```text
data/
  images/                 Processed input images, grouped by day and dose
  segmentation/           Supplied integer instance-label masks and source index
  features/               Eight 16 × N arrays and feature_names.npy
  sample_manifest.csv     Image, mask, feature-column and group mapping
results/
  cnn/                    Stored CNN logits and downstream score tables
  composite/              Supplied composite-score results
  logistic_regression/    Supplied feature-based logistic-regression results
  nonlinear/              Supplementary XGBoost results
  interpretation/         Feature-family and CNN reconstruction tables
  cross_time/             Cross-time transfer tables
  figures/                Figure source tables; reruns write plots here
  feature_visualization/  Additional exploratory tables
scripts/                  Selected source scripts with portable paths
docs/                     Reproduction guide, dictionary and provenance
```

Redundant script revisions, nested archives, and pre-rendered source plots were excluded. Re-running the selected plotting scripts produces figures with corrected time labels. Historical validation exports with contradictory split counts are also excluded; details are in `docs/REPRODUCIBILITY.md` and `docs/excluded_files.csv`.

## Installation

Use Python 3.11 or newer in an isolated environment. Original author package versions were not supplied; `requirements.txt` lists direct dependencies rather than claiming an original environment lock.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_package.py
```

For image-based feature extraction and microscopy plots, additionally install `requirements-imaging.txt`. For supplementary nonlinear models, install `requirements-optional.txt`. Supplied segmentation masks can be used without Cellpose; the original segmentation-generation code and model configuration are missing.

## Run the analyses

**Run commands from the repository root.** Scripts write to `results/` and may replace supplied tables; work on a separate copy to compare reruns with archived outputs. Default real-data modes should be retained. Mock-data options in source plotting scripts are for layout demonstrations only.

```bash
python scripts/figure06_feature_coordination.py
python scripts/figure07_unseen_doses.py
python scripts/figure09_cross_time.py
python scripts/figure08_interpretation.py
```

The interpretation script defaults to the source's `RUN_MODE = "fast"` (300 bootstrap replicates, 200 cross-validation trees, 400 importance trees). For its heavier source publication configuration, change it to `"publication"` (2,000, 600, and 800 respectively). Exact agreement with manuscript values is not established by packaging.

During preparation, the Figure 6, 7 and 9 scripts were run successfully in a separate working copy, including their plot and XLSX exports. The tested environment and numerical comparisons are recorded in `docs/validation_report.json`. Other analysis scripts were checked for Python syntax but were not executed end to end. `requirements-tested.txt` records direct package versions for these three smoke runs; these are preparation-time versions, not the authors' original environment.

```bash
python -m pip install -r requirements-imaging.txt
python scripts/figure05_microscopy.py
# Optional: recompute features from supplied images and masks; replaces feature arrays.
python scripts/extract_features.py
```

The `day5_*` and `day14_*` scripts preserve supplementary standalone composite, logistic-regression, and XGBoost analyses. Their results may differ from the main grouped analyses; see `docs/REPRODUCIBILITY.md`. No command in this package retrains the CNN.

## Manuscript figure correspondence

| Manuscript content | Script or available material |
|---|---|
| Figures 1–4, conceptual/material/whole-body/representative imaging content | Complete original inputs and assembly workflows not included |
| Figure 5, microscopy feature illustration | `scripts/figure05_microscopy.py` |
| Figure 6, feature coordination and composite phenotype | `scripts/figure06_feature_coordination.py` |
| Figure 7, held-out intermediate concentrations | `scripts/figure07_unseen_doses.py`, supplied CNN logits |
| Figure 8, feature-family interpretation of CNN scores | `scripts/figure08_interpretation.py` |
| Figure 9, cross-time transfer | `scripts/figure09_cross_time.py` |

## Availability, citation and reuse

Intended repository: [ParcoDing/MengyuWang_Biodegradable-Microplastics-Brain-Early-Warning-CNN](https://github.com/ParcoDing/MengyuWang_Biodegradable-Microplastics-Brain-Early-Warning-CNN).

Use `CITATION.cff` to cite the repository together with the manuscript. No publication status, DOI, or archival accession is implied. Correspondence: Wen-Xiong Wang, wx.wang@cityu.edu.hk (as listed in the supplied manuscript).

The authors have not yet specified code and data licenses. No reuse license is granted by this preparation step; see `LICENSE_STATUS.md`. Before a final publication release, resolve the missing materials, choose appropriate licenses, and archive the final version with a persistent identifier if desired. See the [upload guide](docs/UPLOAD.md).
