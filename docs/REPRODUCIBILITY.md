# Reproducibility and outstanding items

## What is available

The supplied archive contains 2,968 processed image views, their segmentation masks, eight fixed-panel feature arrays, eight CNN logit arrays, analysis scripts, and numerical result exports. Main Figure 5–9 scripts were selected from the available revisions. Data and paths were organized for execution from the repository root. Numeric input arrays, image pixels and segmentation labels were preserved byte for byte.

Day 10 labels in this brain-image archive were corrected to Day 14 based on explicit author confirmation on 23 October 2026. This is a metadata correction, not creation or interpolation of a new time point. Real Day 10 whole-body imaging mentioned in the manuscript is a separate experiment and is not represented by these brain arrays.

## Materials still needed for a complete study archive

1. CNN architecture, complete training/inference code, preprocessing and augmentation details, model-selection rules, optimizer settings, seeds, and original software versions.
2. Trained CNN weights for the endpoint folds and a sample-level split manifest for each time point. Current stored logits allow downstream analysis but do not independently demonstrate that endpoint logits are out of fold or that intermediate concentrations were excluded from training.
3. An explicit logit-to-image identifier mapping. The packaging manifest follows the extraction script's order; logit alignment is still positional.
4. Original microscopy acquisitions, whole-body fluorescence source measurements, particle-characterization inputs, and complete source material for manuscript Figures 1–4. These were not found in the archive, including its three nested RAR files.
5. Cellpose segmentation-generation code, exact model/weights/version and inference settings. The supplied `extract_features.py` reads existing masks; it does not run Cellpose.
6. Biological-replicate/tank/batch metadata, image calibration and expansion-factor records, and confirmation that filename-defined groups correspond to the experimental fish identities.
7. Author-approved code/data licenses and final publication citation/DOI when available.

## Items requiring reconciliation

### Image count versus fish count

The manuscript table lists the same counts as image columns. The extraction code loops over fish and then four views, and main downstream scripts use `VIEWS_PER_FISH = 4`. Thus the archive supports 2,968 images grouped into 742 fish identifiers. The manuscript sentence that each image-level record is one fish-level observation should be reviewed and clarified. The package does not change the manuscript.

### Historical endpoint-validation exports

Source `CNN结果/fig4A_split_counts.csv` is internally inconsistent: its first fold totals 404 control and 296 high-dose images (matching the late-time endpoint sizes), while other folds total 336 and 364 (matching Day 5). The source design table describes Day 5, and the source group predictions contain 175 Day 5 endpoint fish groups. The related `fig4A_*` and `fig4B_*` tables are excluded from the upload folder to avoid presenting a mixed export as verified CNN validation evidence. Originals remain in the untouched source ZIP. Re-export consistent training, validation, group-overlap and prediction tables from the original training run before adding them to the public release.

### Analysis revisions and sampling units

The selected held-out-dose and interpretation scripts keep four views together using fish groups. Several preserved standalone `day5_*` / `day14_*` scripts use `StratifiedKFold` directly on image rows. Those supplementary scripts should not be used as evidence of fish-independent cross-validation. The coordination script has its own source analysis conventions. Packaging preserves these methods and does not silently replace them or recompute the supplied numerical results.

Match each result to its specific analysis and figure. The supplied intermediate-dose summary reports composite score correlations of approximately 0.281 (Day 5) and 0.492 (Day 14), consistent with manuscript Figure 7. Figures 6 and 9 use other composite-score analysis contexts and report different values. These values should not be substituted for one another. Preserved source exports and new reruns should be identified separately.

### Runtime settings

The interpretation script retains the source `fast` default. Its `publication` setting changes bootstrap counts and random-forest tree counts. Original package versions and the exact settings used for every saved table were not supplied. Requirements describe dependencies, not a historical lockfile. Successful structural validation does not establish numerical reproduction of the entire manuscript.

## Preparation changes

- English directory/script names and portable relative paths.
- Corrected Day 5/Day 14 naming in filenames, code, CSV text cells and XLSX string cells; numerical table entries were preserved.
- Microscopy cache reuse disabled so a rerun does not rely on original machine paths.
- Excel exports use the explicit `sheet_name=` keyword for compatibility with pandas 3; this changes the API call syntax without changing analysis calculations. Old D10 worksheet titles were harmonized to D14.
- Original raster/vector illustrations omitted to avoid stale labels; selected scripts regenerate their figures.
- Nested plot archives omitted after inspection; the v2 scripts duplicate expanded revisions. The nested segmentation archive duplicates 1,393 expanded masks. The older five-script v1 archive is retained only in the source ZIP.
- Existing manuscript DOCX, local user paths, alternative script revisions, and mixed validation exports omitted from the upload payload.

See `validation_report.json` for checks actually performed and `provenance.csv` for file-level changes. Validation is deliberately distinguished from a complete scientific rerun.
