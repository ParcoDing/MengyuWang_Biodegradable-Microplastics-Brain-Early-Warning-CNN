# -*- coding: utf-8 -*-
"""
Figure: 16 explicit features explain CNN phenotypes at Day 5 and Day 14
======================================================================

Uses ONLY existing project files:
1) 16 classical explicit features in ./data/features/
2) CNN logits in ./results/cnn/

No 54-feature CSV is required.

Panels
------
a. Explicit feature families discriminate endpoint exposure (0 vs 500 ug/L)
   - Family-only fish-level AUROC
   - Leave-one-family-out Delta AUROC vs all features

b. Explicit feature families explain the CNN phenotype
   - Family-only fish-level cross-validated R^2
   - Leave-one-family-out Delta R^2 vs all features

c. Explicit descriptors reconstruct the CNN phenotype
   - Left: random-forest feature importance using all 16 explicit features
   - Right: fish-level OOF predicted vs observed CNN score, separately for Day 5/Day 14

d. Residual dose association
   - Fish-level residual = observed CNN score - OOF explicit-feature prediction
   - If residual retains dose association, CNN contains dose-related information not captured
     by the current 16 explicit descriptors.

Important validation rules
--------------------------
- The four views from the same fish are ALWAYS kept in the same CV fold.
- CNN logits are read directly from the eight existing *_deep_learning_logit.npy files.
- According to the existing project workflow, Day-5 endpoint files contain OOF outputs and
  intermediate-dose files contain fold-ensemble outputs; Day-14 files are used as the stored
  CNN phenotype scores for the corresponding dose groups.
- Main metrics and residual plots are reported at fish level.
- Day 5 and Day 14 are analyzed independently; no Day 14 display alias is used.

Assumption inherited from the existing project code:
- Images are ordered as 4 consecutive views per fish within each dose array.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy import stats

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score

try:
    from tqdm.auto import tqdm
except ImportError:
    # Graceful fallback if tqdm is not installed.
    def tqdm(iterable=None, total=None, desc=None, leave=True, disable=False, **kwargs):
        if iterable is None:
            class _Dummy:
                def update(self, n=1):
                    pass
                def close(self):
                    pass
                def set_postfix_str(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return _Dummy()
        return iterable


# =============================================================================
# 1. SETTINGS
# =============================================================================
# If running from the project root, keep Path(".").
# Run from the repository root.
PROJECT_ROOT = Path(".")

FEATURE_DIR = PROJECT_ROOT / "data/features"
CNN_DIR = PROJECT_ROOT / "results/cnn"
OUT_DIR = PROJECT_ROOT / "results/figures"
TABLE_DIR = PROJECT_ROOT / "results/interpretation"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
VIEWS_PER_FISH = 4

# Runtime profile.
# - "fast": recommended while checking code/figure layout; usually much faster.
# - "publication": restores the heavier settings used for the final manuscript analysis.
RUN_MODE = "fast"
SHOW_PROGRESS = True

if RUN_MODE == "fast":
    N_BOOT = 300
    N_RF_CV = 200
    N_RF_IMPORTANCE = 400
else:
    N_BOOT = 2000
    N_RF_CV = 600
    N_RF_IMPORTANCE = 800

MIN_SAMPLES_LEAF = 3
TOP_N_IMPORTANCE = 10
SAVE_FIGURE = True
SAVE_TABLES = True
DPI = 600

DAYS = ["Day 5", "Day 14"]
DOSES = ["0 ug/L", "50 ug/L", "200 ug/L", "500 ug/L"]
DOSE_VALUE = {
    "0 ug/L": 0.0,
    "50 ug/L": 50.0,
    "200 ug/L": 200.0,
    "500 ug/L": 500.0,
}

DAY_COLOR = {
    "Day 5": "#3B73B9",
    "Day 14": "#E74C3C",
}

DOSE_COLOR = {
    "0 ug/L": "#173B73",
    "50 ug/L": "#5D8BC9",
    "200 ug/L": "#E58B00",
    "500 ug/L": "#D9342B",
}

# Exact 16-feature set already used by the project scripts.
FEATURES = [
    "cell_density",
    "cell_roundness",
    "nearest_neighbor_distance",
    "local_density_variance",
    "channel_colocalization",
    "spatial_heterogeneity_increase",
    "green_mean_cell_area",
    "green_mean_eccentricity",
    "green_neighbor_degree",
    "green_close_neighbor_fraction",
    "green_dilation_connectivity",
    "texture_glcm_contrast_std",
    "texture_glcm_energy_std",
    "texture_glcm_dissimilarity_std",
    "texture_glcm_correlation_std",
    "texture_glcm_homogeneity_mean",
]

# Four interpretable feature families covering all 16 features.
FEATURE_FAMILIES = {
    "Morphology": [
        "cell_density",
        "green_mean_cell_area",
        "cell_roundness",
        "green_mean_eccentricity",
    ],
    "Spatial": [
        "nearest_neighbor_distance",
        "local_density_variance",
        "spatial_heterogeneity_increase",
        "green_neighbor_degree",
        "green_close_neighbor_fraction",
        "green_dilation_connectivity",
    ],
    "Channel": [
        "channel_colocalization",
    ],
    "Texture": [
        "texture_glcm_contrast_std",
        "texture_glcm_energy_std",
        "texture_glcm_dissimilarity_std",
        "texture_glcm_correlation_std",
        "texture_glcm_homogeneity_mean",
    ],
}

FEATURE_LABELS = {
    "cell_density": "Cell density",
    "cell_roundness": "Cell roundness",
    "nearest_neighbor_distance": "Nearest-neighbor distance",
    "local_density_variance": "Local density variance",
    "channel_colocalization": "Channel colocalization",
    "spatial_heterogeneity_increase": "Spatial heterogeneity",
    "green_mean_cell_area": "Mean cell area",
    "green_mean_eccentricity": "Mean eccentricity",
    "green_neighbor_degree": "Neighbor degree",
    "green_close_neighbor_fraction": "Close-neighbor fraction",
    "green_dilation_connectivity": "Dilation connectivity",
    "texture_glcm_contrast_std": "GLCM contrast SD",
    "texture_glcm_energy_std": "GLCM energy SD",
    "texture_glcm_dissimilarity_std": "GLCM dissimilarity SD",
    "texture_glcm_correlation_std": "GLCM correlation SD",
    "texture_glcm_homogeneity_mean": "GLCM homogeneity mean",
}


# =============================================================================
# 2. LOAD 16 EXPLICIT FEATURES
# =============================================================================
def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path


require_file(FEATURE_DIR / "feature_names.npy")
saved_names = np.load(FEATURE_DIR / "feature_names.npy", allow_pickle=False).astype(str).tolist()
missing = [f for f in FEATURES if f not in saved_names]
if missing:
    raise ValueError("feature_names.npy is missing required features:\n" + "\n".join(missing))

feature_idx = np.array([saved_names.index(f) for f in FEATURES], dtype=int)
feature_pos = {f: i for i, f in enumerate(FEATURES)}
family_idx = {
    family: np.array([feature_pos[f] for f in feats], dtype=int)
    for family, feats in FEATURE_FAMILIES.items()
}


def load_feature_array(filename: str):
    path = require_file(FEATURE_DIR / filename)
    a = np.load(path)
    if a.ndim != 2 or a.shape[0] != len(saved_names):
        raise ValueError(f"{filename}: unexpected shape {a.shape}; expected features x images")
    if a.shape[1] % VIEWS_PER_FISH != 0:
        raise ValueError(
            f"{filename}: image count {a.shape[1]} is not divisible by {VIEWS_PER_FISH}. "
            "The fish-grouping assumption would be invalid."
        )
    return a[feature_idx].T.astype(float)  # images x 16 features


EXPLICIT = {
    "Day 5": {
        "0 ug/L": load_feature_array("day5_0CT_arr.npy"),
        "50 ug/L": load_feature_array("day5_50ppb_arr.npy"),
        "200 ug/L": load_feature_array("day5_200ppb_arr.npy"),
        "500 ug/L": load_feature_array("day5_500ppb_arr.npy"),
    },
    "Day 14": {
        "0 ug/L": load_feature_array("day14_0CT_arr.npy"),
        "50 ug/L": load_feature_array("day14_50ppb_arr.npy"),
        "200 ug/L": load_feature_array("day14_200ppb_arr.npy"),
        "500 ug/L": load_feature_array("day14_500ppb_arr.npy"),
    },
}


# =============================================================================
# 3. LOAD CNN SCORES FROM THE EIGHT EXISTING PROJECT FILES
# =============================================================================
CNN_FILES = {
    "Day 5": {
        "0 ug/L": "day5_0CT_deep_learning_logit.npy",
        "50 ug/L": "day5_50ppb_deep_learning_logit.npy",
        "200 ug/L": "day5_200ppb_deep_learning_logit.npy",
        "500 ug/L": "day5_500ppb_deep_learning_logit.npy",
    },
    "Day 14": {
        "0 ug/L": "day14_0CT_deep_learning_logit.npy",
        "50 ug/L": "day14_50ppb_deep_learning_logit.npy",
        "200 ug/L": "day14_200ppb_deep_learning_logit.npy",
        "500 ug/L": "day14_500ppb_deep_learning_logit.npy",
    },
}


def load_cnn_scores(day: str):
    out = {}
    print(f"\n[{day}] CNN score files")
    for dose in DOSES:
        p = require_file(CNN_DIR / CNN_FILES[day][dose])
        v = np.load(p).astype(float).ravel()
        if len(v) != len(EXPLICIT[day][dose]):
            raise ValueError(
                f"{day} {dose}: CNN score count {len(v)} != explicit-feature image count "
                f"{len(EXPLICIT[day][dose])}. Check image ordering/files."
            )
        out[dose] = v
        print(f"  {dose:>8}: {p}")
    return out


print("=" * 78)
print(f"16-feature/CNN interpretation | RUN_MODE={RUN_MODE}")
print(f"RF trees per CV fit={N_RF_CV}; importance trees={N_RF_IMPORTANCE}; bootstrap={N_BOOT}")
print("=" * 78)
CNN = {day: load_cnn_scores(day) for day in DAYS}


# =============================================================================
# 4. BUILD DAY-LEVEL DATASETS AND FISH METADATA
# =============================================================================
def build_day_dataset(day: str):
    X_blocks, cnn_blocks, dose_labels, dose_values = [], [], [], []
    fish_ids, view_ids = [], []
    offset = 0

    for dose in DOSES:
        X = EXPLICIT[day][dose]
        ycnn = CNN[day][dose]
        n_img = len(X)
        n_fish = n_img // VIEWS_PER_FISH

        X_blocks.append(X)
        cnn_blocks.append(ycnn)
        dose_labels.extend([dose] * n_img)
        dose_values.extend([DOSE_VALUE[dose]] * n_img)

        fish_ids.extend(np.repeat(np.arange(offset, offset + n_fish), VIEWS_PER_FISH))
        view_ids.extend(np.tile(np.arange(1, VIEWS_PER_FISH + 1), n_fish))
        offset += n_fish

    meta = pd.DataFrame({
        "Day": day,
        "Dose": dose_labels,
        "Dose_value": dose_values,
        "Fish_id": np.asarray(fish_ids, dtype=int),
        "View": np.asarray(view_ids, dtype=int),
    })

    return {
        "X": np.vstack(X_blocks).astype(float),
        "cnn": np.concatenate(cnn_blocks).astype(float),
        "meta": meta,
    }


DAYSETS = {day: build_day_dataset(day) for day in DAYS}


# =============================================================================
# 5. GROUPED CV + METRIC HELPERS
# =============================================================================
def grouped_splits(meta: pd.DataFrame, stratify_col: str):
    fish_table = (
        meta[["Fish_id", stratify_col]]
        .drop_duplicates("Fish_id")
        .sort_values("Fish_id")
        .reset_index(drop=True)
    )
    fish = fish_table["Fish_id"].to_numpy()
    labels = fish_table[stratify_col].to_numpy()

    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(cv.split(fish, labels), start=1):
        train_fish = fish[tr]
        test_fish = fish[te]
        yield (
            fold,
            meta["Fish_id"].isin(train_fish).to_numpy(),
            meta["Fish_id"].isin(test_fish).to_numpy(),
        )


def fish_aggregate(meta: pd.DataFrame, **columns):
    df = meta[["Day", "Dose", "Dose_value", "Fish_id"]].copy()
    for name, values in columns.items():
        df[name] = np.asarray(values, dtype=float)
    agg = (
        df.groupby(["Day", "Dose", "Dose_value", "Fish_id"], as_index=False)
        .mean(numeric_only=True)
    )
    return agg


def bootstrap_ci(values_fn, df: pd.DataFrame, n_boot=N_BOOT, seed=12345):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(df)
    iterator = tqdm(
        range(n_boot),
        desc="bootstrap CI",
        leave=False,
        disable=not SHOW_PROGRESS,
        mininterval=0.5,
    )
    for _ in iterator:
        idx = rng.integers(0, n, n)
        try:
            v = values_fn(df.iloc[idx].reset_index(drop=True))
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def auc_with_ci(fish_df: pd.DataFrame, score_col: str, seed=123):
    y = fish_df["Endpoint_label"].to_numpy(int)
    s = fish_df[score_col].to_numpy(float)
    point = roc_auc_score(y, s)

    def metric(d):
        yy = d["Endpoint_label"].to_numpy(int)
        ss = d[score_col].to_numpy(float)
        if len(np.unique(yy)) < 2:
            return np.nan
        return roc_auc_score(yy, ss)

    lo, hi = bootstrap_ci(metric, fish_df, seed=seed)
    return float(point), float(lo), float(hi)


def r2_with_ci(fish_df: pd.DataFrame, pred_col: str, seed=123):
    y = fish_df["Observed_CNN"].to_numpy(float)
    p = fish_df[pred_col].to_numpy(float)
    point = r2_score(y, p)

    def metric(d):
        yy = d["Observed_CNN"].to_numpy(float)
        pp = d[pred_col].to_numpy(float)
        if np.nanstd(yy) <= 1e-12:
            return np.nan
        return r2_score(yy, pp)

    lo, hi = bootstrap_ci(metric, fish_df, seed=seed)
    return float(point), float(lo), float(hi)


def paired_delta_ci(df: pd.DataFrame, metric: str, all_col: str, variant_col: str, seed=123):
    if metric == "auc":
        def calc(d, col):
            y = d["Endpoint_label"].to_numpy(int)
            s = d[col].to_numpy(float)
            if len(np.unique(y)) < 2:
                return np.nan
            return roc_auc_score(y, s)
    elif metric == "r2":
        def calc(d, col):
            y = d["Observed_CNN"].to_numpy(float)
            p = d[col].to_numpy(float)
            if np.nanstd(y) <= 1e-12:
                return np.nan
            return r2_score(y, p)
    else:
        raise ValueError(metric)

    point = calc(df, variant_col) - calc(df, all_col)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(df)
    iterator = tqdm(
        range(N_BOOT),
        desc=f"bootstrap delta {metric}",
        leave=False,
        disable=not SHOW_PROGRESS,
        mininterval=0.5,
    )
    for _ in iterator:
        idx = rng.integers(0, n, n)
        d = df.iloc[idx].reset_index(drop=True)
        boots.append(calc(d, variant_col) - calc(d, all_col))
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# =============================================================================
# 6. PANEL A: FEATURE FAMILIES DISCRIMINATE ENDPOINT EXPOSURE
# =============================================================================
classifier_template = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )),
])


def endpoint_classifier_oof(day: str, feature_indices: np.ndarray, progress_desc=None):
    ds = DAYSETS[day]
    endpoint = ds["meta"]["Dose"].isin(["0 ug/L", "500 ug/L"]).to_numpy()
    X = ds["X"][endpoint][:, feature_indices]
    meta = ds["meta"].loc[endpoint].reset_index(drop=True).copy()
    meta["Endpoint_label"] = meta["Dose"].eq("500 ug/L").astype(int)
    y = meta["Endpoint_label"].to_numpy(int)

    oof = np.full(len(meta), np.nan)
    splits = list(grouped_splits(meta, "Endpoint_label"))
    iterator = tqdm(
        splits,
        total=len(splits),
        desc=progress_desc or f"{day} classifier",
        leave=False,
        disable=not SHOW_PROGRESS,
    )
    for _, tr, te in iterator:
        model = clone(classifier_template)
        model.fit(X[tr], y[tr])
        oof[te] = model.decision_function(X[te])

    fish = fish_aggregate(meta, Exposure_score=oof)
    label_map = meta[["Fish_id", "Endpoint_label"]].drop_duplicates("Fish_id")
    fish = fish.merge(label_map, on="Fish_id", how="left")
    return oof, fish


PANEL_A_ROWS = []
PANEL_A_FISH = {}
ALL_IDX = np.arange(len(FEATURES), dtype=int)

for di, day in enumerate(DAYS):
    # All features first, used as paired reference for LOFO deltas.
    print(f"\n[Panel a] {day}: exposure discrimination")
    all_oof, all_fish = endpoint_classifier_oof(day, ALL_IDX, f"{day} A all")
    all_fish = all_fish.rename(columns={"Exposure_score": "score_all"})
    PANEL_A_FISH[(day, "All features")] = all_fish.copy()
    auc, lo, hi = auc_with_ci(all_fish.rename(columns={"score_all": "score"}), "score", seed=100 + di)
    PANEL_A_ROWS.append({
        "Day": day, "Analysis": "Family-only", "Feature_set": "All features",
        "Metric": "AUROC", "Estimate": auc, "CI_low": lo, "CI_high": hi,
    })

    # Family-only
    for fi, (family, idx) in enumerate(family_idx.items()):
        _, fam_fish = endpoint_classifier_oof(day, idx, f"{day} A family {family}")
        fam_fish = fam_fish.rename(columns={"Exposure_score": "score"})
        auc, lo, hi = auc_with_ci(fam_fish, "score", seed=200 + di * 20 + fi)
        PANEL_A_FISH[(day, family)] = fam_fish.copy()
        PANEL_A_ROWS.append({
            "Day": day, "Analysis": "Family-only", "Feature_set": family,
            "Metric": "AUROC", "Estimate": auc, "CI_low": lo, "CI_high": hi,
        })

    # Leave-one-family-out
    for fi, (family, idx) in enumerate(family_idx.items()):
        keep = np.setdiff1d(ALL_IDX, idx)
        _, loo_fish = endpoint_classifier_oof(day, keep, f"{day} A minus {family}")
        loo_fish = loo_fish.rename(columns={"Exposure_score": f"score_minus_{family}"})
        paired = all_fish.merge(
            loo_fish[["Fish_id", f"score_minus_{family}"]],
            on="Fish_id", how="inner"
        )
        delta, dlo, dhi = paired_delta_ci(
            paired, "auc", "score_all", f"score_minus_{family}",
            seed=300 + di * 20 + fi,
        )
        PANEL_A_ROWS.append({
            "Day": day, "Analysis": "Leave-one-family-out",
            "Feature_set": f"All - {family}",
            "Metric": "Delta_AUROC_vs_all", "Estimate": delta,
            "CI_low": dlo, "CI_high": dhi,
        })

PANEL_A_DF = pd.DataFrame(PANEL_A_ROWS)


# =============================================================================
# 7. PANEL B/C/D: EXPLICIT FEATURES EXPLAIN CNN PHENOTYPE
# =============================================================================
def rf_regressor(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=N_RF_CV,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def cnn_regression_oof(day: str, feature_indices: np.ndarray, seed_offset=0, progress_desc=None):
    ds = DAYSETS[day]
    X = ds["X"][:, feature_indices]
    y = ds["cnn"]
    meta = ds["meta"].copy()

    oof = np.full(len(meta), np.nan)
    splits = list(grouped_splits(meta, "Dose"))
    iterator = tqdm(
        splits,
        total=len(splits),
        desc=progress_desc or f"{day} RF-CV",
        leave=False,
        disable=not SHOW_PROGRESS,
    )
    t0 = perf_counter()
    for fold, tr, te in iterator:
        iterator.set_postfix_str(f"fold={fold}/{N_SPLITS}")
        model = rf_regressor(RANDOM_STATE + seed_offset + fold)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    if SHOW_PROGRESS:
        print(f"    completed {progress_desc or day} in {perf_counter() - t0:.1f} s")

    fish = fish_aggregate(meta, Observed_CNN=y, Predicted_CNN=oof)
    return oof, fish


PANEL_B_ROWS = []
RECONSTRUCTION = {}
RESIDUAL_FISH = {}

for di, day in enumerate(DAYS):
    print(f"\n[Panel b/c/d] {day}: reconstructing CNN phenotype from 16 explicit features")
    # All-feature OOF reconstruction
    all_pred_img, all_fish = cnn_regression_oof(day, ALL_IDX, seed_offset=1000 * di, progress_desc=f"{day} B all features")
    all_fish = all_fish.rename(columns={"Predicted_CNN": "pred_all"})
    RECONSTRUCTION[day] = all_fish.copy()

    r2, lo, hi = r2_with_ci(
        all_fish.rename(columns={"pred_all": "pred"}), "pred", seed=400 + di
    )
    PANEL_B_ROWS.append({
        "Day": day, "Analysis": "Family-only", "Feature_set": "All features",
        "Metric": "R2", "Estimate": r2, "CI_low": lo, "CI_high": hi,
    })

    # Family-only regression
    for fi, (family, idx) in enumerate(family_idx.items()):
        _, fam_fish = cnn_regression_oof(day, idx, seed_offset=2000 + 100 * fi + 1000 * di, progress_desc=f"{day} B family {family}")
        fam_fish = fam_fish.rename(columns={"Predicted_CNN": "pred"})
        r2, lo, hi = r2_with_ci(fam_fish, "pred", seed=500 + di * 20 + fi)
        PANEL_B_ROWS.append({
            "Day": day, "Analysis": "Family-only", "Feature_set": family,
            "Metric": "R2", "Estimate": r2, "CI_low": lo, "CI_high": hi,
        })

    # Leave-one-family-out regression
    for fi, (family, idx) in enumerate(family_idx.items()):
        keep = np.setdiff1d(ALL_IDX, idx)
        _, loo_fish = cnn_regression_oof(day, keep, seed_offset=4000 + 100 * fi + 1000 * di, progress_desc=f"{day} B minus {family}")
        loo_fish = loo_fish.rename(columns={"Predicted_CNN": f"pred_minus_{family}"})
        paired = all_fish.merge(
            loo_fish[["Fish_id", f"pred_minus_{family}"]],
            on="Fish_id", how="inner"
        )
        delta, dlo, dhi = paired_delta_ci(
            paired, "r2", "pred_all", f"pred_minus_{family}",
            seed=600 + di * 20 + fi,
        )
        PANEL_B_ROWS.append({
            "Day": day, "Analysis": "Leave-one-family-out",
            "Feature_set": f"All - {family}",
            "Metric": "Delta_R2_vs_all", "Estimate": delta,
            "CI_low": dlo, "CI_high": dhi,
        })

    # Residuals from strict all-feature OOF prediction, aggregated at fish level.
    residual_img = DAYSETS[day]["cnn"] - all_pred_img
    residual_fish = fish_aggregate(DAYSETS[day]["meta"], Residual=residual_img)
    RESIDUAL_FISH[day] = residual_fish

PANEL_B_DF = pd.DataFrame(PANEL_B_ROWS)
RECON_FISH_DF = pd.concat([
    RECONSTRUCTION[d].assign(Day=d).rename(columns={"pred_all": "Predicted_CNN"})
    for d in DAYS
], ignore_index=True)
RESIDUAL_FISH_DF = pd.concat([RESIDUAL_FISH[d] for d in DAYS], ignore_index=True)


# =============================================================================
# 8. PANEL C LEFT: REAL 16-FEATURE IMPORTANCE FOR DAY 5 + DAY 10
# =============================================================================
def fit_importance(day: str):
    print(f"\n[Panel c importance] {day}: fitting {N_RF_IMPORTANCE}-tree forest ...", flush=True)
    t0 = perf_counter()
    ds = DAYSETS[day]
    X = ds["X"]
    y = ds["cnn"]

    imputer = SimpleImputer(strategy="median")
    Xi = imputer.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=N_RF_IMPORTANCE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE + (0 if day == "Day 5" else 10000),
        n_jobs=-1,
    )
    rf.fit(Xi, y)
    print(f"    importance forest completed in {perf_counter() - t0:.1f} s", flush=True)
    imp = rf.feature_importances_.astype(float)

    return pd.DataFrame({
        "Day": day,
        "Feature": FEATURES,
        "Feature_label": [FEATURE_LABELS[f] for f in FEATURES],
        "Importance": imp,
    })


IMPORTANCE_DF = pd.concat([fit_importance(day) for day in DAYS], ignore_index=True)

# Rank features by mean importance across time, and plot top N.
rank_df = (
    IMPORTANCE_DF.groupby(["Feature", "Feature_label"], as_index=False)["Importance"]
    .mean()
    .rename(columns={"Importance": "Mean_importance"})
    .sort_values("Mean_importance", ascending=False)
)
TOP_FEATURES = rank_df.head(TOP_N_IMPORTANCE)["Feature"].tolist()


# =============================================================================
# 9. SUMMARY STATISTICS FOR RECONSTRUCTION + RESIDUALS
# =============================================================================
RECON_STATS = []
RESIDUAL_STATS = []

for di, day in enumerate(DAYS):
    d = RECONSTRUCTION[day].copy()
    y = d["Observed_CNN"].to_numpy(float)
    p = d["pred_all"].to_numpy(float)
    r2 = r2_score(y, p)
    rho, rho_p = stats.spearmanr(y, p)
    RECON_STATS.append({
        "Day": day,
        "Fish_n": len(d),
        "R2": float(r2),
        "Spearman_r": float(rho),
        "Spearman_p": float(rho_p),
    })

    r = RESIDUAL_FISH[day].copy()
    rho_res, p_res = stats.spearmanr(r["Dose_value"], r["Residual"])
    RESIDUAL_STATS.append({
        "Day": day,
        "Fish_n": len(r),
        "Residual_Spearman_r_vs_dose": float(rho_res),
        "Residual_Spearman_p": float(p_res),
    })

RECON_STATS_DF = pd.DataFrame(RECON_STATS)
RESIDUAL_STATS_DF = pd.DataFrame(RESIDUAL_STATS)


print("\n[Stage] numerical analyses completed; saving tables and drawing figure ...", flush=True)

# =============================================================================
# 10. SAVE TABLES
# =============================================================================
if SAVE_TABLES:
    PANEL_A_DF.to_csv(TABLE_DIR / "panel_a_exposure_family_metrics.csv", index=False, encoding="utf-8-sig")
    PANEL_B_DF.to_csv(TABLE_DIR / "panel_b_cnn_family_r2_metrics.csv", index=False, encoding="utf-8-sig")
    IMPORTANCE_DF.to_csv(TABLE_DIR / "panel_c_16feature_importance.csv", index=False, encoding="utf-8-sig")
    RECON_FISH_DF.to_csv(TABLE_DIR / "panel_c_reconstruction_fish_level.csv", index=False, encoding="utf-8-sig")
    RESIDUAL_FISH_DF.to_csv(TABLE_DIR / "panel_d_residual_fish_level.csv", index=False, encoding="utf-8-sig")
    RECON_STATS_DF.to_csv(TABLE_DIR / "reconstruction_summary.csv", index=False, encoding="utf-8-sig")
    RESIDUAL_STATS_DF.to_csv(TABLE_DIR / "residual_dose_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(TABLE_DIR / "Fig_feature_family_CNN_interpretation_16features_results.xlsx") as xw:
        PANEL_A_DF.to_excel(xw, sheet_name="exposure_family", index=False); PANEL_B_DF.to_excel(xw, sheet_name="cnn_family_r2", index=False)
        IMPORTANCE_DF.to_excel(xw, sheet_name="feature_importance", index=False); rank_df.to_excel(xw, sheet_name="importance_rank", index=False)
        RECON_STATS_DF.to_excel(xw, sheet_name="reconstruction_summary", index=False); RESIDUAL_STATS_DF.to_excel(xw, sheet_name="residual_summary", index=False)
        RECON_FISH_DF.to_excel(xw, sheet_name="reconstruction_fish", index=False); RESIDUAL_FISH_DF.to_excel(xw, sheet_name="residual_fish", index=False)


# =============================================================================
# 11. PUBLICATION FIGURE
# =============================================================================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.linewidth": 0.75,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.7)


def panel_header(fig, spec, letter, title):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.text(0.00, 0.60, letter, fontsize=18, fontweight="bold", ha="left", va="center")
    ax.text(0.075, 0.60, title, fontsize=14, fontweight="bold", ha="left", va="center")
    return ax


def paired_forest(ax, rows, categories, xlabel, ref=None, chance_label=None):
    y = np.arange(len(categories))[::-1]
    offset = 0.11

    if ref is not None:
        ax.axvline(ref, color="#999999", lw=0.75, ls=(0, (4, 4)), zorder=0)

    for day, dy in [("Day 5", offset), ("Day 14", -offset)]:
        sub = rows[rows["Day"] == day].set_index("Feature_set").reindex(categories)
        est = sub["Estimate"].to_numpy(float)
        lo = sub["CI_low"].to_numpy(float)
        hi = sub["CI_high"].to_numpy(float)
        xerr = np.vstack([est - lo, hi - est])
        ax.errorbar(
            est, y + dy, xerr=xerr,
            fmt="o", ms=4.5, capsize=2.0, elinewidth=0.85,
            color=DAY_COLOR[day], mfc=DAY_COLOR[day], mec=DAY_COLOR[day],
            label=day, zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.5, len(categories) - 0.5)
    clean_axis(ax)

    if chance_label is not None and ref is not None:
        ax.text(ref, 1.01, chance_label, transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=10, color="#777777")


def dynamic_xlim(rows, ref=None, pad_frac=0.15):
    vals = np.r_[rows["CI_low"].to_numpy(float), rows["CI_high"].to_numpy(float)]
    vals = vals[np.isfinite(vals)]
    if ref is not None:
        vals = np.r_[vals, ref]
    lo, hi = vals.min(), vals.max()
    span = max(hi - lo, 0.1)
    return lo - pad_frac * span, hi + pad_frac * span


fig = plt.figure(figsize=(24, 16), facecolor="white")
outer = GridSpec(
    2, 2, figure=fig,
    left=0.07, right=0.985, bottom=0.075, top=0.965,
    wspace=0.34, hspace=0.34,
)

# -----------------------------------------------------------------------------
# Panel a
# -----------------------------------------------------------------------------
gs_a = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[0, 0],
    height_ratios=[0.16, 0.84], width_ratios=[1.0, 1.08],
    wspace=0.62, hspace=0.03,
)
panel_header(fig, gs_a[0, :], "a", "Explicit feature families discriminate exposure")
ax_a1 = fig.add_subplot(gs_a[1, 0])
ax_a2 = fig.add_subplot(gs_a[1, 1])

fam_order = list(FEATURE_FAMILIES.keys()) + ["All features"]
a_family = PANEL_A_DF[PANEL_A_DF["Analysis"] == "Family-only"].copy()
paired_forest(ax_a1, a_family, fam_order, "Fish-level AUROC", ref=0.5, chance_label="")
ax_a1.set_xlim(*dynamic_xlim(a_family, ref=0.5))
ax_a1.set_title("Family-only", fontweight="bold", pad=5)

loo_order = [f"All - {f}" for f in FEATURE_FAMILIES]
a_loo = PANEL_A_DF[PANEL_A_DF["Analysis"] == "Leave-one-family-out"].copy()
paired_forest(ax_a2, a_loo, loo_order, r"$\Delta$AUROC vs all features", ref=0.0)
ax_a2.set_xlim(*dynamic_xlim(a_loo, ref=0.0))
ax_a2.set_title("Leave-one-family-out", fontweight="bold", pad=5)
ax_a2.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.4,0.05))

# -----------------------------------------------------------------------------
# Panel b
# -----------------------------------------------------------------------------
gs_b = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[0, 1],
    height_ratios=[0.16, 0.84], width_ratios=[1.0, 1.08],
    wspace=0.62, hspace=0.03,
)
panel_header(fig, gs_b[0, :], "b", "Explicit feature families explain the CNN phenotype")
ax_b1 = fig.add_subplot(gs_b[1, 0])
ax_b2 = fig.add_subplot(gs_b[1, 1])

b_family = PANEL_B_DF[PANEL_B_DF["Analysis"] == "Family-only"].copy()
paired_forest(ax_b1, b_family, fam_order, r"Fish-level cross-validated $R^2$", ref=0.0)
ax_b1.set_xlim(*dynamic_xlim(b_family, ref=0.0))
ax_b1.set_title("Family-only", fontweight="bold", pad=5)

b_loo = PANEL_B_DF[PANEL_B_DF["Analysis"] == "Leave-one-family-out"].copy()
paired_forest(ax_b2, b_loo, loo_order, r"$\Delta R^2$ vs all features", ref=0.0)
ax_b2.set_xlim(*dynamic_xlim(b_loo, ref=0.0))
ax_b2.set_title("Leave-one-family-out", fontweight="bold", pad=5)
ax_b2.legend(frameon=False, loc="lower right")

# -----------------------------------------------------------------------------
# Panel c: importance + Day5/Day14 reconstruction scatter
# -----------------------------------------------------------------------------
gs_c = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[1, 0],
    height_ratios=[0.16, 0.84], width_ratios=[1.05, 1.45],
    wspace=0.36, hspace=0.03,
)
panel_header(fig, gs_c[0, :], "c", "Explicit descriptors reconstruct the CNN phenotype")
ax_c1 = fig.add_subplot(gs_c[1, 0])

# paired importance bars, top features ranked by mean importance
imp_top = IMPORTANCE_DF[IMPORTANCE_DF["Feature"].isin(TOP_FEATURES)].copy()
order = (
    rank_df[rank_df["Feature"].isin(TOP_FEATURES)]
    .sort_values("Mean_importance", ascending=True)["Feature"].tolist()
)
y = np.arange(len(order)); bar_h = 0.34
for day, dy in [("Day 5", bar_h / 2), ("Day 14", -bar_h / 2)]:
    sub = imp_top[imp_top["Day"] == day].set_index("Feature").reindex(order)
    ax_c1.barh(y + dy, sub["Importance"].to_numpy(float), height=bar_h,
               color=DAY_COLOR[day], label=day, zorder=3)
ax_c1.set_yticks(y)
ax_c1.set_yticklabels([FEATURE_LABELS[f] for f in order])
ax_c1.set_xlabel("Random-forest importance")
ax_c1.set_title(f"Top {TOP_N_IMPORTANCE} explicit descriptors", fontweight="bold", pad=5)
ax_c1.legend(frameon=False, loc="lower right")
clean_axis(ax_c1)

# right side: 2 scatter plots
scatter_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_c[1, 1], wspace=0.34)
scatter_axes = []
for j, day in enumerate(DAYS):
    ax = fig.add_subplot(scatter_spec[0, j]); scatter_axes.append(ax)
    d = RECONSTRUCTION[day].copy()
    x = d["pred_all"].to_numpy(float)
    yy = d["Observed_CNN"].to_numpy(float)
    dose = d["Dose"].to_numpy(str)

    for dose_name in DOSES:
        m = dose == dose_name
        ax.scatter(
            x[m], yy[m], s=15, alpha=0.72,
            color=DOSE_COLOR[dose_name], linewidth=0,
        )

    combined = np.r_[x, yy]
    lo, hi = np.nanpercentile(combined, [1, 99])
    pad = 0.08 * (hi - lo + 1e-8)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.8, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if j == 0:
        ax.set_ylabel("Observed CNN score")
    ax.set_title(day, fontweight="bold", pad=5)

    st = RECON_STATS_DF[RECON_STATS_DF["Day"] == day].iloc[0]
    ax.text(
        0.04, 0.96,
        rf"$R^2$ = {st.R2:.3f}" + "\n" + rf"$r_s$ = {st.Spearman_r:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
    )
    clean_axis(ax)

# shared x label for the two reconstruction plots
pos_l, pos_r = scatter_axes[0].get_position(), scatter_axes[1].get_position()
fig.text((pos_l.x0 + pos_r.x1) / 2, min(pos_l.y0, pos_r.y0) - 0.032,
         "OOF predicted CNN score", ha="center", va="top", fontsize=13)

# one dose legend for panel c
handles = [
    Line2D([0], [0], marker="o", ls="none", ms=5, mfc=DOSE_COLOR[d], mec=DOSE_COLOR[d], label=d)
    for d in DOSES
]
fig.legend(handles=handles, frameon=False, ncol=1,
           bbox_to_anchor=(1.02, 0.5), bbox_transform=scatter_axes[1].transAxes, loc="center left",
           handletextpad=0.25, columnspacing=0.8)

# -----------------------------------------------------------------------------
# Panel d: fish-level residual dose association
# -----------------------------------------------------------------------------
gs_d = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[1, 1],
    height_ratios=[0.16, 0.84], width_ratios=[1, 1],
    wspace=0.34, hspace=0.03,
)
panel_header(fig, gs_d[0, :], "d", "Residual dose association after 16-feature reconstruction")

for j, day in enumerate(DAYS):
    ax = fig.add_subplot(gs_d[1, j])
    d = RESIDUAL_FISH[day].copy()
    rng = np.random.default_rng(900 + j)

    all_vals = []
    for i, dose_name in enumerate(DOSES):
        vals = d.loc[d["Dose"] == dose_name, "Residual"].to_numpy(float)
        all_vals.extend(vals.tolist())
        ax.boxplot(
            vals, positions=[i], widths=0.48, patch_artist=True, showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": DOSE_COLOR[dose_name], "linewidth": 0.85},
            medianprops={"color": DOSE_COLOR[dose_name], "linewidth": 1.1},
            whiskerprops={"color": DOSE_COLOR[dose_name], "linewidth": 0.8},
            capprops={"color": DOSE_COLOR[dose_name], "linewidth": 0.8},
        )
        jitter = rng.normal(0, 0.055, len(vals))
        ax.scatter(i + jitter, vals, s=11, alpha=0.58, color=DOSE_COLOR[dose_name], linewidth=0)

    ax.axhline(0, color="#999999", lw=0.75, ls=(0, (3, 3)))
    ax.set_xticks(range(4), ["0", "50", "200", "500"])
    ax.set_xlabel("Dose (ug/L)")
    if j == 0:
        ax.set_ylabel("Fish-level OOF residual")
    ax.set_title(day, fontweight="bold", pad=5)

    st = RESIDUAL_STATS_DF[RESIDUAL_STATS_DF["Day"] == day].iloc[0]
    ax.text(
        0.04, 0.96,
        rf"Dose $r_s$ = {st.Residual_Spearman_r_vs_dose:.3f}" + "\n" +
        rf"$P$ = {st.Residual_Spearman_p:.2e}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
    )
    clean_axis(ax)


# =============================================================================
# 12. SAVE
# =============================================================================
base = OUT_DIR / "Fig_feature_family_CNN_interpretation_16features_REAL"
if SAVE_FIGURE:
    print("[Stage] saving high-resolution PNG/PDF ...", flush=True)
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

plt.show()

print("\n" + "=" * 90)
print("Panel A: exposure-family metrics")
print(PANEL_A_DF.to_string(index=False))
print("\nPanel B: CNN reconstruction family metrics")
print(PANEL_B_DF.to_string(index=False))
print("\nReconstruction summary")
print(RECON_STATS_DF.to_string(index=False))
print("\nResidual dose summary")
print(RESIDUAL_STATS_DF.to_string(index=False))
print("\nFigure:", base.with_suffix(".png").resolve())
print("Tables:", TABLE_DIR.resolve())
print("=" * 90)


print("\n[DONE] Figure and tables generated.", flush=True)