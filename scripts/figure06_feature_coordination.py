# -*- coding: utf-8 -*-
# Time labels harmonized to Day 5 / Day 14 following author confirmation.
"""
Day 5 / Day 14 explicit phenotype organization
+ phenotype coordination
+ family-balanced composite phenotype

Panels
------
a  Day 5 oriented standardized-effect heatmap
b  Day 14 oriented standardized-effect heatmap
c  Phenotype coordination (PC1 explained variance; feature-bootstrap 95% CI)
d  Family-balanced composite phenotype score (Day 5 / Day 14)

Set USE_MOCK_DATA=True to run without project files. The default is False.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.gridspec import GridSpecFromSubplotSpec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold

# ============================================================
# 1. Settings
# ============================================================
USE_MOCK_DATA = False
DATA_DIR = Path("./data/features")
OUT_DIR = Path("./results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FIGURES = True
SAVE_TABLES = True
N_BOOT = 2000
RANDOM_SEED = 12345
N_SPLITS = 5
COMPOSITE_RANDOM_STATE = 42
MAX_POINTS_PER_DOSE = 55

# ============================================================
# 2. Features and groups
# ============================================================
FEATURES = [
    "cell_density", "cell_roundness", "nearest_neighbor_distance",
    "local_density_variance", "channel_colocalization",
    "spatial_heterogeneity_increase", "green_mean_cell_area",
    "green_mean_eccentricity", "green_neighbor_degree",
    "green_close_neighbor_fraction", "green_dilation_connectivity",
    "texture_glcm_contrast_std", "texture_glcm_energy_std",
    "texture_glcm_dissimilarity_std", "texture_glcm_correlation_std",
    "texture_glcm_homogeneity_mean",
]

LABELS = {
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

GROUPS = {
    "Abundance / size": ["cell_density", "green_mean_cell_area"],
    "Shape": ["cell_roundness", "green_mean_eccentricity"],
    "Spatial organization": [
        "nearest_neighbor_distance", "local_density_variance",
        "spatial_heterogeneity_increase", "green_neighbor_degree",
        "green_close_neighbor_fraction", "green_dilation_connectivity",
    ],
    "Channel relationship": ["channel_colocalization"],
    "Texture": [
        "texture_glcm_contrast_std", "texture_glcm_energy_std",
        "texture_glcm_dissimilarity_std", "texture_glcm_correlation_std",
        "texture_glcm_homogeneity_mean",
    ],
}

FEATURE_ORDER = [f for g in GROUPS.values() for f in g]
DOSES = ["0 ug/L", "50 ug/L", "200 ug/L", "500 ug/L"]
DOSE_NUM = {"0 ug/L": 0.0, "50 ug/L": 50.0, "200 ug/L": 200.0, "500 ug/L": 500.0}
DISPLAY_DOSES = {"0 ug/L": "0 µg/L", "50 ug/L": "50 µg/L", "200 ug/L": "200 µg/L", "500 ug/L": "500 µg/L"}

# ============================================================
# 3. Data loading / mock data
# ============================================================
def make_mock_data(seed=2026):
    rng = np.random.default_rng(seed)
    texture_idx = np.array([FEATURES.index(f) for f in GROUPS["Texture"]])
    raw_direction = rng.choice([-1.0, 1.0], size=len(FEATURES))
    out = {}
    for day, n in [("Day 5", 220), ("Day 14", 240)]:
        out[day] = {}
        base = rng.normal(0, 0.25, size=len(FEATURES))
        scale = rng.uniform(0.75, 1.2, size=len(FEATURES))
        if day == "Day 5":
            strength = {"0 ug/L": 0.0, "50 ug/L": 0.18, "200 ug/L": 0.26, "500 ug/L": 0.62}
            m50 = rng.normal(0.9, 0.45, size=len(FEATURES))
            m200 = rng.normal(0.8, 0.55, size=len(FEATURES))
            m50[texture_idx] += 0.45
            m200[texture_idx] += 0.35
        else:
            strength = {"0 ug/L": 0.0, "50 ug/L": 0.32, "200 ug/L": 0.68, "500 ug/L": 1.12}
            m50 = rng.normal(1.0, 0.15, size=len(FEATURES))
            m200 = rng.normal(1.0, 0.15, size=len(FEATURES))
        for dose in DOSES:
            modifier = m50 if dose == "50 ug/L" else m200 if dose == "200 ug/L" else np.ones(len(FEATURES))
            effect = raw_direction * strength[dose] * modifier
            noise = rng.normal(0, scale[:, None], size=(len(FEATURES), n))
            out[day][dose] = base[:, None] + effect[:, None] + noise
    return FEATURES.copy(), out

if USE_MOCK_DATA:
    SAVED_FEATURES, DAY_DATA = make_mock_data()
else:
    SAVED_FEATURES = np.load(DATA_DIR / "feature_names.npy", allow_pickle=False).astype(str).tolist()
    DAY_DATA = {
        "Day 5": {
            "0 ug/L": np.load(DATA_DIR / "day5_0CT_arr.npy"),
            "50 ug/L": np.load(DATA_DIR / "day5_50ppb_arr.npy"),
            "200 ug/L": np.load(DATA_DIR / "day5_200ppb_arr.npy"),
            "500 ug/L": np.load(DATA_DIR / "day5_500ppb_arr.npy"),
        },
        "Day 14": {
            "0 ug/L": np.load(DATA_DIR / "day14_0CT_arr.npy"),
            "50 ug/L": np.load(DATA_DIR / "day14_50ppb_arr.npy"),
            "200 ug/L": np.load(DATA_DIR / "day14_200ppb_arr.npy"),
            "500 ug/L": np.load(DATA_DIR / "day14_500ppb_arr.npy"),
        },
    }

missing = [f for f in FEATURES if f not in SAVED_FEATURES]
if missing:
    raise ValueError("Missing features:\n" + "\n".join(missing))
for day, dd in DAY_DATA.items():
    for dose, arr in dd.items():
        if arr.ndim != 2 or arr.shape[0] != len(SAVED_FEATURES):
            raise ValueError(f"{day} {dose}: invalid shape {arr.shape}")

FEATURE_INDEX = {f: SAVED_FEATURES.index(f) for f in FEATURES}
SELECTED_IDX = np.array([FEATURE_INDEX[f] for f in FEATURE_ORDER], dtype=int)
GROUP_IDX = {
    g: np.array([FEATURE_ORDER.index(f) for f in fs], dtype=int)
    for g, fs in GROUPS.items()
}

# ============================================================
# 4. Heatmap / coordination
# ============================================================
def fmt(x):
    if pd.isna(x):
        return ""
    x = round(float(x), 1)
    return "0" if x == 0 else f"{x:.1f}"

def annot_df(df):
    return df.map(fmt)

def pc1_ratio(X):
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < 2:
        return np.nan
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    total = np.sum(s ** 2)
    return np.nan if total <= 1e-12 else float(s[0] ** 2 / total)

def bootstrap_pc1(effect_df, n_boot=N_BOOT, seed=RANDOM_SEED):
    X = effect_df.to_numpy(float)
    X = X[np.all(np.isfinite(X), axis=1)]
    n = len(X)
    point = pc1_ratio(X)
    rng = np.random.default_rng(seed)
    boots = np.array([pc1_ratio(X[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), n

def build_day(day, arrays):
    control = arrays["0 ug/L"]
    mu = np.nanmean(control, axis=1, keepdims=True)
    sd = np.nanstd(control, axis=1, keepdims=True)
    sd[(~np.isfinite(sd)) | (sd <= 1e-12)] = np.nan
    rows = []
    for dose in DOSES:
        z = (arrays[dose] - mu) / sd
        for f in FEATURES:
            vals = z[FEATURE_INDEX[f]].astype(float)
            finite = vals[np.isfinite(vals)]
            rows.append({
                "Day": day, "Feature_key": f, "Dose": dose,
                "Mean_control_normalized_z": float(np.mean(finite)) if len(finite) else np.nan,
            })
    summary = pd.DataFrame(rows)
    raw = summary.pivot(index="Feature_key", columns="Dose", values="Mean_control_normalized_z").reindex(index=FEATURE_ORDER, columns=DOSES)
    sign = np.sign(raw["500 ug/L"].astype(float)).where(lambda s: np.isfinite(s) & (s != 0), 1.0)
    oriented = raw.mul(sign, axis=0)
    plot = oriented.copy()
    plot.index = [LABELS[f] for f in plot.index]
    return {"raw": raw, "oriented": oriented, "effect": oriented, "plot": plot, "summary": summary}

RESULTS = {day: build_day(day, arrays) for day, arrays in DAY_DATA.items()}
DAY5_HEAT = RESULTS["Day 5"]["plot"].rename(columns=DISPLAY_DOSES)
DAY10_HEAT = RESULTS["Day 14"]["plot"].rename(columns=DISPLAY_DOSES)
all_abs = np.concatenate([
    np.abs(DAY5_HEAT.to_numpy(float)).ravel(),
    np.abs(DAY10_HEAT.to_numpy(float)).ravel(),
])
all_abs = all_abs[np.isfinite(all_abs)]
COLOR_LIM = max(float(np.nanpercentile(all_abs, 98)), 0.5)

coord_rows = []
for i, day in enumerate(["Day 5", "Day 14"]):
    p, lo, hi, n = bootstrap_pc1(RESULTS[day]["effect"], seed=RANDOM_SEED + i)
    coord_rows.append({"Day": day, "PC1_explained_ratio": p, "CI_low": lo, "CI_high": hi, "N_features_used": n})
COORD_DF = pd.DataFrame(coord_rows)

# ============================================================
# 5. Family-balanced cross-fitted composite
# ============================================================
def row_nanmean(x):
    n = np.sum(np.isfinite(x), axis=1)
    s = np.nansum(x, axis=1)
    out = np.full(x.shape[0], np.nan)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out

def estimate_directions(Xz, dose_num):
    d = np.ones(Xz.shape[1])
    for j in range(Xz.shape[1]):
        y = Xz[:, j]
        m = np.isfinite(y) & np.isfinite(dose_num)
        if m.sum() >= 3 and np.nanstd(y[m]) > 1e-12:
            d[j] = -1.0 if stats.linregress(dose_num[m], y[m]).slope < 0 else 1.0
    return d

def composite_from_aligned(X):
    family_scores = [row_nanmean(X[:, GROUP_IDX[g]]) for g in GROUPS]
    return row_nanmean(np.column_stack(family_scores))

def build_composite(day, arrays):
    X, labels, dose_num, sample_idx = [], [], [], []
    for dose in DOSES:
        arr = arrays[dose].T
        X.append(arr)
        labels.extend([dose] * len(arr))
        dose_num.extend([DOSE_NUM[dose]] * len(arr))
        sample_idx.extend(range(len(arr)))
    X = np.vstack(X).astype(float)
    labels = np.asarray(labels)
    dose_num = np.asarray(dose_num, float)
    sample_idx = np.asarray(sample_idx, int)
    score = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=COMPOSITE_RANDOM_STATE)
    for fold, (tr, te) in enumerate(cv.split(X, labels), 1):
        ctrl = labels[tr] == "0 ug/L"
        mu = np.nanmean(X[tr][ctrl], axis=0)
        sd = np.nanstd(X[tr][ctrl], axis=0)
        sd[(~np.isfinite(sd)) | (sd <= 1e-12)] = np.nan
        Xtr = ((X[tr] - mu) / sd)[:, SELECTED_IDX]
        Xte = ((X[te] - mu) / sd)[:, SELECTED_IDX]
        direction = estimate_directions(Xtr, dose_num[tr])
        score[te] = composite_from_aligned(Xte * direction[None, :])
        fold_id[te] = fold
    return pd.DataFrame({
        "Day": day, "Dose": labels, "Dose_value": dose_num,
        "Composite_score": score, "Sample": sample_idx, "Fold": fold_id,
    })

COMPOSITE_DF = pd.concat([
    build_composite("Day 5", DAY_DATA["Day 5"]),
    build_composite("Day 14", DAY_DATA["Day 14"]),
], ignore_index=True)
COMPOSITE_DF = COMPOSITE_DF[np.isfinite(COMPOSITE_DF["Composite_score"])].copy()

trend_rows = []
for day in ["Day 5", "Day 14"]:
    sub = COMPOSITE_DF[COMPOSITE_DF["Day"] == day]
    rho, p = stats.spearmanr(sub["Dose_value"], sub["Composite_score"])
    trend_rows.append({"Day": day, "Spearman_r": float(rho), "Spearman_p": float(p), "N_images": len(sub)})
TREND_DF = pd.DataFrame(trend_rows)

# ============================================================
# 6. Plot style
# ============================================================
mpl.rcParams.update({
    "font.family": "Arial", "font.size": 13,
    "axes.linewidth": 0.8, "axes.labelsize": 13,
    "axes.titlesize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
})
sns.set_theme(context="notebook", style="white", rc={
    "font.family": "Arial", "font.size": 13, "axes.linewidth": 0.8,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

DOSE_COLORS = {"0 ug/L": "#24478F", "50 ug/L": "#5D8BC9", "200 ug/L": "#E58B00", "500 ug/L": "#E9473F"}

def lighten(c, amount=0.62):
    x = np.asarray(to_rgb(c))
    return to_hex(x + (np.ones(3) - x) * amount)
BOX_COLORS = {k: lighten(v) for k, v in DOSE_COLORS.items()}

# ============================================================
# 7. Figure layout
# ============================================================
fig = plt.figure(figsize=(24.0, 16.5), facecolor="white")
gs = fig.add_gridspec(
    3, 2,
    width_ratios=[1, 1],
    height_ratios=[1.42, 0.055, 0.78],
    left=0.16, right=0.985, top=0.95, bottom=0.09,
    wspace=0.28, hspace=0.42,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
cax = fig.add_subplot(gs[1, :])
ax_c = fig.add_subplot(gs[2, 0])
gs_d = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs[2, 1],
    height_ratios=[0.20, 0.80], wspace=0.38, hspace=0.05
)
ax_d_header = fig.add_subplot(gs_d[0, :])
ax_d_header.axis("off")
ax_d_header.text(0.00, 0.55, "d", ha="left", va="center", fontsize=15, fontweight="bold")
ax_d_header.text(0.12, 0.55, "Family-balanced composite phenotype",
                 ha="left", va="center", fontsize=14, fontweight="bold")
ax_d5 = fig.add_subplot(gs_d[1, 0])
ax_d10 = fig.add_subplot(gs_d[1, 1], sharey=ax_d5)

# a
sns.heatmap(
    DAY5_HEAT, ax=ax_a, cmap="vlag", center=0,
    vmin=-COLOR_LIM, vmax=COLOR_LIM,
    linewidths=0.4, linecolor="white",
    annot=annot_df(DAY5_HEAT), fmt="",
    annot_kws={"fontsize": 12, "color": "black"},
    cbar=False,
)
ax_a.set_title("Day 5", fontweight="bold", pad=6)
ax_a.set_xlabel(""); ax_a.set_ylabel("")
ax_a.tick_params(axis="x", rotation=0, length=0, labelsize=12)
ax_a.tick_params(axis="y", rotation=0, length=0, labelsize=12)

# b
hm = sns.heatmap(
    DAY10_HEAT, ax=ax_b, cmap="vlag", center=0,
    vmin=-COLOR_LIM, vmax=COLOR_LIM,
    linewidths=0.4, linecolor="white",
    annot=annot_df(DAY10_HEAT), fmt="",
    annot_kws={"fontsize": 12, "color": "black"},
    cbar=True, cbar_ax=cax, cbar_kws={"orientation": "horizontal"},
)
ax_b.set_title("Day 14", fontweight="bold", pad=6)
ax_b.set_xlabel(""); ax_b.set_ylabel("")
ax_b.tick_params(axis="x", rotation=0, length=0, labelsize=12)
ax_b.tick_params(axis="y", left=False, labelleft=False, length=0)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(axis="x", labelsize=12, length=2.5, width=0.7)
cbar.set_label("Mean oriented standardized shift", fontsize=13, labelpad=3)
cbar.outline.set_linewidth(0.6)

# c
coord = COORD_DF.set_index("Day").loc[["Day 5", "Day 14"]]
x = np.arange(2)
y = coord["PC1_explained_ratio"].to_numpy(float)
lo = coord["CI_low"].to_numpy(float)
hi = coord["CI_high"].to_numpy(float)
for i, color in enumerate(["#24478F", "#E9473F"]):
    ax_c.errorbar(
        x[i], y[i], yerr=[[y[i]-lo[i]], [hi[i]-y[i]]],
        fmt="o", ms=7.5, mfc=color, mec=color, mew=0.8,
        ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9,
    )
ax_c.set_title("Phenotype coordination", fontweight="bold", pad=6)
ax_c.set_xlim(-0.45, 1.45); ax_c.set_ylim(0, 1.04)
ax_c.set_xticks(x, ["Day 5", "Day 14"])
ax_c.set_yticks(np.linspace(0, 1, 6))
ax_c.set_ylabel("PC1 variance\nexplained")
ax_c.tick_params(axis="x", length=0)
ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)

# d
def draw_composite(ax, day, show_ylabel):
    sub = COMPOSITE_DF[COMPOSITE_DF["Day"] == day]
    rng = np.random.default_rng(500 if day == "Day 5" else 501)
    medians = []
    for i, dose in enumerate(DOSES):
        vals = sub.loc[sub["Dose"] == dose, "Composite_score"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        medians.append(np.nanmedian(vals))
        ax.boxplot(
            vals, positions=[i], widths=0.48, patch_artist=True, showfliers=False,
            boxprops={"facecolor": BOX_COLORS[dose], "edgecolor": DOSE_COLORS[dose], "linewidth": 0.8},
            medianprops={"color": DOSE_COLORS[dose], "linewidth": 1.1},
            whiskerprops={"color": DOSE_COLORS[dose], "linewidth": 0.75},
            capprops={"color": DOSE_COLORS[dose], "linewidth": 0.75},
        )
        choose = rng.choice(len(vals), size=min(MAX_POINTS_PER_DOSE, len(vals)), replace=False)
        jitter = rng.normal(0, 0.055, size=len(choose))
        ax.scatter(i + jitter, vals[choose], s=7, color=DOSE_COLORS[dose], alpha=0.50, linewidth=0, zorder=3)
    ax.plot(np.arange(4), medians, color="#222222", lw=1.0, marker="o", ms=2.8, zorder=4)
    ax.axhline(0, color="#AEB6C2", lw=0.7, ls=(0, (2.2, 2.2)), zorder=0)
    rho = TREND_DF.loc[TREND_DF["Day"] == day, "Spearman_r"].iloc[0]
    ax.set_title(day, fontsize=14, pad=1, y=0.97)
    ax.text(
        0.03, 0.97, r"$r_s$" + f" = {rho:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=12
    )
    ax.set_xticks(np.arange(4), ["0", "50", "200", "500"])
    ax.set_xlabel("Exposure (µg/L)", fontsize=13, labelpad=3)
    if show_ylabel:
        ax.set_ylabel("Composite phenotype\nscore", fontsize=13, labelpad=4)
    ax.tick_params(axis="both", labelsize=12, direction="out", length=2.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

all_vals = COMPOSITE_DF["Composite_score"].to_numpy(float)
q1, q99 = np.nanpercentile(all_vals, [1, 99])
pad = 0.15 * (q99 - q1 + 1e-8)
draw_composite(ax_d5, "Day 5", True)
draw_composite(ax_d10, "Day 14", False)
ax_d5.set_ylim(q1-pad, q99+pad); ax_d10.set_ylim(q1-pad, q99+pad)
ax_d10.tick_params(axis="y", labelleft=False)

ax_a.text(-0.30, 1.035, "a", transform=ax_a.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax_b.text(-0.08, 1.035, "b", transform=ax_b.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax_c.text(-0.12, 1.04, "c", transform=ax_c.transAxes, fontsize=15, fontweight="bold", va="bottom")

# ============================================================
# 8. Save
# ============================================================

BASE = OUT_DIR / "Fig_day5_day14_heatmaps_coordination_composite"
if SAVE_FIGURES:
    fig.savefig(BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

plt.show()

if SAVE_TABLES:
    COORD_DF.to_csv(OUT_DIR / "phenotype_coordination_pc1_explained_ratio.csv", index=False)
    COMPOSITE_DF.to_csv(OUT_DIR / "day5_day14_family_balanced_composite_scores.csv", index=False)
    TREND_DF.to_csv(OUT_DIR / "day5_day14_family_balanced_composite_trend.csv", index=False)
    RESULTS["Day 5"]["oriented"].to_csv(OUT_DIR / "day5_16feature_heatmap_matrix_oriented.csv")
    RESULTS["Day 14"]["oriented"].to_csv(OUT_DIR / "day14_16feature_heatmap_matrix_oriented.csv")
    comp_summary = COMPOSITE_DF.groupby(["Day", "Dose", "Dose_value"])["Composite_score"].agg(N="count", Mean="mean", SD="std", Median="median").reset_index()
    with pd.ExcelWriter(OUT_DIR / "Fig_day5_day14_heatmaps_coordination_results.xlsx") as xw:
        RESULTS["Day 5"]["oriented"].to_excel(xw, sheet_name="D5_oriented"); RESULTS["Day 14"]["oriented"].to_excel(xw, sheet_name="D14_oriented")
        RESULTS["Day 5"]["raw"].to_excel(xw, sheet_name="D5_raw"); RESULTS["Day 14"]["raw"].to_excel(xw, sheet_name="D14_raw")
        COORD_DF.to_excel(xw, sheet_name="coordination", index=False); TREND_DF.to_excel(xw, sheet_name="composite_trend", index=False)
        comp_summary.to_excel(xw, sheet_name="composite_summary", index=False); COMPOSITE_DF.to_excel(xw, sheet_name="composite_scores", index=False)

print("\nPhenotype coordination:")
print(COORD_DF.to_string(index=False))
print("\nComposite phenotype trend:")
print(TREND_DF.to_string(index=False))
print("\nOutput directory:", OUT_DIR.resolve())