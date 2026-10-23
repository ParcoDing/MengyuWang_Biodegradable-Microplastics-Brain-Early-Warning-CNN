# -*- coding: utf-8 -*-
# Time labels harmonized to Day 5 / Day 14 following author confirmation.
"""
Endpoint-trained recovery of unseen intermediate exposure states — revised publication version

Changes in this revision
------------------------
1) Panel c numeric labels are all placed above points to avoid clipping.
2) Panel a is redrawn in the earlier horizontal mockup style, using scatter markers
   so the dose nodes remain circular rather than flattened.
3) The "held-out" text in panel b is removed; the grey background remains.
4) In panel a, 50 and 200 are placed between 0 and 500 inside the dashed held-out box.

Main-text methods:
1) Endpoint-defined Composite
2) Explicit-feature Linear ML
3) CNN
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# ============================================================
# 1. Settings
# ============================================================
FEATURE_DIR = Path("./data/features")
CNN_DIR = Path("./results/cnn")
OUT_DIR = Path("./results/figures")
TABLE_DIR = Path("./results/cnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS, RANDOM_STATE, VIEWS_PER_FISH = 5, 42, 4
N_BOOT, N_SCATTER = 2000, 35
SAVE_FIGURE = True

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
FEATURE_GROUPS = {
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
DAYS = ["Day 5", "Day 14"]
DISPLAY_DAY = {"Day 5": "Day 5", "Day 14": "Day 14"}
DOSES = ["0 ug/L", "50 ug/L", "200 ug/L", "500 ug/L"]
DISPLAY_DOSE = {"0 ug/L": "0 µg/L", "50 ug/L": "50 µg/L", "200 ug/L": "200 µg/L", "500 ug/L": "500 µg/L"}
DOSE_NUM = {"0 ug/L": 0., "50 ug/L": 50., "200 ug/L": 200., "500 ug/L": 500.}
DOSE_COLOR = {"0 ug/L": "#173B73", "50 ug/L": "#5D8BC9", "200 ug/L": "#E58B00", "500 ug/L": "#D9342B"}
DAY_COLOR = {"Day 5": "#2454D3", "Day 14": "#D9342B"}

# ============================================================
# 2. Load classical features
# ============================================================
saved_names = np.load(FEATURE_DIR / "feature_names.npy", allow_pickle=False).astype(str).tolist()
missing = [f for f in FEATURES if f not in saved_names]
if missing:
    raise ValueError("Missing explicit features:\n" + "\n".join(missing))
fidx = np.array([saved_names.index(f) for f in FEATURES])
fpos = {f: i for i, f in enumerate(FEATURES)}
gidx = {g: np.array([fpos[f] for f in fs]) for g, fs in FEATURE_GROUPS.items()}

def load_feature(filename):
    a = np.load(FEATURE_DIR / filename)
    if a.ndim != 2 or a.shape[0] != len(saved_names):
        raise ValueError(f"{filename}: unexpected shape {a.shape}")
    if a.shape[1] % VIEWS_PER_FISH:
        raise ValueError(f"{filename}: image count must be divisible by {VIEWS_PER_FISH}")
    return a[fidx].T.astype(float)

explicit = {
    "Day 5": {
        "0 ug/L": load_feature("day5_0CT_arr.npy"),
        "50 ug/L": load_feature("day5_50ppb_arr.npy"),
        "200 ug/L": load_feature("day5_200ppb_arr.npy"),
        "500 ug/L": load_feature("day5_500ppb_arr.npy"),
    },
    "Day 14": {
        "0 ug/L": load_feature("day14_0CT_arr.npy"),
        "50 ug/L": load_feature("day14_50ppb_arr.npy"),
        "200 ug/L": load_feature("day14_200ppb_arr.npy"),
        "500 ug/L": load_feature("day14_500ppb_arr.npy"),
    },
}

# ============================================================
# 3. Utilities
# ============================================================
def row_nanmean(x):
    n, s = np.sum(np.isfinite(x), axis=1), np.nansum(x, axis=1)
    out = np.full(x.shape[0], np.nan)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out

def composite_score(x):
    family = np.column_stack([row_nanmean(x[:, gidx[g]]) for g in FEATURE_GROUPS])
    return row_nanmean(family)

def endpoint_groups(n0, n500):
    g0 = np.repeat(np.arange(n0 // VIEWS_PER_FISH), VIEWS_PER_FISH)
    off = g0.max() + 1
    g5 = np.repeat(np.arange(off, off + n500 // VIEWS_PER_FISH), VIEWS_PER_FISH)
    return np.r_[g0, g5]

def grouped_splits(y, groups):
    tab = pd.DataFrame({"g": groups, "y": y}).drop_duplicates("g")
    ug, gy = tab["g"].to_numpy(), tab["y"].to_numpy()
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(ug, gy):
        yield np.isin(groups, ug[tr]), np.isin(groups, ug[te])

def control_stats(X0):
    mu, sd = np.nanmean(X0, axis=0), np.nanstd(X0, axis=0)
    sd[(~np.isfinite(sd)) | (sd <= 1e-12)] = np.nan
    return mu, sd

def fish_means(X, groups):
    d = pd.DataFrame(X, columns=FEATURES)
    d["g"] = groups
    return d.groupby("g")[FEATURES].mean().to_numpy(float)

def endpoint_direction(X0z, X500z, g0, g500):
    delta = np.nanmean(fish_means(X500z, g500), axis=0) - np.nanmean(fish_means(X0z, g0), axis=0)
    return np.where(np.isfinite(delta) & (delta < 0), -1., 1.)

def control_normalize_scores(s):
    mu, sd = np.nanmean(s["0 ug/L"]), np.nanstd(s["0 ug/L"])
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.
    return {d: (np.asarray(v, float) - mu) / sd for d, v in s.items()}

def pooled_spearman(score_dict):
    x, y = [], []
    for dose in DOSES:
        v = np.asarray(score_dict[dose], float)
        m = np.isfinite(v)
        x.extend([DOSE_NUM[dose]] * int(m.sum()))
        y.extend(v[m])
    return stats.spearmanr(np.asarray(x, float), np.asarray(y, float))

def bootstrap_spearman_by_fish(score_dict, n_boot=N_BOOT, seed=123):
    rng, boots = np.random.default_rng(seed), []
    fish_arrays = {}
    for dose in DOSES:
        v = np.asarray(score_dict[dose], float)
        if len(v) % VIEWS_PER_FISH:
            raise ValueError(f"{dose}: score count is not divisible by {VIEWS_PER_FISH}")
        fish_arrays[dose] = v.reshape(-1, VIEWS_PER_FISH)
    for _ in range(n_boot):
        x, y = [], []
        for dose in DOSES:
            a = fish_arrays[dose]
            sampled = a[rng.integers(0, len(a), len(a))].reshape(-1)
            m = np.isfinite(sampled)
            x.extend([DOSE_NUM[dose]] * int(m.sum()))
            y.extend(sampled[m])
        boots.append(stats.spearmanr(np.asarray(x), np.asarray(y)).statistic)
    rho = pooled_spearman(score_dict).statistic
    lo, hi = np.nanpercentile(np.asarray(boots, float), [2.5, 97.5])
    return float(rho), float(lo), float(hi)

# ============================================================
# 4. Endpoint-defined Composite
# ============================================================
def endpoint_composite(day):
    X0, X50, X200, X500 = [explicit[day][d] for d in DOSES]
    Xep = np.vstack([X0, X500])
    y = np.r_[np.zeros(len(X0), int), np.ones(len(X500), int)]
    groups = endpoint_groups(len(X0), len(X500))
    oof, pred50, pred200 = np.full(len(Xep), np.nan), [], []
    for tr, te in grouped_splits(y, groups):
        tr0, tr5 = tr & (y == 0), tr & (y == 1)
        mu, sd = control_stats(Xep[tr0])
        X0z, X5z = (Xep[tr0] - mu) / sd, (Xep[tr5] - mu) / sd
        direction = endpoint_direction(X0z, X5z, groups[tr0], groups[tr5])
        oof[te] = composite_score(((Xep[te] - mu) / sd) * direction)
        pred50.append(composite_score(((X50 - mu) / sd) * direction))
        pred200.append(composite_score(((X200 - mu) / sd) * direction))
    return control_normalize_scores({
        "0 ug/L": oof[:len(X0)],
        "50 ug/L": np.nanmean(np.vstack(pred50), axis=0),
        "200 ug/L": np.nanmean(np.vstack(pred200), axis=0),
        "500 ug/L": oof[len(X0):],
    })

# ============================================================
# 5. Linear ML
# ============================================================
linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced", solver="liblinear",
        max_iter=5000, random_state=RANDOM_STATE,
    )),
])

def endpoint_linear(day):
    X0, X50, X200, X500 = [explicit[day][d] for d in DOSES]
    Xep = np.vstack([X0, X500])
    y = np.r_[np.zeros(len(X0), int), np.ones(len(X500), int)]
    groups = endpoint_groups(len(X0), len(X500))
    oof, pred50, pred200 = np.full(len(Xep), np.nan), [], []
    for tr, te in grouped_splits(y, groups):
        model = clone(linear_model).fit(Xep[tr], y[tr])
        oof[te] = model.decision_function(Xep[te])
        pred50.append(model.decision_function(X50))
        pred200.append(model.decision_function(X200))
    return control_normalize_scores({
        "0 ug/L": oof[:len(X0)],
        "50 ug/L": np.nanmean(np.vstack(pred50), axis=0),
        "200 ug/L": np.nanmean(np.vstack(pred200), axis=0),
        "500 ug/L": oof[len(X0):],
    })

# ============================================================
# 6. CNN scores
# ============================================================
def cnn_path(day, dose):
    prefix = "day5" if day == "Day 5" else "day14"
    dkey = {"0 ug/L": "0CT", "50 ug/L": "50ppb", "200 ug/L": "200ppb", "500 ug/L": "500ppb"}[dose]
    candidates = [
        CNN_DIR / f"{prefix}_{dkey}_deep_learning_logit.npy",
        CNN_DIR / f"{prefix}_{dkey}_deep_learning_logit_oof.npy",
        CNN_DIR / f"{prefix}_{dkey}_deep_learning_ensemble_logit.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("CNN score not found:\n" + "\n".join(map(str, candidates)))

def load_cnn(day):
    out = {}
    print(f"\n{day} CNN files:")
    for dose in DOSES:
        p = cnn_path(day, dose)
        out[dose] = np.load(p).astype(float).ravel()
        if len(out[dose]) != len(explicit[day][dose]):
            raise ValueError(f"{day} {dose}: CNN score count does not match feature image count.")
        print(f"{dose:>8}: {p}")
    return control_normalize_scores(out)

# ============================================================
# 7. Compute scores and bootstrap ordinal recovery
# ============================================================
scores = {"Composite": {}, "Linear ML": {}, "CNN": {}}
for day in DAYS:
    scores["Composite"][day] = endpoint_composite(day)
    scores["Linear ML"][day] = endpoint_linear(day)
    scores["CNN"][day] = load_cnn(day)

summary = []
for mi, method in enumerate(scores):
    for di, day in enumerate(DAYS):
        rho, lo, hi = bootstrap_spearman_by_fish(scores[method][day], seed=1000 + 10 * mi + di)
        means = {d: np.nanmean(scores[method][day][d]) for d in DOSES}
        summary.append({
            "Method": method, "Day": day, "Spearman_r": rho, "CI_low": lo, "CI_high": hi,
            **{f"Mean_{d.replace(' ', '_')}": means[d] for d in DOSES},
        })
summary_df = pd.DataFrame(summary)
summary_df.to_csv(TABLE_DIR / "endpoint_defined_unseen_dose_recovery_summary.csv", index=False)

long_rows = []
for method in scores:
    for day in DAYS:
        for dose in DOSES:
            for i, v in enumerate(np.asarray(scores[method][day][dose], float)):
                if np.isfinite(v):
                    long_rows.append({
                        "Method": method, "Day": day, "Dose": dose,
                        "Dose_value": DOSE_NUM[dose], "Image_index": i, "Score_z": v,
                    })
long_df = pd.DataFrame(long_rows)
long_df.to_csv(TABLE_DIR / "endpoint_defined_unseen_dose_recovery_scores.csv", index=False)
fish_rows = []
for method in scores:
    for day in DAYS:
        for dose in DOSES:
            v = np.asarray(scores[method][day][dose], float).reshape(-1, VIEWS_PER_FISH)
            fish_rows += [{"Method": method, "Day": day, "Dose": dose, "Dose_value": DOSE_NUM[dose], "Fish_index": i, "Fish_score": x}
                          for i, x in enumerate(np.nanmean(v, axis=1))]
fish_df = pd.DataFrame(fish_rows)
dose_summary = fish_df.groupby(["Method", "Day", "Dose", "Dose_value"])["Fish_score"].agg(N="count", Mean="mean", SD="std", Median="median").reset_index()
with pd.ExcelWriter(TABLE_DIR / "Fig_unseen_intermediate_dose_recovery_results.xlsx") as xw:
    summary_df.to_excel(xw, sheet_name="ordinal_summary", index=False); dose_summary.to_excel(xw, sheet_name="dose_summary", index=False)
    fish_df.to_excel(xw, sheet_name="fish_scores", index=False); long_df.to_excel(xw, sheet_name="image_scores", index=False)

# ============================================================
# 8. Publication style
# ============================================================
mpl.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.linewidth": 0.8,
    "axes.labelsize": 13, "axes.titlesize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
sns.set_theme(context="notebook", style="white", rc={
    "font.family": "Arial", "font.size": 13, "axes.linewidth": 0.8,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

fig = plt.figure(figsize=(19.0, 13.0))
gs = fig.add_gridspec(
    3, 4, height_ratios=[0.62, 1, 1], width_ratios=[1, 1, 1, 1.08],
    left=0.075, right=0.988, top=0.955, bottom=0.10, wspace=0.40, hspace=0.48,
)
ax_a = fig.add_subplot(gs[0, :])
axes = {(day, method): fig.add_subplot(gs[1 + DAYS.index(day), ["Composite", "Linear ML", "CNN"].index(method)])
        for day in DAYS for method in ["Composite", "Linear ML", "CNN"]}
ax_c = fig.add_subplot(gs[1:, 3])

# ============================================================
# 9. Panel a — previous horizontal mockup style
# ============================================================
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off")

# left sequence with held-out block in the middle
xL = [0.08, 0.18, 0.29, 0.39]
y_node = 0.66
held_x0, held_w = 0.135, 0.19
held_y0, held_h = 0.44, 0.34
ax_a.add_patch(FancyBboxPatch((held_x0, held_y0), held_w, held_h, boxstyle="round,pad=0.012",
                              ec="#8B8B8B", fc="#F7F7F7", lw=1.0, ls=(0, (4, 3))))
ax_a.text(held_x0 + held_w / 2, 0.84, "Held-out during training", ha="center", va="bottom",
          fontsize=13, fontweight="bold")

# scatter circles stay circular regardless of axis aspect
for x, dose in zip(xL, DOSES):
    ax_a.scatter([x], [y_node], s=2500, color=DOSE_COLOR[dose], zorder=3)
    ax_a.text(x, y_node, DISPLAY_DOSE[dose].replace(" ", "\n"), color="white", ha="center", va="center",
              fontsize=13, fontweight="bold")
for xa, xb in zip(xL[:-1], xL[1:]):
    ax_a.annotate("", xy=(xb - 0.025, y_node), xytext=(xa + 0.025, y_node),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))

# bracket-like top input from 0 and 500 to training box
ax_a.plot([xL[0], xL[0], xL[3], xL[3]], [0.58, 0.86, 0.86, 0.58], color="black", lw=1.2)
ax_a.annotate("", xy=(xL[3], 0.58), xytext=(xL[3], 0.86),
              arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))

# endpoint training box
train_box = FancyBboxPatch((0.46, 0.49), 0.15, 0.22, boxstyle="round,pad=0.012",
                           ec="black", fc="white", lw=1.0)
ax_a.add_patch(train_box)
ax_a.text(0.535, 0.60, "Endpoint\ntraining", ha="center", va="center",
          fontsize=13, fontweight="bold")
ax_a.annotate("", xy=(0.46, 0.60), xytext=(xL[3] + 0.03, 0.60),
              arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))

# right recovered ordering
ax_a.text(0.79, 0.79, "Recovered exposure ordering", ha="center", va="center",
          fontsize=13, fontweight="bold")
xR = [0.66, 0.75, 0.84, 0.93]
for x, dose in zip(xR, DOSES):
    ax_a.scatter([x], [0.60], s=2100, color=DOSE_COLOR[dose], zorder=3)
    ax_a.text(x, 0.60, DISPLAY_DOSE[dose].replace(" ", "\n"), color="white", ha="center", va="center",
              fontsize=13, fontweight="bold")
for xa, xb in zip(xR[:-1], xR[1:]):
    ax_a.annotate("", xy=(xb - 0.028, 0.60), xytext=(xa + 0.028, 0.60),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))

ax_a.text(-0.035, 1.02, "a", transform=ax_a.transAxes, fontsize=15, fontweight="bold", va="top")

# ============================================================
# 10. Panel b — distributions + median trajectories
# ============================================================
title = {"Composite": "Explicit phenotype axis", "Linear ML": "Feature ML", "CNN": "CNN"}

def draw_score(ax, method, day, ylabel=False):
    rng = np.random.default_rng(1234 + 100 * DAYS.index(day) + 10 * list(scores).index(method))
    ax.axvspan(0.55, 2.45, color="#F3F3F3", zorder=0)
    med = []
    for i, dose in enumerate(DOSES):
        v = np.asarray(scores[method][day][dose], float)
        v = v[np.isfinite(v)]
        med.append(np.nanmedian(v))
        ax.boxplot(
            v, positions=[i], widths=0.44, showfliers=False, patch_artist=True,
            boxprops=dict(fc="white", ec=DOSE_COLOR[dose], lw=1.0),
            medianprops=dict(color=DOSE_COLOR[dose], lw=1.2),
            whiskerprops=dict(color=DOSE_COLOR[dose], lw=0.8),
            capprops=dict(color=DOSE_COLOR[dose], lw=0.8),
        )
        k = min(N_SCATTER, len(v))
        pick = rng.choice(len(v), k, replace=False) if len(v) > k else np.arange(len(v))
        ax.scatter(i + rng.normal(0, 0.05, len(pick)), v[pick], s=5.5,
                   color=DOSE_COLOR[dose], alpha=0.48, lw=0, zorder=2)
    ax.plot(range(4), med, color="#222222", lw=1.15, marker="o", ms=3.0, zorder=3)
    ax.axhline(0, color="#B8B8B8", lw=0.7, ls=(0, (2.2, 2.2)))
    ax.set_xticks(range(4), ["0", "50", "200", "500"])
    ax.set_xlabel("µg/L")
    ax.set_ylabel("Exposure-associated\nscore" if ylabel else "")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5)
    if day == "Day 5":
        ax.set_title(title[method], fontweight="bold", pad=6)

for method in ["Composite", "Linear ML", "CNN"]:
    vals = np.concatenate([np.asarray(scores[method][day][dose], float) for day in DAYS for dose in DOSES])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.nanpercentile(vals, [1, 99]); pad = 0.16 * (hi - lo + 1e-8)
    for day in DAYS:
        draw_score(axes[(day, method)], method, day, ylabel=(method == "Composite"))
        axes[(day, method)].set_ylim(lo - pad, hi + pad)

axes[("Day 5", "Composite")].text(
    -0.425, 0.5, "Day 5", transform=axes[("Day 5", "Composite")].transAxes,
    ha="center", va="center", rotation=90, fontsize=13, fontweight="bold")
axes[("Day 14", "Composite")].text(
    -0.425, 0.5, "Day 14", transform=axes[("Day 14", "Composite")].transAxes,
    ha="center", va="center", rotation=90, fontsize=13, fontweight="bold")
axes[("Day 5", "Composite")].text(
    -0.29, 1.18, "b", transform=axes[("Day 5", "Composite")].transAxes,
    fontsize=15, fontweight="bold", va="top")

# ============================================================
# 11. Panel c — independent dots + fish-bootstrap 95% CI
# ============================================================
methods = ["Composite", "Linear ML", "CNN"]
labels = ["Explicit axis", "Feature ML", "CNN"]
x = np.arange(3)
offset = {"Day 5": -0.10, "Day 14": 0.10}

for day in DAYS:
    sub = summary_df[summary_df["Day"] == day].set_index("Method").loc[methods]
    xx = x + offset[day]
    yy = sub["Spearman_r"].to_numpy(float)
    lo = sub["CI_low"].to_numpy(float)
    hi = sub["CI_high"].to_numpy(float)
    yerr = np.vstack([yy - lo, hi - yy])
    ax_c.errorbar(xx, yy, yerr=yerr, fmt="o", ms=6.0, capsize=3, capthick=0.9,
                  elinewidth=0.9, color=DAY_COLOR[day], mfc=DAY_COLOR[day],
                  mec=DAY_COLOR[day], label=DISPLAY_DAY[day], zorder=3)
    for xi, yi in zip(xx, yy):
        ax_c.text(xi+0.2, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom",
                  fontsize=12, color=DAY_COLOR[day], clip_on=False)

ax_c.axhline(0, color="#9F9F9F", lw=0.8, ls=(0, (3, 3)))
ax_c.set_xticks(x, labels, rotation=15)
ax_c.set_ylabel(r"Spearman $r_s$")
ax_c.set_title("Ordinal recovery", fontweight="bold", pad=6)
ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
ax_c.tick_params(direction="out", length=3)
ax_c.legend(frameon=False, loc="lower right")
ax_c.set_ylim(-0.20, 1.02)
ax_c.set_xlim(-0.45, 2.35)
ax_c.text(-0.22, 1.04, "c", transform=ax_c.transAxes, fontsize=15, fontweight="bold", va="top")

# ============================================================
# 12. Save
# ============================================================
base = OUT_DIR / "Fig_unseen_intermediate_dose_recovery_publication_v2"
if SAVE_FIGURE:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

plt.show()
# plt.close(fig)

print("\nSummary with fish-bootstrap 95% CI:")
print(summary_df.to_string(index=False))
print("\nFigure:", base.with_suffix(".png").resolve())