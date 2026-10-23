# -*- coding: utf-8 -*-
# Time labels harmonized to Day 5 / Day 14 following author confirmation.
"""
Day 5 ↔ Day 14 results/cross_time — 4-panel publication figure
All results are recalculated directly from:
    DATA_DIR = Path("./data/features")

a  Transfer summary matrix:
   Composite direction transfer + Linear-ML direct transfer (endpoint AUROC)
b  Dose-ordering transfer:
   Spearman r_s, filled=within-time, open=cross-time
c  Day 5 phenotype axis -> Day 14 data:
   four-dose transferred COMPOSITE-score distributions
d  Day 14 phenotype axis -> Day 5 data:
   four-dose transferred COMPOSITE-score distributions

Important
---------
- Final scores/metrics are image-level.
- Fish ID is used only to keep four views from the same fish in the same CV fold.
- Composite cross-time analysis transfers source-day feature directions only;
  target-day 0 ug/L is used for target-day z-score normalization.
- Linear ML cross-time analysis transfers the complete source-day pipeline
  (imputer + scaler + logistic classifier) without target-day refitting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 1. Settings
# ============================================================
DATA_DIR = Path('./data/features')
RESULT_DIR = Path('./results/cross_time')
FIG_DIR = Path("./results/figures")
Path("./results/figures")
RESULT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS, RANDOM_STATE, VIEWS_PER_FISH = 5, 42, 4
SAVE_FIGURE, SAVE_TABLES = True, True

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
    "Spatial": [
        "nearest_neighbor_distance", "local_density_variance",
        "spatial_heterogeneity_increase", "green_neighbor_degree",
        "green_close_neighbor_fraction", "green_dilation_connectivity",
    ],
    "Channel": ["channel_colocalization"],
    "Texture": [
        "texture_glcm_contrast_std", "texture_glcm_energy_std",
        "texture_glcm_dissimilarity_std", "texture_glcm_correlation_std",
        "texture_glcm_homogeneity_mean",
    ],
}
DAYS = ["Day 5", "Day 14"]
DISPLAY_DAY = {"Day 5": "Day 5", "Day 14": "Day 14"}
DISPLAY_DAYS = [DISPLAY_DAY[d] for d in DAYS]
DOSES = ["0 ug/L", "50 ug/L", "200 ug/L", "500 ug/L"]
DISPLAY_DOSE = {"0 ug/L": "0 µg/L", "50 ug/L": "50 µg/L", "200 ug/L": "200 µg/L", "500 ug/L": "500 µg/L"}
DOSE_VALUE = {"0 ug/L": 0., "50 ug/L": 50., "200 ug/L": 200., "500 ug/L": 500.}
DAY_COLOR = {"Day 5": "#24478F", "Day 14": "#E9473F"}
DOSE_COLOR = {"0 ug/L": "#3A6073", "50 ug/L": "#6494AA", "200 ug/L": "#DDA15E", "500 ug/L": "#A64B4B"}

# ============================================================
# 2. Load raw arrays
# ============================================================
saved_names = np.load(DATA_DIR / "feature_names.npy", allow_pickle=False).astype(str).tolist()
missing = [f for f in FEATURES if f not in saved_names]
if missing: raise ValueError("feature_names.npy missing:\n" + "\n".join(missing))

idx = np.array([saved_names.index(f) for f in FEATURES], int)
fpos = {f: i for i, f in enumerate(FEATURES)}
gidx = {g: np.array([fpos[f] for f in fs], int) for g, fs in FEATURE_GROUPS.items()}

def load_selected(filename):
    a = np.load(DATA_DIR / filename)
    if a.ndim != 2 or a.shape[0] != len(saved_names):
        raise ValueError(f"{filename}: unexpected shape {a.shape}")
    if a.shape[1] % VIEWS_PER_FISH:
        raise ValueError(f"{filename}: image number {a.shape[1]} is not divisible by {VIEWS_PER_FISH}")
    return a[idx].T.astype(float)

RAW = {
    "Day 5": {
        "0 ug/L": load_selected("day5_0CT_arr.npy"),
        "50 ug/L": load_selected("day5_50ppb_arr.npy"),
        "200 ug/L": load_selected("day5_200ppb_arr.npy"),
        "500 ug/L": load_selected("day5_500ppb_arr.npy"),
    },
    "Day 14": {
        "0 ug/L": load_selected("day14_0CT_arr.npy"),
        "50 ug/L": load_selected("day14_50ppb_arr.npy"),
        "200 ug/L": load_selected("day14_200ppb_arr.npy"),
        "500 ug/L": load_selected("day14_500ppb_arr.npy"),
    },
}

def build_day(day):
    Xs, doses, dnum, groups, fish, views, offset = [], [], [], [], [], [], 0
    for dose in DOSES:
        X = RAW[day][dose]; n_img, n_fish = len(X), len(X) // VIEWS_PER_FISH
        Xs.append(X); doses += [dose] * n_img; dnum += [DOSE_VALUE[dose]] * n_img
        groups += list(np.repeat(np.arange(offset, offset + n_fish), VIEWS_PER_FISH))
        fish += list(np.repeat(np.arange(n_fish), VIEWS_PER_FISH))
        views += list(np.tile(np.arange(1, VIEWS_PER_FISH + 1), n_fish)); offset += n_fish
    meta = pd.DataFrame({
        "Day": day, "Dose": doses, "Dose_value": dnum, "Group_id": groups,
        "Fish_index_within_dose": fish, "View_index": views,
    })
    return np.vstack(Xs), meta

DAYSETS = {d: dict(zip(["X", "meta"], build_day(d))) for d in DAYS}

# ============================================================
# 3. Utilities
# ============================================================
def row_nanmean(x):
    n, s = np.sum(np.isfinite(x), axis=1), np.nansum(x, axis=1)
    out = np.full(x.shape[0], np.nan); out[n > 0] = s[n > 0] / n[n > 0]
    return out

def composite_score(aligned):
    family = np.column_stack([row_nanmean(aligned[:, gidx[g]]) for g in FEATURE_GROUPS])
    return row_nanmean(family)

def grouped_split(meta, stratify_col):
    gt = meta[["Group_id", stratify_col]].drop_duplicates("Group_id").sort_values("Group_id")
    groups, labels = gt["Group_id"].to_numpy(), gt[stratify_col].to_numpy()
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(cv.split(groups, labels), 1):
        yield fold, meta["Group_id"].isin(groups[tr]).to_numpy(), meta["Group_id"].isin(groups[te]).to_numpy()

def control_standardize(X_ref, dose_ref, X_target):
    control = X_ref[np.asarray(dose_ref) == "0 ug/L"]
    mu, sd = np.nanmean(control, axis=0), np.nanstd(control, axis=0)
    sd[(~np.isfinite(sd)) | (sd <= 1e-12)] = np.nan
    return (X_target - mu) / sd, mu, sd

def estimate_directions(Xz, meta):
    df = pd.DataFrame(Xz, columns=FEATURES)
    df["Group_id"], df["Dose_value"] = meta["Group_id"].to_numpy(), meta["Dose_value"].to_numpy(float)
    fish = df.groupby(["Group_id", "Dose_value"], as_index=False)[FEATURES].mean()
    dose, out = fish["Dose_value"].to_numpy(float), np.ones(len(FEATURES))
    for j, f in enumerate(FEATURES):
        y = fish[f].to_numpy(float); valid = np.isfinite(y) & np.isfinite(dose)
        if valid.sum() >= 3 and np.nanstd(y[valid]) > 1e-12:
            out[j] = -1. if stats.linregress(dose[valid], y[valid]).slope < 0 else 1.
    return out

def endpoint_metrics(meta, score):
    m = meta["Dose"].isin(["0 ug/L", "500 ug/L"]).to_numpy()
    y, s = meta.loc[m, "Dose"].eq("500 ug/L").astype(int).to_numpy(), np.asarray(score, float)[m]
    v = np.isfinite(s)
    return {"AUROC": roc_auc_score(y[v], s[v]), "AUPRC": average_precision_score(y[v], s[v]),
            "N_endpoint_images": int(v.sum())}

def dose_metrics(meta, score):
    d, s = meta["Dose_value"].to_numpy(float), np.asarray(score, float)
    v = np.isfinite(d) & np.isfinite(s)
    rho, rp = stats.spearmanr(d[v], s[v]); tau, tp = stats.kendalltau(d[v], s[v])
    return {"Spearman_r": float(rho), "Spearman_p": float(rp), "Kendall_tau": float(tau),
            "Kendall_p": float(tp), "N_all_images": int(v.sum())}

# ============================================================
# 4. Composite: within-time cross-fitting + source-day consensus
# ============================================================
composite_oof, consensus, direction_rows = {}, {}, []

for day in DAYS:
    X, meta = DAYSETS[day]["X"], DAYSETS[day]["meta"].copy()
    oof, fold_dirs = np.full(len(meta), np.nan), []
    for fold, tr, te in grouped_split(meta, "Dose"):
        Xtr, Xte, mtr = X[tr], X[te], meta.loc[tr].reset_index(drop=True)
        Xtr_z, mu, sd = control_standardize(Xtr, mtr["Dose"].to_numpy(), Xtr)
        direction = estimate_directions(Xtr_z, mtr); fold_dirs.append(direction)
        oof[te] = composite_score(((Xte - mu) / sd) * direction[None, :])
        direction_rows += [{"Source_day": day, "Fold": fold, "Feature": f, "Direction": d}
                           for f, d in zip(FEATURES, direction)]
    composite_oof[day] = oof
    consensus[day] = np.where(np.mean(np.vstack(fold_dirs), axis=0) < 0, -1., 1.)

direction_by_fold_df = pd.DataFrame(direction_rows)

# ============================================================
# 5. Composite 2×2 transfer
# ============================================================
comp_metrics, comp_scores = [], []
for source in DAYS:
    for target in DAYS:
        Xt, mt = DAYSETS[target]["X"], DAYSETS[target]["meta"].copy()
        if source == target:
            score, evaluation = composite_oof[target].copy(), "Within-time cross-fitted"
        else:
            Xt_z, _, _ = control_standardize(Xt, mt["Dose"].to_numpy(), Xt)
            score, evaluation = composite_score(Xt_z * consensus[source][None, :]), "Cross-time direction transfer"
        comp_metrics.append({"Method": "Composite", "Source_day": source, "Target_day": target,
                             "Evaluation": evaluation, **endpoint_metrics(mt, score), **dose_metrics(mt, score)})
        tmp = mt.copy(); tmp["Method"], tmp["Source_day"], tmp["Target_day"] = "Composite", source, target
        tmp["Evaluation"], tmp["Score"] = evaluation, score; comp_scores.append(tmp)

composite_metrics_df = pd.DataFrame(comp_metrics)
composite_scores_df = pd.concat(comp_scores, ignore_index=True)

# ============================================================
# 6. Linear ML within-time + results/cross_time
# ============================================================
linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=1., class_weight="balanced",
                               solver="liblinear", max_iter=5000, random_state=RANDOM_STATE)),
])

def fit_linear_within(day):
    X, meta = DAYSETS[day]["X"], DAYSETS[day]["meta"].copy()
    ep = meta["Dose"].isin(["0 ug/L", "500 ug/L"]).to_numpy()
    Xe, me = X[ep], meta.loc[ep].reset_index(drop=True)
    me["Endpoint_label"] = me["Dose"].eq("500 ug/L").astype(int); y = me["Endpoint_label"].to_numpy(int)
    oof, models = np.full(len(me), np.nan), []
    for _, tr, te in grouped_split(me, "Endpoint_label"):
        model = clone(linear_model).fit(Xe[tr], y[tr]); oof[te] = model.decision_function(Xe[te]); models.append(model)
    full = np.full(len(meta), np.nan); full[np.flatnonzero(ep)] = oof
    mid = meta["Dose"].isin(["50 ug/L", "200 ug/L"]).to_numpy(); Xmid = X[mid]
    full[mid] = np.mean(np.vstack([m.decision_function(Xmid) for m in models]), axis=0)
    return full

linear_within = {d: fit_linear_within(d) for d in DAYS}
lin_metrics, lin_scores = [], []

for source in DAYS:
    Xs, ms = DAYSETS[source]["X"], DAYSETS[source]["meta"].copy()
    ep = ms["Dose"].isin(["0 ug/L", "500 ug/L"]).to_numpy()
    ys = ms.loc[ep, "Dose"].eq("500 ug/L").astype(int).to_numpy()
    source_model = clone(linear_model).fit(Xs[ep], ys)
    for target in DAYS:
        Xt, mt = DAYSETS[target]["X"], DAYSETS[target]["meta"].copy()
        if source == target:
            score, evaluation = linear_within[target].copy(), "Within-time grouped OOF"
        else:
            score, evaluation = source_model.decision_function(Xt), "Cross-time direct transfer"
        lin_metrics.append({"Method": "Linear ML", "Source_day": source, "Target_day": target,
                            "Evaluation": evaluation, **endpoint_metrics(mt, score), **dose_metrics(mt, score)})
        tmp = mt.copy(); tmp["Method"], tmp["Source_day"], tmp["Target_day"] = "Linear ML", source, target
        tmp["Evaluation"], tmp["Score"] = evaluation, score; lin_scores.append(tmp)

linear_metrics_df = pd.DataFrame(lin_metrics)
linear_scores_df = pd.concat(lin_scores, ignore_index=True)

# ============================================================
# 7. Plot tables generated in real time
# ============================================================
def metric_matrix(df, metric):
    return np.array([[df.loc[(df.Source_day == s) & (df.Target_day == t), metric].iloc[0] for t in DAYS] for s in DAYS])

def ann_matrix(mat):
    a = np.empty((2, 2), object)
    for i in range(2):
        for j in range(2):
            tag = "within" if i == j else ("reversed" if mat[i, j] < 0.5 else "cross")
            a[i, j] = f"{mat[i, j]:.3f}\n{tag}"
    return a

comp_auc, lin_auc = metric_matrix(composite_metrics_df, "AUROC"), metric_matrix(linear_metrics_df, "AUROC")

summary_rows = []
for method, df in [("Composite", composite_metrics_df), ("Linear ML", linear_metrics_df)]:
    for source in DAYS:
        target = "Day 14" if source == "Day 5" else "Day 5"
        w = df[(df.Source_day == source) & (df.Target_day == source)].iloc[0]
        c = df[(df.Source_day == source) & (df.Target_day == target)].iloc[0]
        summary_rows.append({"Method": method, "Source_day": source, "Cross_target_day": target,
                             "Within_Spearman_r": w.Spearman_r, "Cross_Spearman_r": c.Spearman_r,
                             "Within_AUROC": w.AUROC, "Cross_AUROC": c.AUROC})
transfer_summary_df = pd.DataFrame(summary_rows)

# c,d use COMPOSITE cross-time scores
c_df = composite_scores_df[(composite_scores_df.Source_day == "Day 5") &
                           (composite_scores_df.Target_day == "Day 14")].copy()
d_df = composite_scores_df[(composite_scores_df.Source_day == "Day 14") &
                           (composite_scores_df.Target_day == "Day 5")].copy()

# ============================================================
# 8. Export recalculated tables
# ============================================================
if SAVE_TABLES:
    composite_metrics_df.to_csv(RESULT_DIR / "cross_time_composite_metrics.csv", index=False)
    linear_metrics_df.to_csv(RESULT_DIR / "cross_time_logistic_metrics.csv", index=False)
    composite_scores_df.to_csv(RESULT_DIR / "cross_time_composite_scores.csv", index=False)
    linear_scores_df.to_csv(RESULT_DIR / "cross_time_logistic_scores.csv", index=False)
    transfer_summary_df.to_csv(RESULT_DIR / "cross_time_transfer_summary.csv", index=False)
    direction_by_fold_df.to_csv(RESULT_DIR / "cross_time_composite_directions_by_fold.csv", index=False)
    with pd.ExcelWriter(RESULT_DIR / "Fig_cross_time_transfer_results.xlsx") as xw:
        composite_metrics_df.to_excel(xw, sheet_name="composite_metrics", index=False); linear_metrics_df.to_excel(xw, sheet_name="linear_metrics", index=False)
        transfer_summary_df.to_excel(xw, sheet_name="transfer_summary", index=False); direction_by_fold_df.to_excel(xw, sheet_name="directions_by_fold", index=False)
        composite_scores_df.to_excel(xw, sheet_name="composite_scores", index=False); linear_scores_df.to_excel(xw, sheet_name="linear_scores", index=False)

# ============================================================
# 9. Publication style
# ============================================================
def lighten(color, amount=0.55):
    c = np.asarray(to_rgb(color)); return to_hex(c + (1 - c) * amount)
BOX_COLOR = {k: lighten(v) for k, v in DOSE_COLOR.items()}

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.linewidth": 0.8,
    "axes.labelsize": 13, "axes.titlesize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
sns.set_theme(context="notebook", style="white", rc={
    "font.family": "Arial", "font.size": 13, "axes.linewidth": 0.8,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

# ============================================================
# 10. Four-panel layout
# ============================================================
fig = plt.figure(figsize=(20.0, 14.5))
outer = fig.add_gridspec(
    2, 2,
    left=0.08, right=0.985, bottom=0.10, top=0.95,
    width_ratios=[1.0, 1.0],
    height_ratios=[0.88, 1.0],
    wspace=0.38, hspace=0.48
)

# panel a contains two small heatmaps + one shared horizontal colorbar
gs_a = outer[0, 0].subgridspec(2, 2, height_ratios=[1, 0.11], hspace=0.28, wspace=0.18)
ax_a1, ax_a2 = fig.add_subplot(gs_a[0, 0]), fig.add_subplot(gs_a[0, 1])
cax_a = fig.add_subplot(gs_a[1, :])
ax_b, ax_c, ax_d = fig.add_subplot(outer[0, 1]), fig.add_subplot(outer[1, 0]), fig.add_subplot(outer[1, 1])

# ---------------- Panel a ----------------
sns.heatmap(comp_auc, ax=ax_a1, cmap="coolwarm", center=.5, vmin=0, vmax=1, square=True,
            annot=ann_matrix(comp_auc), fmt="", annot_kws={"fontsize": 10, "fontweight": "bold"},
            linewidths=.6, linecolor="white", xticklabels=DISPLAY_DAYS, yticklabels=DISPLAY_DAYS, cbar=False)
hm = sns.heatmap(lin_auc, ax=ax_a2, cmap="coolwarm", center=.5, vmin=0, vmax=1, square=True,
                 annot=ann_matrix(lin_auc), fmt="", annot_kws={"fontsize": 10, "fontweight": "bold"},
                 linewidths=.6, linecolor="white", xticklabels=DISPLAY_DAYS, yticklabels=DISPLAY_DAYS,
                 cbar=True, cbar_ax=cax_a, cbar_kws={"orientation": "horizontal"})
ax_a1.set_title("Composite axis", fontweight="bold", pad=5, fontsize=14)
ax_a2.set_title("Linear ML", fontweight="bold", pad=5, fontsize=14)
ax_a1.set(xlabel="", ylabel="Source"); ax_a2.set(xlabel="", ylabel="")
ax_a1.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False, length=0)
ax_a1.tick_params(axis="y", length=0, rotation=0)
ax_a2.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False, length=0)
ax_a2.tick_params(axis="y", left=False, labelleft=False, length=0)
cb = hm.collections[0].colorbar; cb.set_label("Endpoint AUROC", fontsize=13, labelpad=3)
cb.ax.tick_params(labelsize=12, length=2); cb.outline.set_linewidth(.6)

# ---------------- Panel b ----------------
p = transfer_summary_df.copy(); p["x"] = [0, 1, 3, 4]
for _, r in p.iterrows():
    x, y0, y1, color = float(r.x), float(r.Within_Spearman_r), float(r.Cross_Spearman_r), DAY_COLOR[r.Source_day]
    ax_b.plot([x, x], [y0, y1], color=color, lw=1.1, alpha=.8, zorder=1)
    ax_b.scatter(x, y0, s=38, facecolor=color, edgecolor=color, lw=.9, zorder=3)
    ax_b.scatter(x, y1, s=38, facecolor="white", edgecolor=color, lw=1.2, zorder=3)
ax_b.axhline(0, color="#8D98A7", lw=.8, ls=(0, (2.2, 2.2))); ax_b.axvline(2, color="#B8B8B8", lw=.6, ls=(0, (2.2, 2.2)))
all_r = np.r_[p.Within_Spearman_r.to_numpy(float), p.Cross_Spearman_r.to_numpy(float)]
ax_b.set_xlim(-.55, 4.55); ax_b.set_ylim(min(-.35, all_r.min() - .1), max(.70, all_r.max() + .1))
ax_b.set_xticks([0, 1, 3, 4], ["D5", "D14", "D5", "D14"])
ax_b.set_title("Dose-ordering transfer", fontweight="bold", pad=6, fontsize=14)
ax_b.set_ylabel("Dose ordering\n" + r"(Spearman $r_s$)")
ax_b.text(.5, -.16, "Composite", transform=ax_b.get_xaxis_transform(), ha="center", va="top")
ax_b.text(3.5, -.16, "Linear ML", transform=ax_b.get_xaxis_transform(), ha="center", va="top")
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False); ax_b.tick_params(axis="x", length=0)

type_handles = [
    Line2D([0], [0], marker="o", ls="none", ms=5.2,
           mfc="#555555", mec="#555555", label="Within-time"),
    Line2D([0], [0], marker="o", ls="none", ms=5.2,
           mfc="white", mec="#555555", label="Cross-time"),
]

ax_b.legend(
    handles=type_handles,
    frameon=False,
    loc="lower right",
    borderaxespad=0.3,
    handletextpad=.3,
)

# ---------------- Boxplot helper for panels c,d ----------------
def transfer_boxplot(ax, df, title, source, target, text_xy=(.03, .97), text_ha="left", text_va="top"):
    sns.boxplot(data=df, x="Dose", y="Score", order=DOSES, hue="Dose", hue_order=DOSES,
                palette=BOX_COLOR, width=.58, dodge=False, fliersize=0, linewidth=.65,
                saturation=1, legend=False, ax=ax,
                boxprops={"edgecolor": "#2A2A2A", "linewidth": .65},
                whiskerprops={"color": "#2A2A2A", "linewidth": .65},
                capprops={"color": "#2A2A2A", "linewidth": .65},
                medianprops={"color": "#111111", "linewidth": 1.0})
    sns.stripplot(data=df, x="Dose", y="Score", order=DOSES, hue="Dose", hue_order=DOSES,
                  palette=DOSE_COLOR, size=1.45, alpha=.58, jitter=.18, linewidth=0,
                  dodge=False, legend=False, ax=ax, zorder=3)
    ax.axhline(0, color="#AEB6C2", lw=.8, ls=(0, (2.2, 2.2)), zorder=0)
    ax.set_title(title, fontweight="bold", pad=6, fontsize=14); ax.set_xlabel(""); ax.set_ylabel("Transferred composite\nphenotype score")
    ax.set_xticklabels([DISPLAY_DOSE[d] for d in DOSES])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(axis="x", length=0)
    row = composite_metrics_df[(composite_metrics_df.Source_day == source) &
                               (composite_metrics_df.Target_day == target)].iloc[0]
    ax.text(text_xy[0], text_xy[1], f"AUROC = {row.AUROC:.3f}\nSpearman $r_s$ = {row.Spearman_r:.3f}",
            transform=ax.transAxes, ha=text_ha, va=text_va, fontsize=12)
    vals = df.Score.to_numpy(float); vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = np.nanpercentile(vals, [1, 99]); pad = .13 * (hi - lo + 1e-8); ax.set_ylim(lo - pad, hi + pad)

# ---------------- Panels c,d ----------------
transfer_boxplot(ax_c, c_df, "Day 5 phenotype axis → Day 14 data", "Day 5", "Day 14",
                 text_xy=(.03, .05), text_ha="left", text_va="bottom")
transfer_boxplot(ax_d, d_df, "Day 14 phenotype axis → Day 5 data", "Day 14", "Day 5",
                 text_xy=(.97, .97), text_ha="right", text_va="top")

# panel labels: a belongs to the whole nested transfer-matrix block
ax_a1.text(-.26, 1.11, "a", transform=ax_a1.transAxes, fontsize=15, fontweight="bold", va="bottom")
for ax, lab, xoff in [(ax_b, "b", -.14), (ax_c, "c", -.10), (ax_d, "d", -.10)]:
    ax.text(xoff, 1.035, lab, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")

# ============================================================
# 11. Save
# ============================================================
base = FIG_DIR / "Fig_cross_time_transfer_4panel_realtime"
if SAVE_FIGURE:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()

print("\nComposite metrics:\n", composite_metrics_df.to_string(index=False))
print("\nLinear ML metrics:\n", linear_metrics_df.to_string(index=False))
print("\nTransfer summary:\n", transfer_summary_df.to_string(index=False))
print("\nFigure:", base.with_suffix(".png").resolve())