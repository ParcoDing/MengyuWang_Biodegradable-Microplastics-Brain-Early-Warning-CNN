# -*- coding: utf-8 -*-
"""
Day-5 feature-based logistic-regression phenotype score using the fixed 16-feature panel.

固定使用 16 个特征：
- 11 个形态/空间/通道特征
- 5 个 GLCM 纹理特征

方法：
- 0 ppb 与 500 ppb 用于训练端点分类器
- 0/500 ppb 使用严格 out-of-fold decision score
- 50/200 ppb 不参与训练，由每一折模型预测后取平均
- 输出 score、端点分类性能、fold 系数和绘图数据

说明：
原始 npy 数组可以仍然包含 19 个或更多特征；
本脚本仅按 feature_names.npy 中的名称提取指定 16 项。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pathlib import Path
from scipy import stats
from matplotlib.colors import to_rgb, to_hex

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


# =========================
# 1. 参数与路径
# =========================

DATA_DIR = Path("./data/features")
RESULT_DIR = Path("./results/logistic_regression")
FIG_DIR = Path("./results/figures")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 0
SAVE_FIGURES = True


# =========================
# 2. 读取 5 天数据
# 原始 array shape = (n_saved_features, n_images)
# =========================

day5_0CT_arr = np.load(DATA_DIR / "day5_0CT_arr.npy")
day5_50ppb_arr = np.load(DATA_DIR / "day5_50ppb_arr.npy")
day5_200ppb_arr = np.load(DATA_DIR / "day5_200ppb_arr.npy")
day5_500ppb_arr = np.load(DATA_DIR / "day5_500ppb_arr.npy")

saved_feature_names = np.load(
    DATA_DIR / "feature_names.npy",
    allow_pickle=False,
).astype(str).tolist()

dose_arrays = {
    "0 ppb": day5_0CT_arr,
    "50 ppb": day5_50ppb_arr,
    "200 ppb": day5_200ppb_arr,
    "500 ppb": day5_500ppb_arr,
}

dose_order = ["0 ppb", "50 ppb", "200 ppb", "500 ppb"]

dose_values = {
    "0 ppb": 0.0,
    "50 ppb": 50.0,
    "200 ppb": 200.0,
    "500 ppb": 500.0,
}


# =========================
# 3. 数据一致性检查
# =========================

n_features = len(saved_feature_names)

for dose, arr in dose_arrays.items():
    if arr.ndim != 2:
        raise ValueError(f"{dose} 数组必须是二维，当前 shape={arr.shape}")

    if arr.shape[0] != n_features:
        raise ValueError(
            f"{dose} 数组第0维为 {arr.shape[0]}，"
            f"但 feature_names.npy 中有 {n_features} 个特征。"
        )



# =========================
# 4. 选择用于机器学习的特征
# 使用内部特征名定位，不再依赖硬编码索引
# =========================

feature_names = [
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

selected_feature_names = feature_names.copy()

missing_features = [
    name
    for name in selected_feature_names
    if name not in saved_feature_names
]

if missing_features:
    raise ValueError(
        "feature_names.npy 中缺少以下特征：\n"
        + "\n".join(missing_features)
    )

selected_feature_indices = [
    saved_feature_names.index(name)
    for name in selected_feature_names
]

selected_feature_table = pd.DataFrame({
    "Selected_feature_index": selected_feature_indices,
    "Selected_feature_name": selected_feature_names,
})

print("\nSelected features:")
print(selected_feature_table.to_string(index=False))


def arr_to_samples(arr):
    """
    输入:
        arr shape = (n_saved_features, n_images)
    输出:
        X shape = (n_images, n_selected_features)
    """
    return arr[selected_feature_indices, :].T.astype(float)


X_0 = arr_to_samples(day5_0CT_arr)
X_50 = arr_to_samples(day5_50ppb_arr)
X_200 = arr_to_samples(day5_200ppb_arr)
X_500 = arr_to_samples(day5_500ppb_arr)


# =========================
# 5. 0 vs 500 ppb 端点分类
# =========================

X_endpoint = np.vstack([X_0, X_500])

y_endpoint = np.concatenate([
    np.zeros(len(X_0), dtype=int),
    np.ones(len(X_500), dtype=int),
])

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "clf",
        LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
    ),
])

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

endpoint_oof_score = np.full(
    len(y_endpoint),
    np.nan,
    dtype=float,
)

endpoint_fold = np.full(
    len(y_endpoint),
    -1,
    dtype=int,
)

score_50_folds = []
score_200_folds = []
coefficient_rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_endpoint, y_endpoint),
    start=1,
):
    X_train = X_endpoint[train_idx]
    y_train = y_endpoint[train_idx]
    X_test = X_endpoint[test_idx]

    model.fit(X_train, y_train)

    endpoint_oof_score[test_idx] = model.decision_function(X_test)
    endpoint_fold[test_idx] = fold

    score_50_folds.append(
        model.decision_function(X_50)
    )
    score_200_folds.append(
        model.decision_function(X_200)
    )

    coefficients = model.named_steps["clf"].coef_[0]

    for feature_name, coefficient in zip(
        selected_feature_names,
        coefficients,
    ):
        coefficient_rows.append({
            "Fold": fold,
            "Feature": feature_name,
            "Standardized_coefficient": float(coefficient),
        })


# =========================
# 6. 生成各剂量 score
# =========================

score_0 = endpoint_oof_score[:len(X_0)]
score_500 = endpoint_oof_score[len(X_0):]

score_50 = np.nanmean(
    np.vstack(score_50_folds),
    axis=0,
)

score_200 = np.nanmean(
    np.vstack(score_200_folds),
    axis=0,
)

# LogisticRegression 中 label 1 已定义为500 ppb，
# 因而 decision_function 正值本应更接近500 ppb。
# 保留检查，防止异常数据流程导致整体方向反转。
if np.nanmean(score_500) < np.nanmean(score_0):
    score_0 *= -1
    score_50 *= -1
    score_200 *= -1
    score_500 *= -1

# 使用 OOF 0 ppb score 的均值和标准差做可视化标准化
score_0_mean = np.nanmean(score_0)
score_0_std = np.nanstd(score_0)

if not np.isfinite(score_0_std) or score_0_std <= 1e-12:
    score_0_std = 1.0

score_0_z = (score_0 - score_0_mean) / score_0_std
score_50_z = (score_50 - score_0_mean) / score_0_std
score_200_z = (score_200 - score_0_mean) / score_0_std
score_500_z = (score_500 - score_0_mean) / score_0_std

ml_score_arrays = {
    "0 ppb": score_0_z,
    "50 ppb": score_50_z,
    "200 ppb": score_200_z,
    "500 ppb": score_500_z,
}


# =========================
# 7. 端点分类性能
# 性能必须使用未标准化或标准化后的 OOF score，
# 两者 AUROC/AUPRC 相同
# =========================

endpoint_auc = roc_auc_score(
    y_endpoint,
    endpoint_oof_score,
)

endpoint_ap = average_precision_score(
    y_endpoint,
    endpoint_oof_score,
)

print("\nDay-5 endpoint OOF performance:")
print(f"AUROC = {endpoint_auc:.6f}")
print(f"AUPRC = {endpoint_ap:.6f}")


# =========================
# 8. 整理 dataframe
# =========================

rows = []

for dose in dose_order:
    vals = ml_score_arrays[dose]

    for sample_idx, value in enumerate(vals):
        if not np.isfinite(value):
            continue

        rows.append({
            "Dose": dose,
            "Dose_value": dose_values[dose],
            "Feature-based ML late damage score": float(value),
            "Sample": sample_idx,
        })

df_ml = pd.DataFrame(rows)

endpoint_oof_df = pd.DataFrame({
    "Dose": (
        ["0 ppb"] * len(X_0)
        + ["500 ppb"] * len(X_500)
    ),
    "True_label": y_endpoint,
    "OOF_decision_score": endpoint_oof_score,
    "Fold": endpoint_fold,
})

coefficient_by_fold_df = pd.DataFrame(
    coefficient_rows
)

coefficient_summary_df = (
    coefficient_by_fold_df
    .groupby("Feature", as_index=False)
    .agg(
        Mean_coefficient=(
            "Standardized_coefficient",
            "mean",
        ),
        SD_coefficient=(
            "Standardized_coefficient",
            "std",
        ),
        Positive_folds=(
            "Standardized_coefficient",
            lambda x: int(np.sum(np.asarray(x) > 0)),
        ),
        Negative_folds=(
            "Standardized_coefficient",
            lambda x: int(np.sum(np.asarray(x) < 0)),
        ),
    )
)

coefficient_summary_df["Abs_mean_coefficient"] = (
    coefficient_summary_df["Mean_coefficient"].abs()
)

coefficient_summary_df = coefficient_summary_df.sort_values(
    "Abs_mean_coefficient",
    ascending=False,
)


# =========================
# 9. Image-level 剂量趋势检验
# 如果同一条鱼有多张图，主文应改为 fish-level
# =========================

x = df_ml["Dose_value"].to_numpy(dtype=float)
y = df_ml[
    "Feature-based ML late damage score"
].to_numpy(dtype=float)

slope, intercept, r_value, p_value, slope_se = (
    stats.linregress(x, y)
)

ci_low = slope - 1.96 * slope_se
ci_high = slope + 1.96 * slope_se

spearman_r, spearman_p = stats.spearmanr(
    x,
    y,
)

print("\nDay-5 feature-based ML late damage score trend:")
print(f"n = {len(df_ml)} images")
print(f"slope = {slope:.6g}")
print(f"95% CI = [{ci_low:.6g}, {ci_high:.6g}]")
print(f"linear p = {p_value:.6g}")
print(f"Pearson r = {r_value:.6g}")
print(f"Spearman r = {spearman_r:.6g}")
print(f"Spearman p = {spearman_p:.6g}")


# =========================
# 5. 保存 score 和表格
# =========================

score_file_map = {
    "0 ppb": "day5_0CT_featureML_late_damage_score.npy",
    "50 ppb": "day5_50ppb_featureML_late_damage_score.npy",
    "200 ppb": "day5_200ppb_featureML_late_damage_score.npy",
    "500 ppb": "day5_500ppb_featureML_late_damage_score.npy",
}

for dose in dose_order:
    np.save(
        RESULT_DIR / score_file_map[dose],
        ml_score_arrays[dose],
    )

selected_feature_table.to_csv(
    RESULT_DIR / "day5_featureML_selected_features.csv",
    index=False,
)

df_ml.to_csv(
    RESULT_DIR / "day5_featureML_late_damage_score.csv",
    index=False,
)

endpoint_oof_df.to_csv(
    RESULT_DIR / "day5_featureML_endpoint_oof_scores.csv",
    index=False,
)

coefficient_by_fold_df.to_csv(
    RESULT_DIR / "day5_featureML_coefficients_by_fold.csv",
    index=False,
)

coefficient_summary_df.to_csv(
    RESULT_DIR / "day5_featureML_coefficient_summary.csv",
    index=False,
)

performance_df = pd.DataFrame([{
    "Endpoint_AUROC": endpoint_auc,
    "Endpoint_AUPRC": endpoint_ap,
    "Dose_trend_slope": slope,
    "Dose_trend_CI_low": ci_low,
    "Dose_trend_CI_high": ci_high,
    "Dose_trend_linear_p": p_value,
    "Dose_trend_Pearson_r": r_value,
    "Dose_trend_Spearman_r": spearman_r,
    "Dose_trend_Spearman_p": spearman_p,
    "N_images": len(df_ml),
    "N_selected_features": len(selected_feature_names),
}])

performance_df.to_csv(
    RESULT_DIR / "day5_featureML_summary.csv",
    index=False,
)


# =========================
# 11. Nature 风格参数
# =========================

dose_palette = {
    "0 ppb": "#3A6073",
    "50 ppb": "#6494AA",
    "200 ppb": "#DDA15E",
    "500 ppb": "#A64B4B",
}


def lighten_color(color, amount=0.5):
    c = np.asarray(to_rgb(color))
    return to_hex(
        c + (np.ones(3) - c) * amount
    )


box_palette = {
    key: lighten_color(value)
    for key, value in dose_palette.items()
}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

sns.set_theme(
    context="paper",
    style="white",
    rc={
        "font.family": "Arial",
        "axes.linewidth": 0.5,
    },
)


# =========================
# 12. 绘制 feature-based ML score
# =========================

fig, ax = plt.subplots(figsize=(4.5, 3.5))

ax.set_axisbelow(True)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

sns.boxplot(
    data=df_ml,
    x="Dose",
    y="Feature-based ML late damage score",
    order=dose_order,
    hue="Dose",
    hue_order=dose_order,
    palette=box_palette,
    width=0.58,
    dodge=False,
    fliersize=0,
    linewidth=0.5,
    saturation=1.0,
    legend=False,
    ax=ax,
    boxprops={
        "edgecolor": "#2A2A2A",
        "linewidth": 0.5,
        "zorder": 2,
    },
    whiskerprops={
        "color": "#2A2A2A",
        "linewidth": 0.5,
    },
    capprops={
        "color": "#2A2A2A",
        "linewidth": 0.5,
    },
    medianprops={
        "color": "#111111",
        "linewidth": 0.85,
    },
)

sns.stripplot(
    data=df_ml,
    x="Dose",
    y="Feature-based ML late damage score",
    order=dose_order,
    hue="Dose",
    hue_order=dose_order,
    palette=dose_palette,
    size=1.35,
    alpha=0.75,
    jitter=0.18,
    linewidth=0,
    legend=False,
    ax=ax,
    zorder=3,
)

ax.axhline(
    0,
    color="#AEB6C2",
    linewidth=1.0,
    linestyle=(0, (2.2, 2.2)),
    zorder=0,
)

y_vals = df_ml[
    "Feature-based ML late damage score"
].to_numpy(dtype=float)

lo, hi = np.nanpercentile(
    y_vals,
    [1, 99],
)

pad = 0.18 * (hi - lo + 1e-8)
ax.set_ylim(lo - pad, hi + pad)

ax.set_xlabel("")
ax.set_ylabel(
    "Feature-based ML late damage score\n"
    "(control-normalized OOF decision score)",
    labelpad=4,
)

ax.text(
    -0.16,
    1.04,
    "(d)",
    transform=ax.transAxes,
    fontsize=15,
    fontweight="bold",
    va="bottom",
    ha="right",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#2A2A2A")
ax.spines["bottom"].set_color("#2A2A2A")

ax.tick_params(
    axis="x",
    direction="out",
    color="#2A2A2A",
    pad=2,
)

ax.tick_params(
    axis="y",
    direction="out",
    color="#2A2A2A",
    pad=2,
)

for tick_label in ax.get_xticklabels():
    tick_label.set_rotation(0)
    tick_label.set_horizontalalignment("center")

plt.tight_layout()

fig.subplots_adjust(
    left=0.28,
    right=0.96,
    bottom=0.16,
    top=0.92,
)


# =========================
# 13. 保存图片
# =========================

if SAVE_FIGURES:
    '''
    fig.savefig(
        FIG_DIR / "fig2_d_day5_featureML_late_damage_score.pdf",
        bbox_inches="tight",
        transparent=True,
    )

    fig.savefig(
        FIG_DIR / "fig2_d_day5_featureML_late_damage_score.svg",
        bbox_inches="tight",
        transparent=True,
    )

    fig.savefig(
        FIG_DIR / "fig2_d_day5_featureML_late_damage_score.tiff",
        dpi=600,
        bbox_inches="tight",
        transparent=False,
    )
    '''
    '''
    fig.savefig(
        FIG_DIR / "fig2_d_day5_featureML_late_damage_score.png",
        dpi=500,
        bbox_inches="tight",
        transparent=False,
    )
    '''

plt.show()
#plt.close(fig)

print("\n输出目录：")
print("Scores and tables:", RESULT_DIR.resolve())
print("Figures:", FIG_DIR.resolve())