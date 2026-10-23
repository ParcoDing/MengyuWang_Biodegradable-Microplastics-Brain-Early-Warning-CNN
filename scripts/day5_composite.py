# -*- coding: utf-8 -*-
"""
Day-5 cross-fitted composite phenotype score using the fixed 16-feature panel.

关键改动
--------
1. 仅使用指定的 16 个经典显式特征；不再使用：
   reduced_green_cell_density、green_cell_area_cv、
   magenta_largest_component_fraction。
2. 原始 npy 数组可以仍然保存更多特征；脚本按 feature_names.npy 中的名称定位这 16 项。
3. 使用交叉拟合：每一折只用训练集 0 ppb 样本确定 z-score 参数，
   并只用训练集估计各特征的剂量方向，再对测试集计算分数。
4. 默认保留“特征家族等权”模式：先家族内平均，再对 5 个家族等权平均。
5. 若希望 16 个单项特征完全等权，可将 SCORE_MODE 改为 "feature_mean"。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pathlib import Path
from scipy import stats
from matplotlib.colors import to_rgb, to_hex
from sklearn.model_selection import StratifiedKFold


# =========================
# 1. 参数与路径
# =========================

DATA_DIR = Path("./data/features")
SCORE_DIR = Path("./results/composite")
FIG_DIR = Path("./results/figures")

SCORE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# "group_balanced": 先在每类特征内平均，再对类别等权平均
# "feature_mean": 所有选中特征直接等权平均
SCORE_MODE = "group_balanced"

SAVE_FIGURE = True


# =========================
# 2. 读取 5 天数据
# 更新后每个数组应为 (19, n_images)
# =========================

day5_0CT_arr = np.load(DATA_DIR / "day5_0CT_arr.npy")
day5_50ppb_arr = np.load(DATA_DIR / "day5_50ppb_arr.npy")
day5_200ppb_arr = np.load(DATA_DIR / "day5_200ppb_arr.npy")
day5_500ppb_arr = np.load(DATA_DIR / "day5_500ppb_arr.npy")

saved_feature_names = np.load(
    DATA_DIR / "feature_names.npy",
    allow_pickle=False,
).astype(str).tolist()

# 固定使用的 16 个经典特征
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

n_saved_features = len(saved_feature_names)

for dose, arr in dose_arrays.items():
    if arr.ndim != 2:
        raise ValueError(f"{dose} 数组必须是二维，当前 shape={arr.shape}")

    if arr.shape[0] != n_saved_features:
        raise ValueError(
            f"{dose} 数组有 {arr.shape[0]} 行特征，"
            f"但 feature_names.npy 中有 {n_saved_features} 个名称。"
        )

missing = [
    name for name in feature_names
    if name not in saved_feature_names
]
if missing:
    raise ValueError(
        "feature_names.npy 中缺少以下指定特征：\n"
        + "\n".join(missing)
    )


# =========================
# 4. 参与综合分数的特征
# reduced_green_cell_density 被排除，因为它与 cell_density 完全重复
# =========================

score_feature_groups = {
    "density_and_size": [
        "cell_density",
        "green_mean_cell_area",
    ],
    "shape": [
        "cell_roundness",
        "green_mean_eccentricity",
    ],
    "spatial_organization": [
        "nearest_neighbor_distance",
        "local_density_variance",
        "spatial_heterogeneity_increase",
        "green_neighbor_degree",
        "green_close_neighbor_fraction",
        "green_dilation_connectivity",
    ],
    "channel_relationship": [
        "channel_colocalization",
    ],
    "texture": [
        "texture_glcm_contrast_std",
        "texture_glcm_energy_std",
        "texture_glcm_dissimilarity_std",
        "texture_glcm_correlation_std",
        "texture_glcm_homogeneity_mean",
    ],
}

score_feature_names = [
    name
    for group_names in score_feature_groups.values()
    for name in group_names
]

if set(score_feature_names) != set(feature_names) or len(score_feature_names) != 16:
    raise ValueError("score_feature_groups 必须且只能覆盖固定的 16 个特征。")

feature_index = {
    name: saved_feature_names.index(name)
    for name in saved_feature_names
}

score_feature_indices = np.array(
    [feature_index[name] for name in score_feature_names],
    dtype=int,
)

group_indices_in_score_matrix = {
    group_name: np.array(
        [score_feature_names.index(name) for name in names],
        dtype=int,
    )
    for group_name, names in score_feature_groups.items()
}


# =========================
# 5. 合并所有剂量组
# X_raw shape = (n_images, n_saved_features)
# =========================

X_blocks = []
dose_labels = []
dose_numeric = []
within_dose_sample = []

for dose in dose_order:
    arr = dose_arrays[dose].T

    X_blocks.append(arr)
    dose_labels.extend([dose] * len(arr))
    dose_numeric.extend([dose_values[dose]] * len(arr))
    within_dose_sample.extend(range(len(arr)))

X_raw = np.vstack(X_blocks).astype(float)
dose_labels = np.asarray(dose_labels)
dose_numeric = np.asarray(dose_numeric, dtype=float)
within_dose_sample = np.asarray(within_dose_sample, dtype=int)


# =========================
# 6. 工具函数
# =========================

def estimate_feature_directions(X_train_z, train_dose):
    """
    在训练集上逐特征拟合 feature ~ dose 的线性斜率。
    返回 +1 或 -1，使方向对齐后数值越高代表损伤越重。
    """
    directions = np.ones(X_train_z.shape[1], dtype=float)

    for j in range(X_train_z.shape[1]):
        y = X_train_z[:, j]
        valid = np.isfinite(y) & np.isfinite(train_dose)

        if valid.sum() < 3 or np.nanstd(y[valid]) <= 1e-12:
            directions[j] = 1.0
            continue

        slope = stats.linregress(
            train_dose[valid],
            y[valid],
        ).slope

        directions[j] = -1.0 if slope < 0 else 1.0

    return directions


def row_nanmean(matrix):
    """
    对每一行忽略 NaN 求均值。
    若某行全部为 NaN，则返回 NaN。
    """
    finite_count = np.sum(np.isfinite(matrix), axis=1)
    summed = np.nansum(matrix, axis=1)

    out = np.full(matrix.shape[0], np.nan, dtype=float)
    valid = finite_count > 0
    out[valid] = summed[valid] / finite_count[valid]

    return out


def calculate_late_score(aligned_features):
    """
    aligned_features shape = (n_samples, n_score_features)
    """
    if SCORE_MODE == "feature_mean":
        return row_nanmean(aligned_features)

    if SCORE_MODE != "group_balanced":
        raise ValueError(
            "SCORE_MODE 只能是 'group_balanced' 或 'feature_mean'"
        )

    group_scores = []

    for group_name in score_feature_groups:
        idx = group_indices_in_score_matrix[group_name]
        group_scores.append(
            row_nanmean(aligned_features[:, idx])
        )

    group_scores = np.column_stack(group_scores)
    return row_nanmean(group_scores)


# =========================
# 7. 交叉拟合 late-damage score
# =========================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_score = np.full(len(X_raw), np.nan, dtype=float)
fold_id = np.full(len(X_raw), -1, dtype=int)
fold_direction_rows = []

for fold, (train_idx, test_idx) in enumerate(
    skf.split(X_raw, dose_labels),
    start=1,
):
    train_control = (
        dose_labels[train_idx] == "0 ppb"
    )

    X_control_train = X_raw[train_idx][train_control]

    control_mean = np.nanmean(
        X_control_train,
        axis=0,
    )
    control_std = np.nanstd(
        X_control_train,
        axis=0,
    )

    invalid_std = (
        ~np.isfinite(control_std)
        | (control_std <= 1e-12)
    )
    control_std[invalid_std] = np.nan

    X_train_z = (
        X_raw[train_idx] - control_mean
    ) / control_std

    X_test_z = (
        X_raw[test_idx] - control_mean
    ) / control_std

    # 只使用固定的16个经典显式特征
    X_train_score = X_train_z[:, score_feature_indices]
    X_test_score = X_test_z[:, score_feature_indices]

    directions = estimate_feature_directions(
        X_train_score,
        dose_numeric[train_idx],
    )

    aligned_test = (
        X_test_score
        * directions[None, :]
    )

    oof_score[test_idx] = calculate_late_score(
        aligned_test
    )
    fold_id[test_idx] = fold

    for feature_name, direction in zip(
        score_feature_names,
        directions,
    ):
        fold_direction_rows.append({
            "Fold": fold,
            "Feature": feature_name,
            "Direction": direction,
        })


# =========================
# 8. 特征方向稳定性
# =========================

direction_by_fold_df = pd.DataFrame(
    fold_direction_rows
)

direction_summary_df = (
    direction_by_fold_df
    .groupby("Feature", as_index=False)
    .agg(
        Mean_direction=("Direction", "mean"),
        Positive_folds=(
            "Direction",
            lambda x: int(np.sum(np.asarray(x) > 0)),
        ),
        Negative_folds=(
            "Direction",
            lambda x: int(np.sum(np.asarray(x) < 0)),
        ),
    )
)

direction_summary_df["Consensus_direction"] = np.where(
    direction_summary_df["Mean_direction"] < 0,
    -1.0,
    1.0,
)

direction_summary_df = direction_summary_df.set_index(
    "Feature"
).loc[score_feature_names].reset_index()

print("\nCross-fitted feature direction stability:")
print(direction_summary_df.to_string(index=False))


# =========================
# 9. 整理 OOF score dataframe
# =========================

df_late = pd.DataFrame({
    "Dose": dose_labels,
    "Dose_value": dose_numeric,
    "Late damage score": oof_score,
    "Sample": within_dose_sample,
    "Fold": fold_id,
})

df_late = df_late[
    np.isfinite(df_late["Late damage score"])
].copy()


# =========================
# 5. 保存每个剂量组的 OOF score
# =========================

score_file_map = {
    "0 ppb": "day5_0CT_late_damage_score_oof.npy",
    "50 ppb": "day5_50ppb_late_damage_score_oof.npy",
    "200 ppb": "day5_200ppb_late_damage_score_oof.npy",
    "500 ppb": "day5_500ppb_late_damage_score_oof.npy",
}

for dose in dose_order:
    vals = (
        df_late.loc[
            df_late["Dose"] == dose,
            "Late damage score",
        ]
        .to_numpy(dtype=float)
    )

    np.save(
        SCORE_DIR / score_file_map[dose],
        vals,
    )

direction_by_fold_df.to_csv(
    SCORE_DIR / "day5_feature_directions_by_fold.csv",
    index=False,
)

direction_summary_df.to_csv(
    SCORE_DIR / "day5_feature_direction_summary.csv",
    index=False,
)

df_late.to_csv(
    SCORE_DIR / "day5_late_damage_score_oof.csv",
    index=False,
)


# =========================
# 11. Image-level 线性剂量趋势检验
# 正式分析若存在同鱼多图，需改为 fish-level
# =========================

x = df_late["Dose_value"].to_numpy(dtype=float)
y = df_late["Late damage score"].to_numpy(dtype=float)

slope, intercept, r_value, p_value, slope_se = stats.linregress(
    x,
    y,
)

ci_low = slope - 1.96 * slope_se
ci_high = slope + 1.96 * slope_se

spearman_r, spearman_p = stats.spearmanr(
    x,
    y,
)

print("\nDay-5 cross-fitted late damage score trend:")
print(f"score mode = {SCORE_MODE}")
print(f"n = {len(df_late)} images")
print(f"linear slope = {slope:.6g}")
print(f"95% CI = [{ci_low:.6g}, {ci_high:.6g}]")
print(f"linear p = {p_value:.6g}")
print(f"Pearson r = {r_value:.6g}")
print(f"Spearman r = {spearman_r:.6g}")
print(f"Spearman p = {spearman_p:.6g}")


# =========================
# 12. Nature 风格参数
# =========================

dose_palette = {
    "0 ppb": "#3A6073",
    "50 ppb": "#6494AA",
    "200 ppb": "#DDA15E",
    "500 ppb": "#A64B4B",
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


def lighten_color(color, amount=0.5):
    c = np.asarray(to_rgb(color))
    return to_hex(
        c + (np.ones(3) - c) * amount
    )


box_palette = {
    key: lighten_color(value)
    for key, value in dose_palette.items()
}


# =========================
# 13. 绘制 late-damage score
# =========================

fig, ax = plt.subplots(figsize=(4.5, 3.5))

ax.set_axisbelow(True)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

sns.boxplot(
    data=df_late,
    x="Dose",
    y="Late damage score",
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
    data=df_late,
    x="Dose",
    y="Late damage score",
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
    linewidth=1,
    linestyle=(0, (2.2, 2.2)),
    zorder=0,
)

y_vals = df_late[
    "Late damage score"
].to_numpy(dtype=float)

lo, hi = np.nanpercentile(
    y_vals,
    [1, 99],
)

pad = 0.18 * (hi - lo + 1e-8)
ax.set_ylim(lo - pad, hi + pad)

ax.set_xlabel("")
ax.set_ylabel(
    "Late damage score at day 5\n"
    "(cross-fitted, direction-aligned z-score)",
    labelpad=4,
)

ax.text(
    -0.16,
    1.04,
    "(c)",
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
# 14. 保存图片
# =========================

if SAVE_FIGURE:
    '''
    fig.savefig(
        FIG_DIR / "fig2_c_day5_late_damage_score_oof.pdf",
        bbox_inches="tight",
        transparent=True,
    )

    fig.savefig(
        FIG_DIR / "fig2_c_day5_late_damage_score_oof.svg",
        bbox_inches="tight",
        transparent=True,
    )

    fig.savefig(
        FIG_DIR / "fig2_c_day5_late_damage_score_oof.tiff",
        dpi=600,
        bbox_inches="tight",
        transparent=False,
    )
    '''    
    '''
    fig.savefig(
        FIG_DIR / "fig2_c_day5_late_damage_score_oof.png",
        dpi=500,
        bbox_inches="tight",
        transparent=False,
    )
    '''

plt.show()
#plt.close(fig)

print("\n输出目录：")
print("Scores and tables:", SCORE_DIR.resolve())
print("Figures:", FIG_DIR.resolve())