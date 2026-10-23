# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pathlib import Path
from tqdm import trange
from scipy.spatial import KDTree

from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    binary_dilation,
    disk,
)
from skimage.measure import label, regionprops
from skimage import exposure, util, feature


# =========================
# 路径与数据设置
# =========================

root = Path("data/images")
seg_root = Path("data/segmentation")
output_root = Path("data/features")
output_root.mkdir(parents=True, exist_ok=True)

img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

for folder in root.iterdir():
    if folder.is_dir():
        nums = []

        for p in folder.iterdir():
            if p.suffix.lower() in img_exts and "-" in p.stem:
                try:
                    nums.append(int(p.stem.split("-")[0]))
                except ValueError:
                    pass

        if nums:
            print(folder.name, max(nums))


day5_folder_name_ls = ["0CT", "50ppb", "200ppb", "500ppb"]
day14_folder_name_ls = ["0CT", "50ppb", "200ppb", "500ppb"]

day5_sample_number_ls = [84, 84, 89, 91]
day14_sample_number_ls = [101, 127, 92, 74]


# =========================
# 19 个特征的固定顺序
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

    # 新增5个GLCM纹理特征
    "texture_glcm_contrast_std",
    "texture_glcm_energy_std",
    "texture_glcm_dissimilarity_std",
    "texture_glcm_correlation_std",
    "texture_glcm_homogeneity_mean",
]


# =========================
# 工具函数
# =========================

def safe_otsu(vals):
    vals = np.asarray(vals)

    if vals.size == 0:
        return np.nan

    if np.nanstd(vals) < 1e-8:
        return np.nanmean(vals)

    return threshold_otsu(vals)


def read_image(img_path):
    img = cv2.imdecode(
        np.fromfile(str(img_path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# =========================
# GLCM 纹理特征
# =========================

def extract_glcm_features(img):
    """
    在整张背景校正后的灰度图上计算 GLCM 纹理特征。

    参数
    ----
    img : ndarray, shape=(H, W, 3)
        RGB float32 图像，取值范围为 [0, 1]。

    GLCM 设置
    ---------
    distances = [1, 3]
    angles = [0°, 45°, 90°, 135°]

    每个 GLCM 属性产生 2 × 4 = 8 个值。

    返回
    ----
    dict
        contrast、energy、dissimilarity、correlation 的8组标准差，
        以及 homogeneity 的8组平均值。
    """

    gray = cv2.cvtColor(
        img.astype(np.float32),
        cv2.COLOR_RGB2GRAY,
    )

    # 估计并扣除缓慢变化的背景照明
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=12,
        sigmaY=12,
    )

    enhanced = gray - background

    # 将增强结果线性拉伸至 [0, 1]
    enhanced = exposure.rescale_intensity(
        enhanced,
        in_range="image",
        out_range=(0, 1),
    )

    # 转换为0到255的 uint8 灰度图
    gray_u8 = util.img_as_ubyte(enhanced)

    glcm = feature.graycomatrix(
        gray_u8,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = feature.graycoprops(
        glcm,
        "contrast",
    ).ravel()

    energy = feature.graycoprops(
        glcm,
        "energy",
    ).ravel()

    dissimilarity = feature.graycoprops(
        glcm,
        "dissimilarity",
    ).ravel()

    correlation = feature.graycoprops(
        glcm,
        "correlation",
    ).ravel()

    homogeneity = feature.graycoprops(
        glcm,
        "homogeneity",
    ).ravel()

    return {
        "texture_glcm_contrast_std": float(
            np.nanstd(contrast)
        ),
        "texture_glcm_energy_std": float(
            np.nanstd(energy)
        ),
        "texture_glcm_dissimilarity_std": float(
            np.nanstd(dissimilarity)
        ),
        "texture_glcm_correlation_std": float(
            np.nanstd(correlation)
        ),
        "texture_glcm_homogeneity_mean": float(
            np.nanmean(homogeneity)
        ),
    }


# =========================
# 只保留组织和紫色区域分割
# 细胞分割直接读取 Cellpose 标签
# =========================

def get_masks(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    gray = np.max(img, axis=2)
    gray_smooth = gaussian(gray, sigma=2)

    bg = np.percentile(gray_smooth, 5)
    fg = np.percentile(gray_smooth, 99)
    tissue_thr = bg + 0.08 * (fg - bg)

    tissue = gray_smooth > tissue_thr
    tissue = binary_dilation(tissue, disk(6))
    tissue = binary_closing(tissue, disk(8))
    tissue = remove_small_holes(
        tissue,
        area_threshold=500,
    )
    tissue = remove_small_objects(
        tissue,
        min_size=500,
    )

    if tissue.sum() == 0:
        tissue = gray_smooth > np.percentile(
            gray_smooth,
            70,
        )

    magenta = (r + b) / 2
    mag_tissue = magenta[tissue]
    mag_thr = safe_otsu(mag_tissue)

    if np.isnan(mag_thr):
        magenta_mask = np.zeros_like(
            tissue,
            dtype=bool,
        )
    else:
        magenta_mask = (
            (magenta > mag_thr)
            & tissue
            & (magenta > g)
        )

        magenta_mask = remove_small_objects(
            magenta_mask,
            min_size=10,
        )

    return tissue, magenta_mask


# =========================
# 读取 Cellpose 标签计算19个特征
# =========================

def extract_all_features(
    img,
    cell_labels,
    n_grid=8,
    distance_factor=1.5,
    close_distance_factor=0.75,
    dilation_radius=3,
):
    tissue, magenta_mask = get_masks(img)

    cell_labels = np.asarray(cell_labels)

    props = [
        p
        for p in regionprops(cell_labels)
        if p.area >= 20
    ]

    valid_labels = np.array(
        [p.label for p in props],
        dtype=cell_labels.dtype,
    )

    if len(valid_labels) > 0:
        green_mask = (
            np.isin(cell_labels, valid_labels)
            & tissue
        )
    else:
        green_mask = np.zeros_like(
            tissue,
            dtype=bool,
        )

    tissue_area = tissue.sum()
    cell_count = len(props)

    # =========================
    # 1. cell_density
    # =========================

    if tissue_area > 0:
        cell_density_val = (
            cell_count
            / tissue_area
            * 100000
        )
    else:
        cell_density_val = np.nan

    # =========================
    # 2. cell_roundness
    # =========================

    circularities = [
        4 * np.pi * p.area / (p.perimeter ** 2)
        for p in props
        if p.perimeter > 0
    ]

    if circularities:
        cell_roundness_val = np.mean(circularities)
    else:
        cell_roundness_val = np.nan

    # =========================
    # 3. nearest_neighbor_distance
    # =========================

    centroids = np.array(
        [p.centroid for p in props],
        dtype=float,
    )

    if len(centroids) >= 2:
        tree = KDTree(centroids)

        distances, _ = tree.query(
            centroids,
            k=2,
        )

        nearest_neighbor_distance_val = (
            distances[:, 1].mean()
        )
    else:
        nearest_neighbor_distance_val = np.nan

    # =========================
    # 4. local_density_variance
    # =========================

    h, w = green_mask.shape
    densities = []

    for i in range(n_grid):
        for j in range(n_grid):
            y1 = i * h // n_grid
            y2 = (i + 1) * h // n_grid
            x1 = j * w // n_grid
            x2 = (j + 1) * w // n_grid

            tile_tissue = tissue[
                y1:y2,
                x1:x2,
            ]

            tile_green = green_mask[
                y1:y2,
                x1:x2,
            ]

            if tile_tissue.sum() > 100:
                densities.append(
                    tile_green.sum()
                    / tile_tissue.sum()
                )

    densities = np.asarray(
        densities,
        dtype=float,
    )

    if len(densities) > 0:
        local_density_variance_val = np.var(
            densities
        )
    else:
        local_density_variance_val = np.nan

    # =========================
    # 5. channel_colocalization
    # =========================

    if tissue_area > 0:
        g_vals = img[:, :, 1][tissue]

        mag_vals = (
            (
                img[:, :, 0]
                + img[:, :, 2]
            )
            / 2
        )[tissue]

        if (
            np.std(g_vals) > 1e-8
            and np.std(mag_vals) > 1e-8
        ):
            channel_colocalization_val = np.corrcoef(
                g_vals,
                mag_vals,
            )[0, 1]
        else:
            channel_colocalization_val = np.nan
    else:
        channel_colocalization_val = np.nan

    # =========================
    # 6. reduced_green_cell_density
    # =========================

    reduced_green_cell_density_val = (
        -cell_density_val
    )

    # =========================
    # 7. spatial_heterogeneity_increase
    # =========================

    if len(densities) > 0:
        spatial_heterogeneity_increase_val = (
            densities.std()
            / (densities.mean() + 1e-8)
        )
    else:
        spatial_heterogeneity_increase_val = np.nan

    # =========================
    # 8. green_mean_cell_area
    # 9. green_cell_area_cv
    # =========================

    areas = np.array(
        [p.area for p in props],
        dtype=float,
    )

    if len(areas) > 0:
        green_mean_cell_area_val = areas.mean()

        green_cell_area_cv_val = (
            areas.std()
            / (areas.mean() + 1e-8)
        )
    else:
        green_mean_cell_area_val = np.nan
        green_cell_area_cv_val = np.nan

    # =========================
    # 10. green_mean_eccentricity
    # =========================

    eccentricities = np.array(
        [p.eccentricity for p in props],
        dtype=float,
    )

    if len(eccentricities) > 0:
        green_mean_eccentricity_val = (
            eccentricities.mean()
        )
    else:
        green_mean_eccentricity_val = np.nan

    # =========================
    # 11. green_neighbor_degree
    # 12. green_close_neighbor_fraction
    # =========================

    diameters = np.array(
        [p.equivalent_diameter for p in props],
        dtype=float,
    )

    if (
        len(centroids) >= 2
        and len(diameters) >= 2
    ):
        median_diameter = np.median(diameters)
        tree = KDTree(centroids)

        pairs = tree.query_pairs(
            r=median_diameter * distance_factor
        )

        degree = np.zeros(
            len(centroids),
            dtype=float,
        )

        for a, b in pairs:
            degree[a] += 1
            degree[b] += 1

        close_pairs = tree.query_pairs(
            r=(
                median_diameter
                * close_distance_factor
            )
        )

        close_connected = set()

        for a, b in close_pairs:
            close_connected.add(a)
            close_connected.add(b)

        green_neighbor_degree_val = degree.mean()

        green_close_neighbor_fraction_val = (
            len(close_connected)
            / len(centroids)
        )
    else:
        green_neighbor_degree_val = np.nan
        green_close_neighbor_fraction_val = np.nan

    # =========================
    # 13. green_dilation_connectivity
    # =========================

    if cell_count > 0:
        dilated = binary_dilation(
            green_mask,
            disk(dilation_radius),
        )

        n_after = label(dilated).max()

        green_dilation_connectivity_val = (
            1
            - n_after
            / (cell_count + 1e-8)
        )
    else:
        green_dilation_connectivity_val = np.nan

    # =========================
    # 14. magenta_largest_component_fraction
    # =========================

    labeled_mag = label(magenta_mask)
    counts = np.bincount(
        labeled_mag.ravel()
    )

    if (
        len(counts) > 1
        and magenta_mask.sum() > 0
    ):
        largest = counts[1:].max()

        magenta_largest_component_fraction_val = (
            largest
            / (magenta_mask.sum() + 1e-8)
        )
    else:
        magenta_largest_component_fraction_val = np.nan

    # =========================
    # 组合原有14个特征
    # =========================

    features_dict = {
        "cell_density": cell_density_val,
        "cell_roundness": cell_roundness_val,
        "nearest_neighbor_distance": nearest_neighbor_distance_val,
        "local_density_variance": local_density_variance_val,
        "channel_colocalization": channel_colocalization_val,
        "reduced_green_cell_density": reduced_green_cell_density_val,
        "spatial_heterogeneity_increase": spatial_heterogeneity_increase_val,
        "green_mean_cell_area": green_mean_cell_area_val,
        "green_cell_area_cv": green_cell_area_cv_val,
        "green_mean_eccentricity": green_mean_eccentricity_val,
        "green_neighbor_degree": green_neighbor_degree_val,
        "green_close_neighbor_fraction": green_close_neighbor_fraction_val,
        "green_dilation_connectivity": green_dilation_connectivity_val,
        "magenta_largest_component_fraction": magenta_largest_component_fraction_val,
    }

    # =========================
    # 添加5个 GLCM 特征
    # =========================

    features_dict.update(
        extract_glcm_features(img)
    )

    return features_dict


# =========================
# 提取某一天、某个剂量组特征
# =========================

def extract_group_features(
    day_prefix,
    folder_name,
    sample_number,
    verbose_nan=False,
):
    f_ls = []

    for i in trange(
        1,
        sample_number + 1,
        desc=f"{day_prefix} {folder_name}",
    ):
        for j in range(1, 5):
            img_path = (
                root
                / f"{day_prefix} {folder_name}"
                / f"{i}-{j}.png"
            )

            label_path = (
                seg_root
                / f"{day_prefix} {folder_name}"
                / f"{i}-{j}_labels.npy"
            )

            if not img_path.exists():
                print(
                    "原图不存在:",
                    img_path,
                )
                continue

            if not label_path.exists():
                print(
                    "分割标签不存在:",
                    label_path,
                )
                continue

            img = read_image(img_path)

            if img is None:
                print(
                    "原图读取失败:",
                    img_path,
                )
                continue

            cell_labels = np.load(label_path)

            if cell_labels.shape != img.shape[:2]:
                print(
                    "原图和标签尺寸不一致:",
                    img_path,
                    img.shape[:2],
                    label_path,
                    cell_labels.shape,
                )
                continue

            features_dict = extract_all_features(
                img,
                cell_labels,
            )

            features = np.array(
                [
                    features_dict[name]
                    for name in feature_names
                ],
                dtype=np.float32,
            )

            f_ls.append(features)

            if (
                verbose_nan
                and np.any(np.isnan(features))
            ):
                nan_names = [
                    feature_names[k]
                    for k in np.where(
                        np.isnan(features)
                    )[0]
                ]

                print(
                    "存在 nan:",
                    img_path,
                    nan_names,
                )

    if len(f_ls) == 0:
        return np.empty(
            (len(feature_names), 0),
            dtype=np.float32,
        )

    # 输出形状：
    # (特征数, 图像数) = (19, N)
    return np.asarray(
        f_ls,
        dtype=np.float32,
    ).T


# =========================
# 计算day5、day14，各4个剂量组
# =========================

day5_0CT_arr = extract_group_features(
    day_prefix="day5",
    folder_name="0CT",
    sample_number=day5_sample_number_ls[0],
    verbose_nan=False,
)

day5_50ppb_arr = extract_group_features(
    day_prefix="day5",
    folder_name="50ppb",
    sample_number=day5_sample_number_ls[1],
    verbose_nan=False,
)

day5_200ppb_arr = extract_group_features(
    day_prefix="day5",
    folder_name="200ppb",
    sample_number=day5_sample_number_ls[2],
    verbose_nan=False,
)

day5_500ppb_arr = extract_group_features(
    day_prefix="day5",
    folder_name="500ppb",
    sample_number=day5_sample_number_ls[3],
    verbose_nan=False,
)

day14_0CT_arr = extract_group_features(
    day_prefix="day14",
    folder_name="0CT",
    sample_number=day14_sample_number_ls[0],
    verbose_nan=False,
)

day14_50ppb_arr = extract_group_features(
    day_prefix="day14",
    folder_name="50ppb",
    sample_number=day14_sample_number_ls[1],
    verbose_nan=False,
)

day14_200ppb_arr = extract_group_features(
    day_prefix="day14",
    folder_name="200ppb",
    sample_number=day14_sample_number_ls[2],
    verbose_nan=False,
)

day14_500ppb_arr = extract_group_features(
    day_prefix="day14",
    folder_name="500ppb",
    sample_number=day14_sample_number_ls[3],
    verbose_nan=False,
)


# =========================
# 汇总数组
# =========================

arrays = {
    "day5_0CT_arr": day5_0CT_arr,
    "day5_50ppb_arr": day5_50ppb_arr,
    "day5_200ppb_arr": day5_200ppb_arr,
    "day5_500ppb_arr": day5_500ppb_arr,
    "day14_0CT_arr": day14_0CT_arr,
    "day14_50ppb_arr": day14_50ppb_arr,
    "day14_200ppb_arr": day14_200ppb_arr,
    "day14_500ppb_arr": day14_500ppb_arr,
}


# =========================
# 检查 shape
# =========================

print("\nFeature count:", len(feature_names))
print("Feature names:")

for index, name in enumerate(
    feature_names,
    start=1,
):
    print(f"{index:02d}. {name}")

print()

for name, arr in arrays.items():
    print(name, "shape:", arr.shape)

    assert arr.ndim == 2
    assert arr.shape[0] == len(feature_names), (
        f"{name} 特征维度错误："
        f"{arr.shape[0]} != {len(feature_names)}"
    )


# =========================
# 检查 NaN 和无穷值数量
# =========================

print("\nNaN and infinite-value check:")

for name, arr in arrays.items():
    print(
        name,
        "nan count:",
        int(np.isnan(arr).sum()),
        "inf count:",
        int(np.isinf(arr).sum()),
        "total:",
        arr.size,
    )


# =========================
# 按特征检查 NaN 数量
# =========================

print("\nNaN count by feature:")

for array_name, arr in arrays.items():
    print(f"\n{array_name}")

    for feature_name, row in zip(
        feature_names,
        arr,
    ):
        nan_count = int(np.isnan(row).sum())

        if nan_count > 0:
            print(
                feature_name,
                "nan:",
                nan_count,
                "/",
                row.size,
            )


# =========================
# 保存结果
# =========================

np.save(
    output_root / "feature_names.npy",
    np.asarray(feature_names),
)

np.save(
    output_root / "day5_0CT_arr.npy",
    day5_0CT_arr,
)

np.save(
    output_root / "day5_50ppb_arr.npy",
    day5_50ppb_arr,
)

np.save(
    output_root / "day5_200ppb_arr.npy",
    day5_200ppb_arr,
)

np.save(
    output_root / "day5_500ppb_arr.npy",
    day5_500ppb_arr,
)

np.save(
    output_root / "day14_0CT_arr.npy",
    day14_0CT_arr,
)

np.save(
    output_root / "day14_50ppb_arr.npy",
    day14_50ppb_arr,
)

np.save(
    output_root / "day14_200ppb_arr.npy",
    day14_200ppb_arr,
)

np.save(
    output_root / "day14_500ppb_arr.npy",
    day14_500ppb_arr,
)

print("\n结果已保存到:", output_root.resolve())