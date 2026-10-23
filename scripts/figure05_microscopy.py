# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:11:54 2026

@author: Administrator
"""
# -*- coding: utf-8 -*-
# Time labels harmonized to Day 5 / Day 14 following author confirmation.
"""
Microscopy-centered three-feature figure (real + mock modes)

Layout
------
Left column:
    a. Representative raw confocal images
    b. Cell roundness visualization
Middle column:
    c. Cell eccentricity visualization
    d. Local texture-contrast visualization
Right column:
    e. Roundness distributions (Day 5 / Day 14)
    f. Eccentricity distributions (Day 5 / Day 14)
    g. GLCM contrast-SD distributions (Day 5 / Day 14)

The same four representative images are used in panels a-d:
    Day 5 : 0 ug/L, 500 ug/L
    Day 14: 0 ug/L, 500 ug/L

Real feature definitions
------------------------
Roundness:
    4*pi*area/perimeter^2 for each segmented cell instance.
Eccentricity:
    skimage.measure.regionprops(...).eccentricity for each cell instance.
    The corresponding image-level descriptor is the mean across cells
    (green_mean_eccentricity).
Texture:
    The right-side distribution uses the exact image-level
    texture_glcm_contrast_std calculated from the background-corrected image
    with distances [1, 3] and angles [0, 45, 90, 135 degrees].

Panel d intentionally uses a more intuitive LOCAL RMS contrast image as a
spatial visualization of texture heterogeneity. It is a visual proxy showing
WHERE texture variation occurs; it is not claimed to be a pixel-wise GLCM map.

Data levels
-----------
- Roundness distribution: image-level mean cell roundness.
- Eccentricity distribution: image-level mean cell eccentricity.
- GLCM contrast SD distribution: image-level.

Expected real paths
-------------------
data/images\\day5 0CT\\1-1.png
data/images\\day5 500ppb\\...
data/images\\day14 0CT\\...
data/images\\day14 500ppb\\...
"""

from pathlib import Path
import re
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from skimage import exposure, util, feature
from skimage.measure import regionprops, find_contours


# =============================================================================
# 1. SETTINGS
# =============================================================================

# Use "real" for the project data; change to "mock" to test without project files.
DATA_MODE = "real"

PROJECT_ROOT = Path(".")
IMAGE_ROOT = PROJECT_ROOT / "data/images"
SEG_ROOT_CANDIDATES = [
    PROJECT_ROOT / "data/segmentation",
    PROJECT_ROOT / "data/segmentation",
]

FIG_DIR = PROJECT_ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PNG = True
SAVE_PDF = True
SAVE_TABLES = True
DPI = 500

USE_CACHE = False
FORCE_RECOMPUTE = False
CACHE_IMAGE_SUMMARY = FIG_DIR / "microscopy_3feature_image_summary.csv"
CACHE_CELL_LONG = FIG_DIR / "microscopy_3feature_cell_long.csv"

GROUPS = [
    ("Day 5", "0 ug/L"),
    ("Day 5", "500 ug/L"),
    ("Day 14", "0 ug/L"),
    ("Day 14", "500 ug/L"),
]
DISPLAY_DOSE = {"0 ug/L": "0 µg/L", "500 ug/L": "500 µg/L"}

GROUP_FOLDER_CANDIDATES = {
    ("Day 5", "0 ug/L"): ["day5 0CT", "day50CT", "day5 0ct", "day50ct"],
    ("Day 5", "500 ug/L"): ["day5 500ppb", "day5500ppb", "day5 500PPB", "day5500PPB"],
    ("Day 14", "0 ug/L"): ["day14 0CT", "day140CT", "day14 0ct", "day140ct"],
    ("Day 14", "500 ug/L"): ["day14 500ppb", "day14500ppb", "day14 500PPB", "day14500PPB"],
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_CELL_AREA = 20

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
LOCAL_TEXTURE_SIGMA = 2.5

MAX_CELL_RUG_POINTS = 220
MOCK_IMAGES_PER_GROUP = 8

COLOR_0 = "#24478F"
COLOR_500 = "#E9473F"
ROUNDNESS_CMAP = "viridis"
ECCENTRICITY_CMAP = "plasma"
TEXTURE_CMAP = "magma"


# =============================================================================
# 2. PLOT STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =============================================================================
# 3. BASIC HELPERS
# =============================================================================

def natural_sort_key(path_or_name):
    parts = re.split(r"(\d+)", Path(path_or_name).name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def finite_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def robust_range(values, low=1.0, high=99.0, pad_fraction=0.06, fallback=(0, 1)):
    vals = finite_array(values)
    if len(vals) == 0:
        return fallback
    lo, hi = np.nanpercentile(vals, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return fallback
    if hi <= lo:
        delta = max(abs(lo) * 0.05, 1e-6)
        return lo - delta, hi + delta
    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


# =============================================================================
# 4. REAL PATH HELPERS
# =============================================================================

def resolve_seg_root():
    for path in SEG_ROOT_CANDIDATES:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        "找不到细胞分割根目录。\n尝试过：\n  - "
        + "\n  - ".join(str(x) for x in SEG_ROOT_CANDIDATES)
    )


def resolve_group_folder(day, dose):
    checked = []
    for folder_name in GROUP_FOLDER_CANDIDATES[(day, dose)]:
        path = IMAGE_ROOT / folder_name
        checked.append(str(path))
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        f"找不到原图文件夹：{day} | {dose}\n尝试过：\n  - "
        + "\n  - ".join(checked)
    )


def list_group_images(day, dose):
    folder = resolve_group_folder(day, dose)
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files = sorted(files, key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"文件夹中没有图片：{folder}")
    return folder, files


def build_label_index(seg_root):
    index = {}
    print("[Label index] scanning:", seg_root)
    for path in seg_root.rglob("*_labels.npy"):
        stem = path.name[:-len("_labels.npy")]
        index.setdefault(stem, []).append(path)
    print("[Label index] stems:", len(index))
    return index


def find_label_path(img_path, seg_root, label_index):
    expected = seg_root / img_path.parent.name / f"{img_path.stem}_labels.npy"
    if expected.exists():
        return expected

    matches = label_index.get(img_path.stem, [])
    same_group = [
        p for p in matches
        if img_path.parent.name.lower() in str(p.parent).lower()
    ]
    if len(same_group) == 1:
        return same_group[0]
    if len(same_group) > 1:
        same_group = sorted(same_group, key=lambda x: len(str(x)))
        warnings.warn(f"{img_path.name} 找到多个同组标签，使用：{same_group[0]}")
        return same_group[0]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"{img_path.name} 找到多个标签，但无法确认对应关系：\n"
            + "\n".join(str(x) for x in matches[:20])
        )
    raise FileNotFoundError(f"找不到标签：{img_path.name}\n预期：{expected}")


def read_rgb_image(img_path):
    data = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"原图读取失败：{img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# =============================================================================
# 5. FEATURE EXTRACTION
# =============================================================================

def calculate_glcm_analysis(img):
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=12, sigmaY=12)
    residual = gray - background

    if (
        not np.all(np.isfinite(residual))
        or np.nanmax(residual) - np.nanmin(residual) <= 1e-12
    ):
        enhanced = np.zeros_like(residual, dtype=np.float32)
    else:
        enhanced = exposure.rescale_intensity(
            residual, in_range="image", out_range=(0, 1)
        ).astype(np.float32)

    gray_u8 = util.img_as_ubyte(np.clip(enhanced, 0, 1))
    glcm = feature.graycomatrix(
        gray_u8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast_matrix = feature.graycoprops(glcm, "contrast").astype(float)
    return {
        "enhanced_gray": enhanced,
        "texture_glcm_contrast_std": float(np.nanstd(contrast_matrix)),
    }


def calculate_local_texture_map(enhanced_gray):
    gray = enhanced_gray.astype(np.float32)
    mean = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=LOCAL_TEXTURE_SIGMA, sigmaY=LOCAL_TEXTURE_SIGMA
    )
    mean_sq = cv2.GaussianBlur(
        gray ** 2, (0, 0), sigmaX=LOCAL_TEXTURE_SIGMA, sigmaY=LOCAL_TEXTURE_SIGMA
    )
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var).astype(np.float32)


def props_to_cell_metrics(props):
    rows = []
    for prop in props:
        if prop.perimeter <= 0:
            continue
        roundness = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        rows.append({
            "label": int(prop.label),
            "cell_roundness": float(roundness),
            "cell_eccentricity": float(prop.eccentricity),
        })
    return rows


def extract_real_image_record(img_path, day, dose, seg_root, label_index):
    img = read_rgb_image(img_path)
    label_path = find_label_path(img_path, seg_root, label_index)
    labels = np.load(label_path)
    props = [p for p in regionprops(labels) if p.area >= MIN_CELL_AREA]
    cell_metrics = props_to_cell_metrics(props)

    rvals = np.asarray([x["cell_roundness"] for x in cell_metrics], dtype=float)
    evals = np.asarray([x["cell_eccentricity"] for x in cell_metrics], dtype=float)
    glcm = calculate_glcm_analysis(img)

    image_record = {
        "day": day,
        "dose": dose,
        "image_name": img_path.name,
        "image_stem": img_path.stem,
        "image_path": str(img_path),
        "label_path": str(label_path),
        "n_cells": int(len(cell_metrics)),
        "mean_cell_roundness": float(np.nanmean(rvals)) if len(rvals) else np.nan,
        "green_mean_eccentricity": float(np.nanmean(evals)) if len(evals) else np.nan,
        "texture_glcm_contrast_std": glcm["texture_glcm_contrast_std"],
    }

    cell_rows = [
        {
            "day": day,
            "dose": dose,
            "image_name": img_path.name,
            "image_stem": img_path.stem,
            "cell_label": row["label"],
            "cell_roundness": row["cell_roundness"],
            "cell_eccentricity": row["cell_eccentricity"],
        }
        for row in cell_metrics
    ]
    return image_record, cell_rows


# =============================================================================
# 6. MOCK DATA
# =============================================================================

def mock_condition_parameters(day, dose):
    if day == "Day 5" and dose == "0 ug/L":
        return 0.20, 0.48, 0.025, 35
    if day == "Day 5" and dose == "500 ug/L":
        return 0.35, 0.73, 0.055, 62
    if day == "Day 14" and dose == "0 ug/L":
        return 0.24, 0.57, 0.032, 42
    return 0.48, 0.88, 0.080, 76


def make_mock_image_and_labels(day, dose, seed):
    rng = np.random.default_rng(seed)
    h = w = 192
    e_lo, e_hi, noise_sd, red_spots = mock_condition_parameters(day, dose)

    green = rng.normal(0.015, noise_sd, (h, w)).astype(np.float32)
    red = rng.normal(0.008, noise_sd * 0.55, (h, w)).astype(np.float32)
    labels = np.zeros((h, w), dtype=np.int32)

    n_cells = 30 if dose == "0 ug/L" else 27
    if day == "Day 14":
        n_cells += 2

    for label_id in range(1, n_cells + 1):
        ecc = rng.uniform(e_lo, e_hi)
        a = rng.uniform(10, 16)
        b = a * np.sqrt(max(1 - ecc ** 2, 0.05))
        cx = int(rng.uniform(18, w - 18))
        cy = int(rng.uniform(18, h - 18))
        angle = float(rng.uniform(0, 180))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (cx, cy),
            (max(2, int(a)), max(2, int(b))),
            angle,
            0,
            360,
            1,
            -1,
        )
        mask = (mask > 0) & (labels == 0)
        if np.sum(mask) < MIN_CELL_AREA:
            continue
        labels[mask] = label_id
        green[mask] += rng.uniform(0.25, 0.55)

    yy, xx = np.indices((h, w))
    for _ in range(red_spots):
        x0 = rng.uniform(0, w)
        y0 = rng.uniform(0, h)
        sigma = rng.uniform(1.0, 3.2)
        amp = rng.uniform(0.12, 0.45)
        rr = (xx - x0) ** 2 + (yy - y0) ** 2
        red += amp * np.exp(-rr / (2 * sigma ** 2))

    green = np.clip(green, 0, None)
    red = np.clip(red, 0, None)
    green /= max(np.percentile(green, 99.8), 1e-8)
    red /= max(np.percentile(red, 99.8), 1e-8)
    green = np.clip(green, 0, 1)
    red = np.clip(red, 0, 1)

    img = np.dstack([red, green, np.zeros_like(red)]).astype(np.float32)
    return img, labels


def build_mock_tables():
    image_rows = []
    cell_rows = []
    storage = {}

    for group_idx, (day, dose) in enumerate(GROUPS):
        for image_idx in range(MOCK_IMAGES_PER_GROUP):
            img, labels = make_mock_image_and_labels(
                day, dose, seed=10000 + group_idx * 1000 + image_idx
            )
            props = [p for p in regionprops(labels) if p.area >= MIN_CELL_AREA]
            metrics = props_to_cell_metrics(props)
            rvals = np.asarray([x["cell_roundness"] for x in metrics], dtype=float)
            evals = np.asarray([x["cell_eccentricity"] for x in metrics], dtype=float)
            glcm = calculate_glcm_analysis(img)
            stem = f"mock_{image_idx:03d}"

            storage[(day, dose, stem)] = {
                "img": img,
                "labels": labels,
                "local_texture_map": calculate_local_texture_map(glcm["enhanced_gray"]),
                "image_name": stem + ".png",
            }

            image_rows.append({
                "day": day,
                "dose": dose,
                "image_name": stem + ".png",
                "image_stem": stem,
                "image_path": f"MOCK/{day}/{dose}/{stem}.png",
                "label_path": f"MOCK/{day}/{dose}/{stem}_labels.npy",
                "n_cells": len(metrics),
                "mean_cell_roundness": float(np.nanmean(rvals)) if len(rvals) else np.nan,
                "green_mean_eccentricity": float(np.nanmean(evals)) if len(evals) else np.nan,
                "texture_glcm_contrast_std": glcm["texture_glcm_contrast_std"],
            })

            for row in metrics:
                cell_rows.append({
                    "day": day,
                    "dose": dose,
                    "image_name": stem + ".png",
                    "image_stem": stem,
                    "cell_label": row["label"],
                    "cell_roundness": row["cell_roundness"],
                    "cell_eccentricity": row["cell_eccentricity"],
                })

    return pd.DataFrame(image_rows), pd.DataFrame(cell_rows), storage


# =============================================================================
# 7. REAL TABLES + CACHE
# =============================================================================

def build_real_tables():
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"原图根目录不存在：{IMAGE_ROOT}")

    seg_root = resolve_seg_root()
    label_index = build_label_index(seg_root)
    image_rows = []
    cell_rows = []

    for day, dose in GROUPS:
        folder, files = list_group_images(day, dose)
        print(f"[Group] {day} | {dose} | {folder} | n={len(files)}")
        for idx, img_path in enumerate(files, start=1):
            if idx == 1 or idx % 50 == 0 or idx == len(files):
                print(f"  [{idx}/{len(files)}] {img_path.name}")
            try:
                image_record, cells = extract_real_image_record(
                    img_path, day, dose, seg_root, label_index
                )
                image_rows.append(image_record)
                cell_rows.extend(cells)
            except Exception as exc:
                print("  [SKIP]", img_path, type(exc).__name__, exc)

    return pd.DataFrame(image_rows), pd.DataFrame(cell_rows)


def load_or_build_real_tables():
    if (
        USE_CACHE
        and not FORCE_RECOMPUTE
        and CACHE_IMAGE_SUMMARY.exists()
        and CACHE_CELL_LONG.exists()
    ):
        image_df = pd.read_csv(CACHE_IMAGE_SUMMARY)
        cell_df = pd.read_csv(CACHE_CELL_LONG)
        req_i = {
            "day", "dose", "image_path", "label_path",
            "mean_cell_roundness", "green_mean_eccentricity",
            "texture_glcm_contrast_std",
        }
        req_c = {"day", "dose", "cell_roundness", "cell_eccentricity"}
        if req_i.issubset(image_df.columns) and req_c.issubset(cell_df.columns):
            print("[Cache] loaded.")
            return image_df, cell_df
        print("[Cache] old/incomplete cache; recomputing.")

    image_df, cell_df = build_real_tables()
    if USE_CACHE:
        image_df.to_csv(CACHE_IMAGE_SUMMARY, index=False, encoding="utf-8-sig")
        cell_df.to_csv(CACHE_CELL_LONG, index=False, encoding="utf-8-sig")
    return image_df, cell_df


# =============================================================================
# 8. REPRESENTATIVE IMAGE SELECTION
# =============================================================================

def select_representative_images(image_df):
    feature_columns = [
        "mean_cell_roundness",
        "green_mean_eccentricity",
        "texture_glcm_contrast_std",
    ]
    chosen = []

    for day, dose in GROUPS:
        sub = image_df[(image_df["day"] == day) & (image_df["dose"] == dose)].copy()
        for col in feature_columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=feature_columns).copy()
        if len(sub) == 0:
            raise RuntimeError(f"{day} | {dose} 没有有效图片可用于代表图选择。")

        X = sub[feature_columns].to_numpy(dtype=float)
        center = np.nanmedian(X, axis=0)
        scale = np.nanstd(X, axis=0)
        scale[(~np.isfinite(scale)) | (scale <= 1e-12)] = 1.0
        distance = np.sqrt(np.sum(((X - center) / scale) ** 2, axis=1))
        sub["distance_to_3feature_center"] = distance
        sub = sub.sort_values(
            ["distance_to_3feature_center", "image_name"], kind="mergesort"
        )
        chosen.append(sub.iloc[0])

    return pd.DataFrame(chosen).reset_index(drop=True)


def load_real_representative_bundle(selection_df):
    bundle = {}
    for _, row in selection_df.iterrows():
        img = read_rgb_image(Path(row["image_path"]))
        labels = np.load(Path(row["label_path"]))
        glcm = calculate_glcm_analysis(img)
        bundle[(row["day"], row["dose"])] = {
            "img": img,
            "labels": labels,
            "local_texture_map": calculate_local_texture_map(glcm["enhanced_gray"]),
            "image_name": row["image_name"],
        }
    return bundle


def load_mock_representative_bundle(selection_df, storage):
    return {
        (row["day"], row["dose"]): storage[(row["day"], row["dose"], row["image_stem"])]
        for _, row in selection_df.iterrows()
    }


# =============================================================================
# 9. INSTANCE VISUALIZATIONS
# =============================================================================

def metric_by_label(labels, metric_name):
    result = {}
    for prop in regionprops(labels):
        if prop.area < MIN_CELL_AREA:
            continue
        if metric_name == "roundness":
            if prop.perimeter <= 0:
                continue
            value = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        elif metric_name == "eccentricity":
            value = float(prop.eccentricity)
        else:
            raise ValueError(metric_name)
        result[int(prop.label)] = float(value)
    return result


def draw_roundness_boundaries(ax, img, labels, norm):
    ax.imshow(img)
    cmap = plt.get_cmap(ROUNDNESS_CMAP)
    values = metric_by_label(labels, "roundness")
    for label_id, value in values.items():
        contours = find_contours((labels == label_id).astype(float), 0.5)
        color = cmap(norm(value))
        for contour in contours:
            ax.plot(
                contour[:, 1], contour[:, 0],
                color=color, linewidth=0.85, alpha=0.98,
            )
    hide_axes(ax)


def draw_eccentricity_fill(ax, img, labels, norm):
    """
    Eccentricity visualization without fitted ellipses.

    Visualization follows the original feature-visualization logic:
        - one colored centroid per segmented cell
        - one major-axis direction line per segmented cell
        - color encodes cell eccentricity

    No fitted ellipse is drawn.
    """
    ax.imshow(img)

    cmap = plt.get_cmap(ECCENTRICITY_CMAP)

    for prop in regionprops(labels):
        if prop.area < MIN_CELL_AREA:
            continue

        ecc = float(prop.eccentricity)
        cy, cx = prop.centroid
        major = float(prop.major_axis_length)
        orientation = float(prop.orientation)

        color = cmap(norm(ecc))

        # Same geometric idea as the original feature-visualization script:
        # display the major-axis orientation of each cell.
        length = max(3.0, major / 2.0)
        theta = -orientation

        ax.scatter(
            cx,
            cy,
            s=10,
            color=[color],
            edgecolors="none",
            zorder=3,
        )

        ax.plot(
            [
                cx - length * np.cos(theta),
                cx + length * np.cos(theta),
            ],
            [
                cy - length * np.sin(theta),
                cy + length * np.sin(theta),
            ],
            color=color,
            linewidth=0.85,
            alpha=0.95,
            zorder=2,
        )

    h, w = img.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    hide_axes(ax)


# =============================================================================
# 10. PANEL BUILDERS
# =============================================================================

def create_image_panel(
    fig, outer_spec, panel_letter, title, bundle, mode,
    norm=None, cmap_name=None, colorbar_label=None,
):
    has_colorbar = (
        norm is not None
        and cmap_name is not None
    )

    # Feature panels reserve a dedicated bottom row for the colorbar.
    if has_colorbar:
        panel_gs = GridSpecFromSubplotSpec(
            3, 1,
            subplot_spec=outer_spec,
            height_ratios=[0.12, 0.78, 0.10],
            hspace=0.025,
        )
        image_spec = panel_gs[1, 0]
        colorbar_spec = panel_gs[2, 0]
    else:
        panel_gs = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_spec,
            height_ratios=[0.12, 0.88],
            hspace=0.01,
        )
        image_spec = panel_gs[1, 0]
        colorbar_spec = None

    header = fig.add_subplot(panel_gs[0, 0])
    hide_axes(header)

    header.text(
        0.00, 0.55, panel_letter,
        ha="left", va="center",
        fontsize=12, fontweight="bold",
    )
    if title:
        header.text(
            0.09, 0.55, title,
            ha="left", va="center",
            fontsize=14, fontweight="bold",
        )

    image_gs = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=image_spec,
        wspace=0.06,
        hspace=0.08,
    )

    conditions = [
        ("Day 5", "0 ug/L"),
        ("Day 5", "500 ug/L"),
        ("Day 14", "0 ug/L"),
        ("Day 14", "500 ug/L"),
    ]

    for idx, cond in enumerate(conditions):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(image_gs[row, col])
        item = bundle[cond]

        if mode == "raw":
            ax.imshow(item["img"])
            hide_axes(ax)

        elif mode == "roundness":
            draw_roundness_boundaries(
                ax,
                item["img"],
                item["labels"],
                norm,
            )

        elif mode == "eccentricity":
            draw_eccentricity_fill(
                ax,
                item["img"],
                item["labels"],
                norm,
            )

        elif mode == "texture":
            ax.imshow(
                item["local_texture_map"],
                cmap=TEXTURE_CMAP,
                norm=norm,
            )
            hide_axes(ax)

        else:
            raise ValueError(mode)

        if row == 0:
            ax.set_title(
                DISPLAY_DOSE.get(cond[1], cond[1]),
                fontsize=13,
                fontweight="bold",
                pad=3,
            )

        if col == 0:
            ax.text(
                -0.08,
                0.5,
                "Day 14" if cond[0] == "Day 14" else cond[0],
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

    # Dedicated horizontal colorbar below the whole panel.
    if has_colorbar:
        cb_holder = fig.add_subplot(colorbar_spec)
        hide_axes(cb_holder)

        cax = cb_holder.inset_axes(
            [0.16, 0.48, 0.68, 0.30]
        )

        sm = ScalarMappable(
            norm=norm,
            cmap=plt.get_cmap(cmap_name),
        )

        cb = fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
        )

        cb.ax.tick_params(
            labelsize=12,
            length=1.8,
            pad=1.2,
        )
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.xaxis.set_label_position("bottom")
        if colorbar_label:
            cb.set_label(colorbar_label, fontsize=13, labelpad=3)
        cb.outline.set_linewidth(0.5)


def draw_hist_comparison(ax, vals0, vals500, xlim, xlabel, day_title, level_label):
    vals0 = finite_array(vals0)
    vals500 = finite_array(vals500)
    bins = np.linspace(xlim[0], xlim[1], 51)

    ax.hist(
        vals0, bins=bins, density=True, histtype="stepfilled",
        color=COLOR_0, alpha=0.18, edgecolor=COLOR_0, linewidth=0.9,
    )
    ax.hist(
        vals500, bins=bins, density=True, histtype="stepfilled",
        color=COLOR_500, alpha=0.18, edgecolor=COLOR_500, linewidth=0.9,
    )

    if len(vals0):
        ax.axvline(np.nanmedian(vals0), color=COLOR_0,
                   linewidth=0.85, linestyle=(0, (3, 2)))
    if len(vals500):
        ax.axvline(np.nanmedian(vals500), color=COLOR_500,
                   linewidth=0.85, linestyle=(0, (3, 2)))

    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Density", labelpad=2)
    ax.set_title(day_title, fontsize=13, pad=3, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.2, width=0.7, pad=1.5)


def create_hist_feature_panel(
    fig, outer_spec, panel_letter, title,
    day5_data, day14_data, xlabel, xlim,
    level_label, show_legend=False,
):
    panel_gs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_spec,
        height_ratios=[0.20, 0.80], hspace=0.04, wspace=0.40,
    )
    header = fig.add_subplot(panel_gs[0, :])
    hide_axes(header)
    header.text(0.00, 0.55, panel_letter, ha="left", va="center",
                fontsize=13, fontweight="bold")
    if title:
        header.text(0.10, 0.55, title, ha="left", va="center",
                    fontsize=14, fontweight="bold")

    if show_legend:
        handles = [
            Line2D([0], [0], color=COLOR_0, linewidth=1.5, label="0 µg/L"),
            Line2D([0], [0], color=COLOR_500, linewidth=1.5, label="500 µg/L"),
        ]
        header.legend(
            handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.72),
            ncol=2, frameon=False, fontsize=12, handlelength=1.4,
            columnspacing=0.7, borderaxespad=0,
        )

    ax5 = fig.add_subplot(panel_gs[1, 0])
    ax10 = fig.add_subplot(panel_gs[1, 1])
    draw_hist_comparison(ax5, day5_data[0], day5_data[1], xlim,
                         xlabel, "Day 5", level_label)
    draw_hist_comparison(ax10, day14_data[0], day14_data[1], xlim,
                         xlabel, "Day 14", level_label)


# =============================================================================
# 11. FULL FIGURE
# =============================================================================

def draw_full_figure(image_df, cell_df, representative_bundle):
    round_all = finite_array(image_df["mean_cell_roundness"])
    ecc_all = finite_array(image_df["green_mean_eccentricity"])
    tex_all = finite_array(image_df["texture_glcm_contrast_std"])

    round_xlim = robust_range(round_all, 0.5, 99.5, fallback=(0, 1.1))
    ecc_xlim = robust_range(ecc_all, 0.5, 99.5, fallback=(0, 1.0))
    tex_xlim = robust_range(tex_all, 0.5, 99.5, fallback=(0, 1.0))

    round_norm = colors.Normalize(round_xlim[0], round_xlim[1], clip=True)
    ecc_norm = colors.Normalize(max(0, ecc_xlim[0]), min(1, ecc_xlim[1]), clip=True)

    local_values = np.concatenate([
        finite_array(item["local_texture_map"])
        for item in representative_bundle.values()
    ])
    lt_lo, lt_hi = np.nanpercentile(local_values, [2, 99])
    if lt_hi <= lt_lo:
        lt_hi = lt_lo + 1e-6
    local_norm = colors.Normalize(lt_lo, lt_hi, clip=True)

    fig = plt.figure(figsize=(30.0, 20.0), facecolor="white")
    outer = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.0, 1.0, 1.30],
        height_ratios=[1.0, 1.0],
        left=0.065, right=0.99, top=0.965, bottom=0.07,
        wspace=0.23, hspace=0.22,
    )

    create_image_panel(
        fig, outer[0, 0], "a", "",
        representative_bundle, "raw",
    )
    create_image_panel(
        fig, outer[1, 0], "b", "",
        representative_bundle, "roundness", round_norm, ROUNDNESS_CMAP,
        colorbar_label="Roundness",
    )
    create_image_panel(
        fig, outer[0, 1], "c", "",
        representative_bundle, "eccentricity", ecc_norm, ECCENTRICITY_CMAP,
        colorbar_label="Eccentricity",
    )
    create_image_panel(
        fig, outer[1, 1], "d", "",
        representative_bundle, "texture", local_norm, TEXTURE_CMAP,
        colorbar_label="Local texture contrast",
    )

    right = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[:, 2],
        height_ratios=[1, 1, 1], hspace=0.24,
    )

    def cvals(day, dose, col):
        return cell_df.loc[
            (cell_df["day"] == day) & (cell_df["dose"] == dose), col
        ].to_numpy(dtype=float)

    def ivals(day, dose, col):
        return image_df.loc[
            (image_df["day"] == day) & (image_df["dose"] == dose), col
        ].to_numpy(dtype=float)

    create_hist_feature_panel(
        fig, right[0, 0], "e", "",
        day5_data=(ivals("Day 5", "0 ug/L", "mean_cell_roundness"),
                   ivals("Day 5", "500 ug/L", "mean_cell_roundness")),
        day14_data=(ivals("Day 14", "0 ug/L", "mean_cell_roundness"),
                    ivals("Day 14", "500 ug/L", "mean_cell_roundness")),
        xlabel="Roundness", xlim=round_xlim,
        level_label="Image-level", show_legend=True,
    )
    create_hist_feature_panel(
        fig, right[1, 0], "f", "",
        day5_data=(ivals("Day 5", "0 ug/L", "green_mean_eccentricity"),
                   ivals("Day 5", "500 ug/L", "green_mean_eccentricity")),
        day14_data=(ivals("Day 14", "0 ug/L", "green_mean_eccentricity"),
                    ivals("Day 14", "500 ug/L", "green_mean_eccentricity")),
        xlabel="Eccentricity", xlim=ecc_xlim,
        level_label="Image-level", show_legend=False,
    )
    create_hist_feature_panel(
        fig, right[2, 0], "g", "",
        day5_data=(ivals("Day 5", "0 ug/L", "texture_glcm_contrast_std"),
                   ivals("Day 5", "500 ug/L", "texture_glcm_contrast_std")),
        day14_data=(ivals("Day 14", "0 ug/L", "texture_glcm_contrast_std"),
                    ivals("Day 14", "500 ug/L", "texture_glcm_contrast_std")),
        xlabel="GLCM contrast SD", xlim=tex_xlim,
        level_label="Image-level", show_legend=False,
    )

    return fig


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    t0 = perf_counter()
    print("=" * 80)
    print("[Microscopy 3-feature figure]")
    print("[DATA_MODE]", DATA_MODE)
    print("=" * 80)

    if DATA_MODE == "mock":
        image_df, cell_df, storage = build_mock_tables()
        selection_df = select_representative_images(image_df)
        bundle = load_mock_representative_bundle(selection_df, storage)
        output_dir = Path("./results/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_base = output_dir / "Fig_microscopy_3feature_MOCK_validation"

    elif DATA_MODE == "real":
        image_df, cell_df = load_or_build_real_tables()
        for day, dose in GROUPS:
            n_img = int(((image_df["day"] == day) & (image_df["dose"] == dose)).sum())
            n_cell = int(((cell_df["day"] == day) & (cell_df["dose"] == dose)).sum())
            if n_img == 0:
                raise RuntimeError(f"没有有效数据：{day} | {dose}")
            print(f"[Group] {day} | {dose}: {n_img} images, {n_cell} cells")
        selection_df = select_representative_images(image_df)
        bundle = load_real_representative_bundle(selection_df)
        output_base = FIG_DIR / "Fig_microscopy_3feature_realdata"

    else:
        raise ValueError("DATA_MODE must be 'real' or 'mock'.")

    print("\n[Representative images]")
    print(selection_df[[
        "day", "dose", "image_name", "mean_cell_roundness",
        "green_mean_eccentricity", "texture_glcm_contrast_std",
    ]].to_string(index=False))

    fig = draw_full_figure(image_df, cell_df, bundle)

    if SAVE_PNG:
        fig.savefig(
            output_base.with_suffix(".png"), dpi=DPI,
            bbox_inches="tight", facecolor="white",
        )
    if SAVE_PDF:
        fig.savefig(
            output_base.with_suffix(".pdf"),
            bbox_inches="tight", facecolor="white",
        )
    if SAVE_TABLES:
        selection_df.to_csv(
            output_base.with_name(output_base.name + "_representative_selection.csv"),
            index=False, encoding="utf-8-sig",
        )
        image_df.to_csv(
            output_base.with_name(output_base.name + "_image_level_features.csv"),
            index=False, encoding="utf-8-sig",
        )
        cell_df.to_csv(
            output_base.with_name(output_base.name + "_cell_level_features.csv"),
            index=False, encoding="utf-8-sig",
        )
        cols = ["mean_cell_roundness", "green_mean_eccentricity", "texture_glcm_contrast_std"]
        group_summary = image_df.groupby(["day", "dose"])[cols].agg(["count", "mean", "std", "median"]); group_summary.columns = ["_".join(x) for x in group_summary.columns]; group_summary = group_summary.reset_index()
        cell_summary = cell_df.groupby(["day", "dose"])[["cell_roundness", "cell_eccentricity"]].agg(["count", "mean", "std", "median"]); cell_summary.columns = ["_".join(x) for x in cell_summary.columns]; cell_summary = cell_summary.reset_index()
        with pd.ExcelWriter(output_base.with_name(output_base.name + "_results.xlsx")) as xw:
            selection_df.to_excel(xw, sheet_name="representatives", index=False); group_summary.to_excel(xw, sheet_name="image_group_summary", index=False)
            cell_summary.to_excel(xw, sheet_name="cell_group_summary", index=False); image_df.to_excel(xw, sheet_name="image_features", index=False)

    plt.show()
    print("=" * 80)
    print(f"[DONE] seconds = {perf_counter() - t0:.1f}")
    print("[OUTPUT]", output_base)
    print("=" * 80)


if __name__ == "__main__":
    main()