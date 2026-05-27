from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import feature as sk_feature


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
DEFAULT_COLOR_INPUT = DATA_DIR / "input_images" / "vision_lab5_colorwheel.png"
DEFAULT_TEXTURE_INPUT = DATA_DIR / "input_images" / "vision_lab5_ihc.png"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"


def ensure_color_input(path: Path = DEFAULT_COLOR_INPUT) -> Path:
    """Use skimage.data.colorwheel() — a 370x371 synthetic color chart with
    every HSV hue spread radially. Perfect for demonstrating channel splits
    and HSV-range based color extraction.
    """
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = data.colorwheel()
    _write_image(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def ensure_texture_input(path: Path = DEFAULT_TEXTURE_INPUT) -> Path:
    """Use skimage.data.immunohistochemistry() — a 512x512 medical microscopy
    image of stained tissue. The brown DAB stain on purple haematoxylin counter-
    stain produces dense local texture (good for LBP) and many small cell-nuclei
    blobs with strong luminance variation (good for SIFT keypoint detection).
    """
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = data.immunohistochemistry()
    _write_image(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def split_color_spaces(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Return individual channels of the BGR image in RGB, HSV and Lab spaces."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    r, g, b = cv2.split(rgb)
    h, s, v = cv2.split(hsv)
    L, a, bl = cv2.split(lab)
    return {
        "R": r, "G": g, "B": b,
        "H": h, "S": s, "V": v,
        "L": L, "a": a, "b": bl,
    }


def extract_yellow(
    bgr: np.ndarray,
    h_min: int = 20,
    h_max: int = 35,
    s_min: int = 100,
    v_min: int = 100,
    morph_ksize: int = 5,
    min_area: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Extract yellow regions using HSV thresholding + close/open + contour analysis.

    Returns (raw_mask, cleaned_mask, annotated_bgr, per-contour stats).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    raw_mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = bgr.copy()
    stats: list[dict[str, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=2)
        cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 0), thickness=-1)
        cv2.putText(
            overlay,
            f"A={int(area)}",
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        stats.append({
            "area": float(area),
            "cx": float(cx),
            "cy": float(cy),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
        })
    return raw_mask, mask, overlay, stats


def compute_lbp(
    bgr: np.ndarray,
    P: int = 8,
    R: int = 1,
    method: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute LBP map and normalised histogram. Returns (gray, lbp, hist)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = sk_feature.local_binary_pattern(gray, P=P, R=R, method=method)
    n_bins = P + 2 if method == "uniform" else int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=False)
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return gray, lbp, hist


def rotate_image(image: np.ndarray, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """Rotate `image` around its centre by `angle_deg` (counter-clockwise)."""
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle_deg, scale)
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def sift_match(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    ratio: float = 0.7,
    nfeatures: int = 0,
) -> dict[str, object]:
    """Run SIFT on two BGR images, KNN-match with FLANN, apply Lowe ratio test."""
    gray_a = cv2.cvtColor(image_a_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        raise RuntimeError("SIFT failed to extract enough descriptors")

    flann = cv2.FlannBasedMatcher(
        indexParams=dict(algorithm=1, trees=5),
        searchParams=dict(checks=50),
    )
    knn = flann.knnMatch(des_a, des_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    good.sort(key=lambda m: m.distance)

    match_image = cv2.drawMatches(
        image_a_bgr, kp_a, image_b_bgr, kp_b, good[:60], None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )
    kp_image_a = cv2.drawKeypoints(image_a_bgr, kp_a, None, color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_image_b = cv2.drawKeypoints(image_b_bgr, kp_b, None, color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return {
        "kp_a": kp_a,
        "kp_b": kp_b,
        "des_a": des_a,
        "des_b": des_b,
        "good_matches": good,
        "match_image": match_image,
        "kp_image_a": kp_image_a,
        "kp_image_b": kp_image_b,
    }


def align_by_homography(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    kp_a: list[cv2.KeyPoint],
    kp_b: list[cv2.KeyPoint],
    good: list[cv2.DMatch],
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Estimate homography from `image_b` (query) back to `image_a` (template)
    using matched keypoints, then warp the query into the template's frame.
    """
    if len(good) < 4:
        return None, None, 0
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, 0
    h, w = image_a_bgr.shape[:2]
    warped = cv2.warpPerspective(image_b_bgr, H, (w, h))
    inliers = int(mask.sum()) if mask is not None else 0
    return warped, H, inliers


# ---------- plotting helpers ----------

def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_color_channels(bgr: np.ndarray, channels: dict[str, np.ndarray], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(11, 13.5), dpi=150)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original (RGB)")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    cmaps = {
        "R": "Reds", "G": "Greens", "B": "Blues",
        "H": "hsv", "S": "gray", "V": "gray",
        "L": "gray", "a": "gray", "b": "gray",
    }
    layout = [["R", "G", "B"], ["H", "S", "V"], ["L", "a", "b"]]
    row_labels = ["RGB color space", "HSV color space", "Lab color space"]
    for row_idx, (row, label) in enumerate(zip(layout, row_labels), start=1):
        for col_idx, name in enumerate(row):
            ax = axes[row_idx, col_idx]
            data = channels[name]
            ax.imshow(data, cmap=cmaps[name])
            ax.set_title(f"{name} channel")
            ax.axis("off")
        # Add row label on the left
        axes[row_idx, 0].set_ylabel(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_yellow_extraction(
    bgr: np.ndarray,
    raw_mask: np.ndarray,
    clean_mask: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=150)
    axes[0].imshow(_bgr_to_rgb(bgr))
    axes[0].set_title("Original")
    axes[1].imshow(raw_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Raw HSV mask\n(H=[20,35], S>=100, V>=100)")
    axes[2].imshow(clean_mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("After close 5x5 + open 5x5")
    axes[3].imshow(_bgr_to_rgb(overlay))
    axes[3].set_title("Contour + centroid on original")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_lbp(
    bgr: np.ndarray,
    gray: np.ndarray,
    lbp: np.ndarray,
    hist: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=150)
    axes[0].imshow(_bgr_to_rgb(bgr))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale")
    axes[1].axis("off")
    axes[2].imshow(lbp, cmap="gray")
    axes[2].set_title("LBP map\n(P=8, R=1, uniform)")
    axes[2].axis("off")
    axes[3].bar(np.arange(len(hist)), hist, color="#3a6ea5", edgecolor="black")
    axes[3].set_title("LBP histogram\n(normalised)")
    axes[3].set_xlabel("LBP code")
    axes[3].set_ylabel("Probability")
    axes[3].set_xticks(np.arange(len(hist)))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_sift_keypoints(
    kp_image_a: np.ndarray,
    kp_image_b: np.ndarray,
    n_a: int,
    n_b: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=150)
    axes[0].imshow(_bgr_to_rgb(kp_image_a))
    axes[0].set_title(f"Template — {n_a} keypoints")
    axes[1].imshow(_bgr_to_rgb(kp_image_b))
    axes[1].set_title(f"Rotated query — {n_b} keypoints")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_sift_match(match_image: np.ndarray, n_good: int, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13, 6), dpi=150)
    ax.imshow(_bgr_to_rgb(match_image))
    ax.set_title(f"SIFT matches after Lowe ratio test (showing top 60 of {n_good})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_alignment(
    template_bgr: np.ndarray,
    query_bgr: np.ndarray,
    warped_bgr: np.ndarray | None,
    inliers: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), dpi=150)
    axes[0].imshow(_bgr_to_rgb(template_bgr))
    axes[0].set_title("Template (a)")
    axes[1].imshow(_bgr_to_rgb(query_bgr))
    axes[1].set_title("Rotated query (b)")
    if warped_bgr is not None:
        axes[2].imshow(_bgr_to_rgb(warped_bgr))
        axes[2].set_title(f"Query warped back\n(RANSAC inliers: {inliers})")
        blend = cv2.addWeighted(template_bgr, 0.5, warped_bgr, 0.5, 0)
        axes[3].imshow(_bgr_to_rgb(blend))
        axes[3].set_title("Overlay (template + warped)")
    else:
        for k in (2, 3):
            axes[k].text(0.5, 0.5, "Homography failed", ha="center", va="center")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_results(
    color_bgr: np.ndarray,
    yellow_overlay: np.ndarray,
    texture_bgr: np.ndarray,
    lbp: np.ndarray,
    match_image: np.ndarray,
    align_blend: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    axes[0, 0].imshow(_bgr_to_rgb(color_bgr))
    axes[0, 0].set_title("Color image: colorwheel")
    axes[0, 1].imshow(_bgr_to_rgb(yellow_overlay))
    axes[0, 1].set_title("Yellow extraction\n(contour + centroid)")
    axes[0, 2].imshow(_bgr_to_rgb(texture_bgr))
    axes[0, 2].set_title("Texture image: IHC slide")
    axes[1, 0].imshow(lbp, cmap="gray")
    axes[1, 0].set_title("LBP texture map (IHC)")
    axes[1, 1].imshow(_bgr_to_rgb(match_image))
    axes[1, 1].set_title("SIFT matches (IHC vs rotated)")
    if align_blend is not None:
        axes[1, 2].imshow(_bgr_to_rgb(align_blend))
        axes[1, 2].set_title("Alignment overlay")
    else:
        axes[1, 2].text(0.5, 0.5, "Homography failed", ha="center", va="center")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ---------- IO helpers ----------

def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def _write_metrics(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- pipeline ----------

def run_experiment(
    color_input: Path | None = None,
    texture_input: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rotate_angle: float = 45.0,
    h_min: int = 20,
    h_max: int = 35,
) -> dict[str, Path]:
    if color_input is None:
        color_input = ensure_color_input()
    if texture_input is None:
        texture_input = ensure_texture_input()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_bgr = cv2.imread(str(color_input), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"could not read color image: {color_input}")
    texture_bgr = cv2.imread(str(texture_input), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise FileNotFoundError(f"could not read texture image: {texture_input}")

    # --- Part 1: color spaces on colorwheel ---
    channels = split_color_spaces(color_bgr)

    # --- Part 2: yellow extraction on colorwheel ---
    raw_mask, yellow_mask, yellow_overlay, yellow_stats = extract_yellow(
        color_bgr, h_min=h_min, h_max=h_max
    )

    # --- Part 3: LBP on IHC slide ---
    gray, lbp, hist = compute_lbp(texture_bgr)
    lbp_disp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Part 4: SIFT on IHC slide ---
    rotated = rotate_image(texture_bgr, angle_deg=rotate_angle)
    sift_out = sift_match(texture_bgr, rotated, ratio=0.7)
    warped, H, inliers = align_by_homography(
        texture_bgr, rotated, sift_out["kp_a"], sift_out["kp_b"], sift_out["good_matches"]
    )
    align_blend: np.ndarray | None = None
    if warped is not None:
        align_blend = cv2.addWeighted(texture_bgr, 0.5, warped, 0.5, 0)

    paths = {
        "color_original": output_dir / "01_color_original.png",
        "texture_original": output_dir / "02_texture_original.png",
        "texture_gray": output_dir / "03_texture_gray.png",
        "texture_rotated": output_dir / "04_texture_rotated.png",
        "channel_R": output_dir / "10_channel_R.png",
        "channel_G": output_dir / "11_channel_G.png",
        "channel_B": output_dir / "12_channel_B.png",
        "channel_H": output_dir / "13_channel_H.png",
        "channel_S": output_dir / "14_channel_S.png",
        "channel_V": output_dir / "15_channel_V.png",
        "channel_L": output_dir / "16_channel_L.png",
        "channel_a": output_dir / "17_channel_a.png",
        "channel_b": output_dir / "18_channel_b.png",
        "color_channels": output_dir / "20_color_channels.png",
        "yellow_raw_mask": output_dir / "30_yellow_raw_mask.png",
        "yellow_clean_mask": output_dir / "31_yellow_clean_mask.png",
        "yellow_overlay": output_dir / "32_yellow_overlay.png",
        "yellow_compare": output_dir / "33_yellow_compare.png",
        "lbp_map": output_dir / "40_lbp_map.png",
        "lbp_compare": output_dir / "41_lbp_compare.png",
        "sift_kp_a": output_dir / "50_sift_keypoints_template.png",
        "sift_kp_b": output_dir / "51_sift_keypoints_query.png",
        "sift_kp_compare": output_dir / "52_sift_keypoints.png",
        "sift_match": output_dir / "53_sift_match.png",
        "sift_align": output_dir / "54_sift_alignment.png",
        "all_results": output_dir / "60_all_results.png",
    }

    _write_image(paths["color_original"], color_bgr)
    _write_image(paths["texture_original"], texture_bgr)
    _write_image(paths["texture_gray"], gray)
    _write_image(paths["texture_rotated"], rotated)
    _write_image(paths["channel_R"], channels["R"])
    _write_image(paths["channel_G"], channels["G"])
    _write_image(paths["channel_B"], channels["B"])
    _write_image(paths["channel_H"], channels["H"])
    _write_image(paths["channel_S"], channels["S"])
    _write_image(paths["channel_V"], channels["V"])
    _write_image(paths["channel_L"], channels["L"])
    _write_image(paths["channel_a"], channels["a"])
    _write_image(paths["channel_b"], channels["b"])
    _write_image(paths["yellow_raw_mask"], raw_mask)
    _write_image(paths["yellow_clean_mask"], yellow_mask)
    _write_image(paths["yellow_overlay"], yellow_overlay)
    _write_image(paths["lbp_map"], lbp_disp)
    _write_image(paths["sift_kp_a"], sift_out["kp_image_a"])
    _write_image(paths["sift_kp_b"], sift_out["kp_image_b"])
    _write_image(paths["sift_match"], sift_out["match_image"])

    plot_color_channels(color_bgr, channels, paths["color_channels"])
    plot_yellow_extraction(color_bgr, raw_mask, yellow_mask, yellow_overlay, paths["yellow_compare"])
    plot_lbp(texture_bgr, gray, lbp, hist, paths["lbp_compare"])
    plot_sift_keypoints(
        sift_out["kp_image_a"], sift_out["kp_image_b"],
        len(sift_out["kp_a"]), len(sift_out["kp_b"]),
        paths["sift_kp_compare"],
    )
    plot_sift_match(sift_out["match_image"], len(sift_out["good_matches"]), paths["sift_match"])
    plot_alignment(texture_bgr, rotated, warped, inliers, paths["sift_align"])
    plot_all_results(
        color_bgr, yellow_overlay, texture_bgr, lbp,
        sift_out["match_image"], align_blend, paths["all_results"],
    )

    metrics: list[str] = []
    metrics.append(f"color_input\t{color_input}")
    metrics.append(f"texture_input\t{texture_input}")
    metrics.append(f"color_shape\t{color_bgr.shape}")
    metrics.append(f"texture_shape\t{texture_bgr.shape}")
    metrics.append(f"rotate_angle_deg\t{rotate_angle}")
    metrics.append(f"yellow_hsv_range\tH=[{h_min},{h_max}] S>=100 V>=100")
    metrics.append(f"yellow_pixels_raw\t{int((raw_mask>0).sum())}")
    metrics.append(f"yellow_pixels_clean\t{int((yellow_mask>0).sum())}")
    metrics.append(f"yellow_contour_count\t{len(yellow_stats)}")
    for i, s in enumerate(yellow_stats):
        metrics.append(
            f"yellow_contour_{i}\tarea={s['area']:.0f}\tcentroid=({s['cx']:.1f},{s['cy']:.1f})\tbbox=({s['x']},{s['y']},{s['w']},{s['h']})"
        )
    metrics.append(f"lbp_params\tP=8,R=1,method=uniform,bins={len(hist)}")
    metrics.append("lbp_hist\t" + ",".join(f"{p:.4f}" for p in hist))
    metrics.append(f"sift_keypoints_template\t{len(sift_out['kp_a'])}")
    metrics.append(f"sift_keypoints_query\t{len(sift_out['kp_b'])}")
    metrics.append(f"sift_good_matches\t{len(sift_out['good_matches'])}")
    metrics.append(f"homography_inliers\t{inliers}")
    if H is not None:
        metrics.append("homography_matrix\t" + ";".join(",".join(f"{v:.4f}" for v in row) for row in H))
    _write_metrics(output_dir / "metrics.txt", metrics)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 5: image feature extraction.")
    parser.add_argument("--color", type=Path, default=None, help="Color image for HSV/Lab/yellow demo (BGR).")
    parser.add_argument("--texture", type=Path, default=None, help="Texture image for LBP and SIFT (BGR).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--rotate-angle", type=float, default=45.0, help="Rotation angle for SIFT query in degrees.")
    parser.add_argument("--h-min", type=int, default=20, help="Lower bound of HSV hue for yellow extraction.")
    parser.add_argument("--h-max", type=int, default=35, help="Upper bound of HSV hue for yellow extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_experiment(
        color_input=args.color,
        texture_input=args.texture,
        output_dir=args.output_dir,
        rotate_angle=args.rotate_angle,
        h_min=args.h_min,
        h_max=args.h_max,
    )
    print("Generated files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
