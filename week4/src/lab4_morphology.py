from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
DEFAULT_BINARY_INPUT = DATA_DIR / "input_images" / "vision_lab4_binary.png"
DEFAULT_GRAY_INPUT = DATA_DIR / "input_images" / "vision_lab4_gray.png"
DEFAULT_HITMISS_INPUT = DATA_DIR / "input_images" / "vision_lab4_hitmiss.png"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"


def ensure_binary_input(path: Path = DEFAULT_BINARY_INPUT) -> Path:
    """Use the horse silhouette from skimage as the standard binary test image.

    skimage.data.horse() returns a bool array where True = background, False = horse.
    We invert it so the horse (foreground) is white (255) on a black (0) background.
    """
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    horse_bool = data.horse()
    binary = np.where(horse_bool, 0, 255).astype(np.uint8)
    _write_image(path, binary)
    return path


def ensure_gray_input(path: Path = DEFAULT_GRAY_INPUT) -> Path:
    """Use skimage.data.page() — a scanned text page with uneven illumination.

    The non-uniform background and dark text make it a textbook example for
    morphological gradient, top-hat (small bright structures), and black-hat
    (small dark structures — the text itself).
    """
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    page = data.page().astype(np.uint8)
    _write_image(path, page)
    return path


def ensure_hitmiss_input(path: Path = DEFAULT_HITMISS_INPUT) -> Path:
    """Synthetic binary image with several solid rectangles of different sizes.

    Used to demonstrate the hit-or-miss transform: detect upper-left corners
    of every white rectangle on the black background.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((240, 360), dtype=np.uint8)
    rects = [(20, 30, 60, 60), (40, 130, 100, 80), (150, 40, 50, 120), (160, 200, 70, 70), (40, 280, 40, 40)]
    for y, x, h, w in rects:
        canvas[y : y + h, x : x + w] = 255
    _write_image(path, canvas)
    return path


def add_binary_salt_pepper(binary: np.ndarray, prob: float = 0.04, seed: int = 20260520) -> np.ndarray:
    """Flip a fraction of pixels at random — adds salt and pepper to a binary image."""
    rng = np.random.default_rng(seed)
    noisy = binary.copy()
    mask = rng.random(binary.shape)
    noisy[mask < prob / 2] = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy


def get_kernel(shape: str = "rect", ksize: int = 5) -> np.ndarray:
    """Return a structuring element of the given OpenCV shape and odd size."""
    mapping = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    if shape not in mapping:
        raise ValueError(f"unknown kernel shape: {shape}")
    return cv2.getStructuringElement(mapping[shape], (ksize, ksize))


def erode(image: np.ndarray, ksize: int = 5, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    return cv2.erode(image, get_kernel(shape, ksize), iterations=iterations)


def dilate(image: np.ndarray, ksize: int = 5, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    return cv2.dilate(image, get_kernel(shape, ksize), iterations=iterations)


def opening(image: np.ndarray, ksize: int = 5, shape: str = "rect") -> np.ndarray:
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, get_kernel(shape, ksize))


def closing(image: np.ndarray, ksize: int = 5, shape: str = "rect") -> np.ndarray:
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, get_kernel(shape, ksize))


def morphological_gradient(image: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, get_kernel(shape, ksize))


def top_hat(image: np.ndarray, ksize: int = 15, shape: str = "rect") -> np.ndarray:
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, get_kernel(shape, ksize))


def black_hat(image: np.ndarray, ksize: int = 15, shape: str = "rect") -> np.ndarray:
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, get_kernel(shape, ksize))


def boundary_extraction(binary: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """Boundary = image - erosion(image).  Works on binary or grayscale images."""
    eroded = erode(binary, ksize=ksize, shape=shape)
    return cv2.subtract(binary, eroded)


def hit_or_miss(binary: np.ndarray, kernel_hit: np.ndarray, kernel_miss: np.ndarray) -> np.ndarray:
    """Classical hit-or-miss using two complementary structuring elements.

    Implemented manually as (erode(image, hit)) AND (erode(NOT image, miss))
    rather than cv2.morphologyEx(.., MORPH_HITMISS, ..), because the OpenCV
    builtin uses a signed-mask convention that is less explicit.
    """
    image = (binary > 0).astype(np.uint8) * 255
    inv = cv2.bitwise_not(image)
    hit = cv2.erode(image, kernel_hit)
    miss = cv2.erode(inv, kernel_miss)
    return cv2.bitwise_and(hit, miss)


def make_corner_hitmiss_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Hit + miss kernels that fire on the upper-left corner of a solid white region.

    Hit pattern requires the pixel itself and its right/below neighbours to be
    foreground; miss pattern requires the upper/left neighbours to be background.
    """
    hit = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)
    miss = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.uint8)
    return hit, miss


def overlay_points(binary: np.ndarray, mask: np.ndarray, radius: int = 4, colour: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """Visualise points in `mask` as red circles overlaid on `binary` (returns BGR)."""
    canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(mask > 0)
    for y, x in zip(ys, xs):
        cv2.circle(canvas, (int(x), int(y)), radius, colour, thickness=-1)
    return canvas


def run_experiment(
    binary_path: Path | None = None,
    gray_path: Path | None = None,
    hitmiss_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    if binary_path is None:
        binary_path = ensure_binary_input()
    if gray_path is None:
        gray_path = ensure_gray_input()
    if hitmiss_path is None:
        hitmiss_path = ensure_hitmiss_input()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(f"could not read image: {binary_path}")
    gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"could not read image: {gray_path}")
    hitmiss_src = cv2.imread(str(hitmiss_path), cv2.IMREAD_GRAYSCALE)
    if hitmiss_src is None:
        raise FileNotFoundError(f"could not read image: {hitmiss_path}")

    noisy_binary = add_binary_salt_pepper(binary, prob=0.04)

    ero3 = erode(binary, ksize=3)
    ero5 = erode(binary, ksize=5)
    ero9 = erode(binary, ksize=9)
    dil3 = dilate(binary, ksize=3)
    dil5 = dilate(binary, ksize=5)
    dil9 = dilate(binary, ksize=9)
    open5_clean = opening(binary, ksize=5)
    close5_clean = closing(binary, ksize=5)
    open5_noisy = opening(noisy_binary, ksize=5)
    close5_noisy = closing(noisy_binary, ksize=5)

    boundary_clean = boundary_extraction(binary, ksize=3)
    boundary_thick = boundary_extraction(binary, ksize=5)

    gradient_gray = morphological_gradient(gray, ksize=3)
    tophat_gray = top_hat(gray, ksize=25)
    blackhat_gray = black_hat(gray, ksize=25)

    hit_k, miss_k = make_corner_hitmiss_kernels()
    hitmiss_result = hit_or_miss(hitmiss_src, hit_k, miss_k)
    hitmiss_overlay = overlay_points(hitmiss_src, hitmiss_result, radius=4)

    paths = {
        "binary_original": output_dir / "01_binary_original.png",
        "binary_noisy": output_dir / "02_binary_noisy.png",
        "gray_original": output_dir / "03_gray_original.png",
        "hitmiss_source": output_dir / "04_hitmiss_source.png",
        "erode_3": output_dir / "10_erode_3.png",
        "erode_5": output_dir / "11_erode_5.png",
        "erode_9": output_dir / "12_erode_9.png",
        "dilate_3": output_dir / "13_dilate_3.png",
        "dilate_5": output_dir / "14_dilate_5.png",
        "dilate_9": output_dir / "15_dilate_9.png",
        "opening_clean": output_dir / "16_opening_clean.png",
        "closing_clean": output_dir / "17_closing_clean.png",
        "opening_noisy": output_dir / "18_opening_noisy.png",
        "closing_noisy": output_dir / "19_closing_noisy.png",
        "boundary_3": output_dir / "20_boundary_3.png",
        "boundary_5": output_dir / "21_boundary_5.png",
        "gradient": output_dir / "30_gradient.png",
        "tophat": output_dir / "31_tophat.png",
        "blackhat": output_dir / "32_blackhat.png",
        "hitmiss_result": output_dir / "40_hitmiss_result.png",
        "hitmiss_overlay": output_dir / "41_hitmiss_overlay.png",
        "basic_compare": output_dir / "50_basic_compare.png",
        "kernel_size_compare": output_dir / "51_kernel_size_compare.png",
        "open_close_compare": output_dir / "52_open_close_compare.png",
        "boundary_compare": output_dir / "53_boundary_compare.png",
        "other_compare": output_dir / "54_other_compare.png",
        "hitmiss_compare": output_dir / "55_hitmiss_compare.png",
        "all_results": output_dir / "60_all_results.png",
    }

    _write_image(paths["binary_original"], binary)
    _write_image(paths["binary_noisy"], noisy_binary)
    _write_image(paths["gray_original"], gray)
    _write_image(paths["hitmiss_source"], hitmiss_src)
    _write_image(paths["erode_3"], ero3)
    _write_image(paths["erode_5"], ero5)
    _write_image(paths["erode_9"], ero9)
    _write_image(paths["dilate_3"], dil3)
    _write_image(paths["dilate_5"], dil5)
    _write_image(paths["dilate_9"], dil9)
    _write_image(paths["opening_clean"], open5_clean)
    _write_image(paths["closing_clean"], close5_clean)
    _write_image(paths["opening_noisy"], open5_noisy)
    _write_image(paths["closing_noisy"], close5_noisy)
    _write_image(paths["boundary_3"], boundary_clean)
    _write_image(paths["boundary_5"], boundary_thick)
    _write_image(paths["gradient"], gradient_gray)
    _write_image(paths["tophat"], tophat_gray)
    _write_image(paths["blackhat"], blackhat_gray)
    _write_image(paths["hitmiss_result"], hitmiss_result)
    _write_image(paths["hitmiss_overlay"], hitmiss_overlay)

    plot_basic_compare(binary, ero5, dil5, open5_clean, close5_clean, paths["basic_compare"])
    plot_kernel_size_compare(binary, ero3, ero5, ero9, dil3, dil5, dil9, paths["kernel_size_compare"])
    plot_open_close_compare(noisy_binary, open5_noisy, close5_noisy, paths["open_close_compare"])
    plot_boundary_compare(binary, boundary_clean, boundary_thick, paths["boundary_compare"])
    plot_other_compare(gray, gradient_gray, tophat_gray, blackhat_gray, paths["other_compare"])
    plot_hitmiss_compare(hitmiss_src, hitmiss_result, hitmiss_overlay, paths["hitmiss_compare"])
    plot_all_results(
        binary, ero5, dil5, open5_clean, close5_clean,
        boundary_clean, gray, gradient_gray, tophat_gray, blackhat_gray,
        paths["all_results"],
    )

    metrics = compute_metrics(binary, noisy_binary, ero5, dil5, open5_noisy, close5_noisy)
    _write_metrics(output_dir / "metrics.txt", metrics)
    return paths


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_basic_compare(
    original: np.ndarray,
    eroded: np.ndarray,
    dilated: np.ndarray,
    opened: np.ndarray,
    closed: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), dpi=150)
    items = [
        ("Original (clean horse)", original),
        ("Erosion 5x5\nshrinks foreground", eroded),
        ("Dilation 5x5\nexpands foreground", dilated),
        ("Opening 5x5\n(dilate(erode(.)))", opened),
        ("Closing 5x5\n(erode(dilate(.)))", closed),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_kernel_size_compare(
    original: np.ndarray,
    ero3: np.ndarray, ero5: np.ndarray, ero9: np.ndarray,
    dil3: np.ndarray, dil5: np.ndarray, dil9: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), dpi=150)
    items = [
        ("Original", original),
        ("Erosion 3x3", ero3),
        ("Erosion 5x5", ero5),
        ("Erosion 9x9", ero9),
        ("Original", original),
        ("Dilation 3x3", dil3),
        ("Dilation 5x5", dil5),
        ("Dilation 9x9", dil9),
    ]
    for ax, (title, image) in zip(axes.ravel(), items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_open_close_compare(
    noisy_binary: np.ndarray,
    opened: np.ndarray,
    closed: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    items = [
        ("Noisy Binary (p=0.04)", noisy_binary),
        ("Opening 5x5\n(removes salt spots)", opened),
        ("Closing 5x5\n(fills pepper holes)", closed),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_boundary_compare(
    original: np.ndarray,
    boundary3: np.ndarray,
    boundary5: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    items = [
        ("Binary Original", original),
        ("Boundary (k=3)", boundary3),
        ("Boundary (k=5)", boundary5),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_other_compare(
    gray: np.ndarray,
    gradient: np.ndarray,
    tophat: np.ndarray,
    blackhat: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=150)
    items = [
        ("Gray Original (page)", gray),
        ("Gradient (k=3)\nboundary", gradient),
        ("Top-hat (k=25)\nbright details", tophat),
        ("Black-hat (k=25)\ndark text", blackhat),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_hitmiss_compare(
    source: np.ndarray,
    result: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(source, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Source (synthetic rects)")
    axes[1].imshow(result, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Hit-or-Miss\n(upper-left corner hits)")
    axes[2].imshow(_bgr_to_rgb(overlay))
    axes[2].set_title("Detected corners overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_results(
    binary: np.ndarray,
    eroded: np.ndarray, dilated: np.ndarray,
    opened: np.ndarray, closed: np.ndarray,
    boundary: np.ndarray,
    gray: np.ndarray, gradient: np.ndarray,
    tophat: np.ndarray, blackhat: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=150)
    items = [
        ("Binary Original", binary),
        ("Erosion 5x5", eroded),
        ("Dilation 5x5", dilated),
        ("Opening 5x5", opened),
        ("Closing 5x5", closed),
        ("Boundary (k=3)", boundary),
        ("(blank)", None),
        ("(blank)", None),
        ("Gray Original (page)", gray),
        ("Morph Gradient", gradient),
        ("Top-hat", tophat),
        ("Black-hat", blackhat),
    ]
    for ax, (title, image) in zip(axes.ravel(), items):
        if image is None:
            ax.axis("off")
            ax.set_title("")
            continue
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_metrics(
    original: np.ndarray,
    noisy: np.ndarray,
    eroded: np.ndarray,
    dilated: np.ndarray,
    opened: np.ndarray,
    closed: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Foreground-pixel counts and how each operation perturbs them."""
    def fg(image: np.ndarray) -> int:
        return int((image > 127).sum())

    total = original.size
    return {
        "original": {"foreground_px": fg(original), "ratio": fg(original) / total},
        "noisy": {"foreground_px": fg(noisy), "ratio": fg(noisy) / total},
        "erosion_5": {"foreground_px": fg(eroded), "ratio": fg(eroded) / total},
        "dilation_5": {"foreground_px": fg(dilated), "ratio": fg(dilated) / total},
        "opening_5": {"foreground_px": fg(opened), "ratio": fg(opened) / total},
        "closing_5": {"foreground_px": fg(closed), "ratio": fg(closed) / total},
    }


def _write_metrics(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Image\tForegroundPx\tForegroundRatio\n")
        for name, vals in metrics.items():
            fh.write(f"{name}\t{vals['foreground_px']}\t{vals['ratio']:.6f}\n")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 4: morphological image processing.")
    parser.add_argument("--binary", type=Path, default=None, help="Binary input image (foreground white).")
    parser.add_argument("--gray", type=Path, default=None, help="Grayscale input for gradient/tophat/blackhat.")
    parser.add_argument("--hitmiss", type=Path, default=None, help="Binary input for hit-or-miss demo.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_experiment(
        binary_path=args.binary,
        gray_path=args.gray,
        hitmiss_path=args.hitmiss,
        output_dir=args.output_dir,
    )
    print("Generated files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
