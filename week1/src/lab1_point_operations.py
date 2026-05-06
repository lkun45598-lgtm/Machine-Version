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
DEFAULT_INPUT = DATA_DIR / "input_images" / "vision_lab_input.png"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/BGRA/color or grayscale image to uint8 grayscale."""
    if image is None:
        raise ValueError("image must not be None")
    if image.ndim == 2:
        return _as_uint8(image)
    if image.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {image.shape}")
    if image.shape[2] == 3:
        return cv2.cvtColor(_as_uint8(image), cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(_as_uint8(image), cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"expected 3 or 4 color channels, got {image.shape[2]}")


def gray_inversion(gray: np.ndarray) -> np.ndarray:
    """Apply gray-level inversion: s = 255 - r."""
    return cv2.subtract(255, to_grayscale(gray))


def threshold_image(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Apply binary thresholding with output values 0 and 255."""
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0, 255]")
    _, result = cv2.threshold(to_grayscale(gray), threshold, 255, cv2.THRESH_BINARY)
    return result


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to a grayscale image."""
    return cv2.equalizeHist(to_grayscale(gray))


def apply_pseudocolor(gray: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Map a grayscale image to pseudocolor using an OpenCV colormap."""
    return cv2.applyColorMap(to_grayscale(gray), colormap)


def ensure_default_input(path: Path = DEFAULT_INPUT) -> Path:
    """Create a distinct synthetic input image when no user image is provided."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 360, 540
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]

    blue = 55 + 125 * x + 30 * y
    green = 80 + 95 * (1 - x) + 45 * y
    red = 75 + 85 * np.sin(np.pi * x)[None, :] + 45 * (1 - y)
    image = np.dstack([blue + np.zeros_like(y), green + np.zeros_like(y), red])
    image = np.clip(image, 0, 255).astype(np.uint8)

    cv2.rectangle(image, (35, 45), (190, 165), (35, 170, 230), -1)
    cv2.circle(image, (340, 110), 58, (220, 65, 70), -1)
    cv2.ellipse(image, (400, 250), (85, 45), 22, 0, 360, (40, 205, 120), -1)
    cv2.line(image, (60, 300), (495, 210), (245, 245, 245), 6, cv2.LINE_AA)
    cv2.line(image, (70, 320), (470, 320), (35, 35, 35), 3, cv2.LINE_AA)

    for row in range(5):
        for col in range(8):
            color = (235, 235, 235) if (row + col) % 2 == 0 else (45, 45, 45)
            x0 = 225 + col * 22
            y0 = 215 + row * 22
            cv2.rectangle(image, (x0, y0), (x0 + 21, y0 + 21), color, -1)

    cv2.putText(
        image,
        "LAB 1",
        (54, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Point Ops",
        (258, 303),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    noise = np.random.default_rng(20260506).normal(0, 8, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    _write_image(path, image)
    return path


def run_experiment(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    threshold: int = 127,
) -> dict[str, Path]:
    """Run every operation required by the PPT and save result images."""
    if input_path is None:
        input_path = ensure_default_input()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"could not read image: {input_path}")

    gray = to_grayscale(original)
    inverted = gray_inversion(gray)
    thresholded = threshold_image(gray, threshold)
    equalized = equalize_histogram(gray)
    pseudocolor = apply_pseudocolor(gray)

    paths = {
        "original": output_dir / "01_original_color.png",
        "gray": output_dir / "02_grayscale.png",
        "inverted": output_dir / "03_gray_inversion.png",
        "threshold": output_dir / "04_threshold_binary.png",
        "equalized": output_dir / "05_histogram_equalized.png",
        "pseudocolor": output_dir / "06_pseudocolor.png",
        "histogram": output_dir / "07_histogram_comparison.png",
        "all_results": output_dir / "08_all_results.png",
    }

    _write_image(paths["original"], original)
    _write_image(paths["gray"], gray)
    _write_image(paths["inverted"], inverted)
    _write_image(paths["threshold"], thresholded)
    _write_image(paths["equalized"], equalized)
    _write_image(paths["pseudocolor"], pseudocolor)
    plot_histogram_comparison(gray, equalized, paths["histogram"])
    plot_all_results(original, gray, inverted, thresholded, equalized, pseudocolor, paths["all_results"])
    return paths


def plot_histogram_comparison(gray: np.ndarray, equalized: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].hist(gray.ravel(), bins=256, range=(0, 256), color="#4c78a8")
    axes[0].set_title("Original Histogram")
    axes[0].set_xlim(0, 255)
    axes[0].set_xlabel("Gray Level")
    axes[0].set_ylabel("Pixel Count")

    axes[1].hist(equalized.ravel(), bins=256, range=(0, 256), color="#f58518")
    axes[1].set_title("Equalized Histogram")
    axes[1].set_xlim(0, 255)
    axes[1].set_xlabel("Gray Level")
    axes[1].set_ylabel("Pixel Count")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_results(
    original: np.ndarray,
    gray: np.ndarray,
    inverted: np.ndarray,
    thresholded: np.ndarray,
    equalized: np.ndarray,
    pseudocolor: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    items = [
        ("Original Color", cv2.cvtColor(original, cv2.COLOR_BGR2RGB), None),
        ("Grayscale", gray, "gray"),
        ("Gray Inversion", inverted, "gray"),
        ("Binary Threshold", thresholded, "gray"),
        ("Histogram Equalization", equalized, "gray"),
        ("Pseudocolor", cv2.cvtColor(pseudocolor, cv2.COLOR_BGR2RGB), None),
    ]

    for ax, (title, image, cmap) in zip(axes.ravel(), items):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255 if cmap == "gray" else None)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 1: spatial-domain point operations.")
    parser.add_argument("--input", type=Path, default=None, help="Input image path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for results.")
    parser.add_argument("--threshold", type=int, default=127, help="Binary threshold in [0, 255].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else ensure_default_input()
    paths = run_experiment(input_path=input_path, output_dir=args.output_dir, threshold=args.threshold)
    print(f"Input image: {input_path}")
    print(f"Threshold: {args.threshold}")
    print("Generated files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
