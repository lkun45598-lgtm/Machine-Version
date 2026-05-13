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
DEFAULT_INPUT = DATA_DIR / "input_images" / "vision_lab2_input.png"
DEFAULT_INPUT_AUX = DATA_DIR / "input_images" / "vision_lab2_aux.png"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"


def ensure_default_input(path: Path = DEFAULT_INPUT) -> Path:
    """Use skimage.data.chelsea() (cat photo) as the standard color test image."""
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = data.chelsea()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _write_image(path, bgr)
    return path


def ensure_aux_input(path: Path = DEFAULT_INPUT_AUX) -> Path:
    """Provide a second image (coffee) for algebraic-operation experiments."""
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = data.coffee()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _write_image(path, bgr)
    return path


def add_salt_and_pepper_noise(image: np.ndarray, prob: float = 0.05, seed: int = 20260513) -> np.ndarray:
    """Add salt-and-pepper noise to an image with given probability."""
    if not 0 <= prob <= 1:
        raise ValueError("prob must be in [0, 1]")
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    mask = rng.random(image.shape[:2])
    noisy[mask < prob / 2] = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy


def add_gaussian_noise(image: np.ndarray, sigma: float = 15.0, seed: int = 20260513) -> np.ndarray:
    """Add zero-mean Gaussian noise to an image."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def make_noisy_image(image: np.ndarray) -> np.ndarray:
    """Build a clearly noisy image combining salt-pepper and Gaussian noise."""
    return add_gaussian_noise(add_salt_and_pepper_noise(image, prob=0.04), sigma=12.0)


def make_blurred_image(image: np.ndarray, ksize: int = 9, sigma: float = 2.5) -> np.ndarray:
    """Build a noticeably blurred image for sharpening experiments."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def mean_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.blur(image, (ksize, ksize))


def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def gaussian_filter(image: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def prewitt_operator(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Prewitt operator. Returns (grad_x, grad_y, magnitude) as uint8."""
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy), cv2.convertScaleAbs(mag)


def sobel_operator(gray: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy), cv2.convertScaleAbs(mag)


def laplacian_operator(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return cv2.convertScaleAbs(lap)


def laplacian_sharpen(image: np.ndarray, ksize: int = 3, weight: float = 1.0) -> np.ndarray:
    """Sharpen with Laplacian feedback: g = f - w * Laplacian(f)."""
    if image.ndim == 3:
        channels = [laplacian_sharpen(image[:, :, c], ksize=ksize, weight=weight) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=ksize)
    sharp = image.astype(np.float32) - weight * lap
    return np.clip(sharp, 0, 255).astype(np.uint8)


def canny_operator(gray: np.ndarray, low: int = 60, high: int = 150) -> np.ndarray:
    return cv2.Canny(gray, low, high)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def algebraic_operations(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Resize b to a's shape and compute add/sub/mul/div with saturation."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    add = cv2.addWeighted(a, 0.5, b, 0.5, 0)
    sub = cv2.convertScaleAbs(af - bf)
    mul = np.clip(af * bf / 255.0, 0, 255).astype(np.uint8)
    div = np.clip((af + 1.0) / (bf + 1.0) * 64.0, 0, 255).astype(np.uint8)
    return {"add": add, "sub": sub, "mul": mul, "div": div}


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    err = mse(a, b)
    if err <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / err))


def run_experiment(
    input_path: Path | None = None,
    aux_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Run every operation required by the lab and save result images."""
    if input_path is None:
        input_path = ensure_default_input()
    if aux_path is None:
        aux_path = ensure_aux_input()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"could not read image: {input_path}")
    aux = cv2.imread(str(aux_path), cv2.IMREAD_COLOR)
    if aux is None:
        raise FileNotFoundError(f"could not read image: {aux_path}")

    noisy = make_noisy_image(original)
    blurred = make_blurred_image(original)

    mean3 = mean_filter(noisy, ksize=3)
    mean5 = mean_filter(noisy, ksize=5)
    mean9 = mean_filter(noisy, ksize=9)
    med3 = median_filter(noisy, ksize=3)
    med5 = median_filter(noisy, ksize=5)
    med9 = median_filter(noisy, ksize=9)
    gauss3 = gaussian_filter(noisy, ksize=3, sigma=1.0)
    gauss5 = gaussian_filter(noisy, ksize=5, sigma=1.5)
    gauss9 = gaussian_filter(noisy, ksize=9, sigma=2.0)

    blurred_gray = to_grayscale(blurred)
    px, py, pm = prewitt_operator(blurred_gray)
    sx, sy, sm = sobel_operator(blurred_gray, ksize=3)
    lap = laplacian_operator(blurred_gray, ksize=3)
    lap_sharp = laplacian_sharpen(blurred, ksize=3, weight=1.0)
    canny = canny_operator(blurred_gray, low=60, high=150)

    alg = algebraic_operations(original, aux)

    paths = {
        "original": output_dir / "01_original.png",
        "noisy": output_dir / "02_noisy.png",
        "blurred": output_dir / "03_blurred.png",
        "mean_3": output_dir / "10_mean_3.png",
        "mean_5": output_dir / "11_mean_5.png",
        "mean_9": output_dir / "12_mean_9.png",
        "median_3": output_dir / "13_median_3.png",
        "median_5": output_dir / "14_median_5.png",
        "median_9": output_dir / "15_median_9.png",
        "gauss_3": output_dir / "16_gauss_3.png",
        "gauss_5": output_dir / "17_gauss_5.png",
        "gauss_9": output_dir / "18_gauss_9.png",
        "prewitt_x": output_dir / "20_prewitt_x.png",
        "prewitt_y": output_dir / "21_prewitt_y.png",
        "prewitt_mag": output_dir / "22_prewitt_mag.png",
        "sobel_x": output_dir / "23_sobel_x.png",
        "sobel_y": output_dir / "24_sobel_y.png",
        "sobel_mag": output_dir / "25_sobel_mag.png",
        "laplacian": output_dir / "26_laplacian.png",
        "laplacian_sharpen": output_dir / "27_laplacian_sharpen.png",
        "canny": output_dir / "28_canny.png",
        "alg_add": output_dir / "30_alg_add.png",
        "alg_sub": output_dir / "31_alg_sub.png",
        "alg_mul": output_dir / "32_alg_mul.png",
        "alg_div": output_dir / "33_alg_div.png",
        "smoothing_compare": output_dir / "40_smoothing_compare.png",
        "median_vs_mean_sp": output_dir / "41_median_vs_mean_saltpepper.png",
        "gaussian_size_compare": output_dir / "42_gaussian_size_compare.png",
        "sharpening_compare": output_dir / "50_sharpening_compare.png",
        "edge_directions": output_dir / "51_edge_directions.png",
        "algebraic_compare": output_dir / "60_algebraic_compare.png",
        "all_results": output_dir / "70_all_results.png",
    }

    _write_image(paths["original"], original)
    _write_image(paths["noisy"], noisy)
    _write_image(paths["blurred"], blurred)
    _write_image(paths["mean_3"], mean3)
    _write_image(paths["mean_5"], mean5)
    _write_image(paths["mean_9"], mean9)
    _write_image(paths["median_3"], med3)
    _write_image(paths["median_5"], med5)
    _write_image(paths["median_9"], med9)
    _write_image(paths["gauss_3"], gauss3)
    _write_image(paths["gauss_5"], gauss5)
    _write_image(paths["gauss_9"], gauss9)
    _write_image(paths["prewitt_x"], px)
    _write_image(paths["prewitt_y"], py)
    _write_image(paths["prewitt_mag"], pm)
    _write_image(paths["sobel_x"], sx)
    _write_image(paths["sobel_y"], sy)
    _write_image(paths["sobel_mag"], sm)
    _write_image(paths["laplacian"], lap)
    _write_image(paths["laplacian_sharpen"], lap_sharp)
    _write_image(paths["canny"], canny)
    _write_image(paths["alg_add"], alg["add"])
    _write_image(paths["alg_sub"], alg["sub"])
    _write_image(paths["alg_mul"], alg["mul"])
    _write_image(paths["alg_div"], alg["div"])

    plot_smoothing_comparison(noisy, mean5, med5, gauss5, paths["smoothing_compare"])
    plot_median_vs_mean_saltpepper(original, mean5, med5, paths["median_vs_mean_sp"])
    plot_kernel_size_compare(noisy, mean3, mean5, mean9, paths["gaussian_size_compare"])
    plot_sharpening_comparison(blurred, pm, sm, lap, canny, paths["sharpening_compare"])
    plot_edge_directions(blurred_gray, sx, sy, sm, paths["edge_directions"])
    plot_algebraic_compare(original, aux, alg, paths["algebraic_compare"])
    plot_all_results(
        original, noisy, blurred, mean5, med5, gauss5, pm, sm, lap, canny, paths["all_results"]
    )

    metrics = compute_metrics(original, noisy, mean5, med5, gauss5)
    _write_metrics(output_dir / "metrics.txt", metrics)
    return paths


def compute_metrics(
    original: np.ndarray,
    noisy: np.ndarray,
    mean5: np.ndarray,
    median5: np.ndarray,
    gauss5: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "noisy": {"mse": mse(original, noisy), "psnr": psnr(original, noisy)},
        "mean_5": {"mse": mse(original, mean5), "psnr": psnr(original, mean5)},
        "median_5": {"mse": mse(original, median5), "psnr": psnr(original, median5)},
        "gauss_5": {"mse": mse(original, gauss5), "psnr": psnr(original, gauss5)},
    }


def _write_metrics(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Image\tMSE\tPSNR(dB)\n")
        for name, vals in metrics.items():
            fh.write(f"{name}\t{vals['mse']:.3f}\t{vals['psnr']:.3f}\n")


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_smoothing_comparison(
    noisy: np.ndarray,
    mean5: np.ndarray,
    median5: np.ndarray,
    gauss5: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=150)
    items = [
        ("Noisy (Salt&Pepper + Gaussian)", noisy),
        ("Mean Filter 5x5", mean5),
        ("Median Filter 5x5", median5),
        ("Gaussian Filter 5x5", gauss5),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(_bgr_to_rgb(image))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_median_vs_mean_saltpepper(
    original: np.ndarray, mean5: np.ndarray, median5: np.ndarray, output_path: Path
) -> None:
    sp = add_salt_and_pepper_noise(original, prob=0.08)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    items = [
        ("Salt&Pepper Noise (p=0.08)", sp),
        ("Mean 5x5 on S&P", mean_filter(sp, 5)),
        ("Median 5x5 on S&P", median_filter(sp, 5)),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(_bgr_to_rgb(image))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_kernel_size_compare(
    noisy: np.ndarray,
    mean3: np.ndarray,
    mean5: np.ndarray,
    mean9: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=150)
    items = [
        ("Noisy", noisy),
        ("Mean 3x3", mean3),
        ("Mean 5x5", mean5),
        ("Mean 9x9", mean9),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(_bgr_to_rgb(image))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_sharpening_comparison(
    blurred: np.ndarray,
    prewitt_mag: np.ndarray,
    sobel_mag: np.ndarray,
    lap: np.ndarray,
    canny: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), dpi=150)
    items = [
        ("Blurred Input", _bgr_to_rgb(blurred), None),
        ("Prewitt |G|", prewitt_mag, "gray"),
        ("Sobel |G|", sobel_mag, "gray"),
        ("Laplacian |L|", lap, "gray"),
        ("Canny", canny, "gray"),
    ]
    for ax, (title, image, cmap) in zip(axes, items):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=255 if cmap == "gray" else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_edge_directions(
    gray: np.ndarray, sx: np.ndarray, sy: np.ndarray, mag: np.ndarray, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=150)
    items = [
        ("Gray Input", gray),
        ("Sobel Gx (vertical edges)", sx),
        ("Sobel Gy (horizontal edges)", sy),
        ("Sobel |G|", mag),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_algebraic_compare(
    a: np.ndarray, b: np.ndarray, alg: dict[str, np.ndarray], output_path: Path
) -> None:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    items = [
        ("Image A (chelsea)", _bgr_to_rgb(a)),
        ("Image B (coffee, resized)", _bgr_to_rgb(b)),
        ("0.5*A + 0.5*B (add)", _bgr_to_rgb(alg["add"])),
        ("|A - B| (sub)", _bgr_to_rgb(alg["sub"])),
        ("A * B / 255 (mul)", _bgr_to_rgb(alg["mul"])),
        ("(A+1)/(B+1) * 64 (div)", _bgr_to_rgb(alg["div"])),
    ]
    for ax, (title, image) in zip(axes.ravel(), items):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_results(
    original: np.ndarray,
    noisy: np.ndarray,
    blurred: np.ndarray,
    mean5: np.ndarray,
    median5: np.ndarray,
    gauss5: np.ndarray,
    prewitt_mag: np.ndarray,
    sobel_mag: np.ndarray,
    lap: np.ndarray,
    canny: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=150)
    items = [
        ("Original", _bgr_to_rgb(original), None),
        ("Noisy Input", _bgr_to_rgb(noisy), None),
        ("Mean 5x5", _bgr_to_rgb(mean5), None),
        ("Median 5x5", _bgr_to_rgb(median5), None),
        ("Gaussian 5x5", _bgr_to_rgb(gauss5), None),
        ("Blurred Input", _bgr_to_rgb(blurred), None),
        ("Prewitt |G|", prewitt_mag, "gray"),
        ("Sobel |G|", sobel_mag, "gray"),
        ("Laplacian |L|", lap, "gray"),
        ("Canny", canny, "gray"),
        ("(blank)", np.full_like(prewitt_mag, 255), "gray"),
        ("(blank)", np.full_like(prewitt_mag, 255), "gray"),
    ]
    for ax, (title, image, cmap) in zip(axes.ravel(), items):
        if title == "(blank)":
            ax.axis("off")
            continue
        ax.imshow(image, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=255 if cmap == "gray" else None)
        ax.set_title(title)
        ax.axis("off")
    # Hide remaining blank axes cleanly
    for ax in axes.ravel():
        if ax.get_title() == "(blank)":
            ax.set_title("")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 2: spatial-domain neighborhood filtering.")
    parser.add_argument("--input", type=Path, default=None, help="Primary input image path.")
    parser.add_argument("--aux", type=Path, default=None, help="Auxiliary image for algebraic ops.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_experiment(input_path=args.input, aux_path=args.aux, output_dir=args.output_dir)
    print("Generated files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
