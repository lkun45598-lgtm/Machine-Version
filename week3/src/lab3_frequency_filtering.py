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
DEFAULT_INPUT = DATA_DIR / "input_images" / "vision_lab3_input.png"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"


def ensure_default_input(path: Path = DEFAULT_INPUT) -> Path:
    """Use skimage.data.astronaut() as the standard test image for FFT experiments."""
    if path.exists():
        return path
    from skimage import data

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = data.astronaut()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return path


def fft_spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered FFT, log-magnitude spectrum, and phase spectrum (all float)."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    phase = np.angle(fshift)
    return fshift, magnitude, phase


def rect_highpass_mask(shape: tuple[int, int], half: int = 10) -> np.ndarray:
    """Rectangular high-pass mask: 1 everywhere, 0 in a (2·half+1)^2 block at center."""
    rows, cols = shape
    mid_r, mid_c = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[mid_r - half : mid_r + half + 1, mid_c - half : mid_c + half + 1] = 0.0
    return mask


def rect_lowpass_mask(shape: tuple[int, int], half: int = 10) -> np.ndarray:
    """Rectangular low-pass mask: 0 everywhere, 1 in a (2·half+1)^2 block at center.

    By construction this is the exact complement of the rectangular high-pass mask.
    """
    rows, cols = shape
    mid_r, mid_c = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[mid_r - half : mid_r + half + 1, mid_c - half : mid_c + half + 1] = 1.0
    return mask


def apply_mask(fshift: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multiply centered spectrum by mask, then inverse FFT.

    Returns (filtered_image_float, log_magnitude_of_masked_spectrum).
    """
    masked = fshift * mask
    inv_shift = np.fft.ifftshift(masked)
    img = np.abs(np.fft.ifft2(inv_shift))
    masked_log = np.log1p(np.abs(masked))
    return img, masked_log


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize a float array to [0, 255] uint8 for display/saving."""
    image = image.astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-9:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - lo) * 255.0 / (hi - lo)).astype(np.uint8)


def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Return per-pixel Euclidean distance to the spectrum center."""
    rows, cols = shape
    u = np.arange(rows).reshape(-1, 1) - rows // 2
    v = np.arange(cols).reshape(1, -1) - cols // 2
    return np.sqrt(u * u + v * v).astype(np.float32)


def ideal_lowpass(shape: tuple[int, int], d0: float) -> np.ndarray:
    return (distance_grid(shape) <= d0).astype(np.float32)


def butterworth_lowpass(shape: tuple[int, int], d0: float, n: int = 2) -> np.ndarray:
    d = distance_grid(shape)
    return (1.0 / (1.0 + (d / max(d0, 1e-6)) ** (2 * n))).astype(np.float32)


def gaussian_lowpass(shape: tuple[int, int], d0: float) -> np.ndarray:
    d = distance_grid(shape)
    return np.exp(-(d ** 2) / (2.0 * d0 ** 2)).astype(np.float32)


def frequency_filter(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a centered-spectrum mask to a grayscale image and return uint8 result."""
    fshift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    inv = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
    return normalize_uint8(np.abs(inv))


def save_gray(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def show_gray(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    save_gray(output_dir / "01_original_color.png", bgr)
    save_gray(output_dir / "02_original_gray.png", gray)

    # ----- Step 1: FFT spectra (magnitude + phase) -----
    fshift, mag, phase = fft_spectrum(gray)
    save_gray(output_dir / "10_magnitude_spectrum.png", normalize_uint8(mag))
    save_gray(output_dir / "11_phase_spectrum.png", normalize_uint8(phase))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original (color)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    show_gray(axes[1], mag, "Magnitude spectrum  log|F|")
    show_gray(axes[2], phase, "Phase spectrum  angle(F)")
    save_figure(fig, output_dir / "12_amp_phase_compare.png")

    # ----- Step 2: Rectangular high/low-pass masks (the PPT exercise) -----
    half = 10
    hp_mask = rect_highpass_mask(gray.shape, half=half)
    lp_mask = rect_lowpass_mask(gray.shape, half=half)
    save_gray(output_dir / "20_mask_highpass_rect.png", (hp_mask * 255).astype(np.uint8))
    save_gray(output_dir / "21_mask_lowpass_rect.png", (lp_mask * 255).astype(np.uint8))

    img_hp, spec_hp = apply_mask(fshift, hp_mask)
    img_lp, spec_lp = apply_mask(fshift, lp_mask)
    save_gray(output_dir / "22_rect_highpass_img.png", normalize_uint8(img_hp))
    save_gray(output_dir / "23_rect_lowpass_img.png", normalize_uint8(img_lp))

    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    show_gray(axes[0, 0], gray, "Original")
    show_gray(axes[0, 1], mag, "Original FFT  log|F|")
    show_gray(axes[1, 0], img_hp, "High-pass result")
    show_gray(axes[1, 1], spec_hp, "High-pass FFT")
    show_gray(axes[2, 0], img_lp, "Low-pass result")
    show_gray(axes[2, 1], spec_lp, "Low-pass FFT")
    save_figure(fig, output_dir / "24_rect_filtering_compare.png")

    # ----- Step 3: Spatial vs frequency domain comparison -----
    spatial_lp = cv2.GaussianBlur(gray, (15, 15), 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    show_gray(axes[0], gray, "Original")
    show_gray(axes[1], spatial_lp, "Spatial low-pass  Gaussian 15x15")
    show_gray(axes[2], img_lp, "Frequency low-pass  (rect mask)")
    save_figure(fig, output_dir / "30_spatial_vs_freq_lowpass.png")

    laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    show_gray(axes[0], gray, "Original")
    show_gray(axes[1], laplacian, "Spatial high-pass  Laplacian")
    show_gray(axes[2], img_hp, "Frequency high-pass  (rect mask)")
    save_figure(fig, output_dir / "31_spatial_vs_freq_highpass.png")

    # ----- Step 4 (extension): Ideal / Butterworth / Gaussian frequency filters -----
    d0 = 40.0
    n = 2
    ideal_lp = ideal_lowpass(gray.shape, d0)
    butter_lp = butterworth_lowpass(gray.shape, d0, n=n)
    gauss_lp = gaussian_lowpass(gray.shape, d0)
    ideal_hp = 1.0 - ideal_lp
    butter_hp = 1.0 - butter_lp
    gauss_hp = 1.0 - gauss_lp

    out_ideal_lp = frequency_filter(gray, ideal_lp)
    out_butter_lp = frequency_filter(gray, butter_lp)
    out_gauss_lp = frequency_filter(gray, gauss_lp)
    out_ideal_hp = frequency_filter(gray, ideal_hp)
    out_butter_hp = frequency_filter(gray, butter_hp)
    out_gauss_hp = frequency_filter(gray, gauss_hp)

    save_gray(output_dir / "40_ideal_lpf.png", out_ideal_lp)
    save_gray(output_dir / "41_butter_lpf.png", out_butter_lp)
    save_gray(output_dir / "42_gauss_lpf.png", out_gauss_lp)
    save_gray(output_dir / "43_ideal_hpf.png", out_ideal_hp)
    save_gray(output_dir / "44_butter_hpf.png", out_butter_hp)
    save_gray(output_dir / "45_gauss_hpf.png", out_gauss_hp)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    show_gray(axes[0, 0], gray, "Original")
    show_gray(axes[0, 1], out_ideal_lp, f"Ideal LPF  D0={int(d0)}")
    show_gray(axes[0, 2], out_butter_lp, f"Butterworth LPF  n={n}")
    show_gray(axes[0, 3], out_gauss_lp, "Gaussian LPF")
    show_gray(axes[1, 0], ideal_lp, "Ideal LPF mask")
    show_gray(axes[1, 1], butter_lp, "Butterworth LPF mask")
    show_gray(axes[1, 2], gauss_lp, "Gaussian LPF mask")
    axes[1, 3].axis("off")
    save_figure(fig, output_dir / "50_freq_filter_compare_lpf.png")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    show_gray(axes[0, 0], gray, "Original")
    show_gray(axes[0, 1], out_ideal_hp, f"Ideal HPF  D0={int(d0)}")
    show_gray(axes[0, 2], out_butter_hp, f"Butterworth HPF  n={n}")
    show_gray(axes[0, 3], out_gauss_hp, "Gaussian HPF")
    show_gray(axes[1, 0], ideal_hp, "Ideal HPF mask")
    show_gray(axes[1, 1], butter_hp, "Butterworth HPF mask")
    show_gray(axes[1, 2], gauss_hp, "Gaussian HPF mask")
    axes[1, 3].axis("off")
    save_figure(fig, output_dir / "51_freq_filter_compare_hpf.png")

    # ----- Aggregate result -----
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    show_gray(axes[0, 0], gray, "Original")
    show_gray(axes[0, 1], mag, "Magnitude")
    show_gray(axes[0, 2], phase, "Phase")
    axes[0, 3].axis("off")
    show_gray(axes[1, 0], img_lp, "Rect LPF")
    show_gray(axes[1, 1], out_ideal_lp, "Ideal LPF")
    show_gray(axes[1, 2], out_butter_lp, "Butterworth LPF")
    show_gray(axes[1, 3], out_gauss_lp, "Gaussian LPF")
    show_gray(axes[2, 0], img_hp, "Rect HPF")
    show_gray(axes[2, 1], out_ideal_hp, "Ideal HPF")
    show_gray(axes[2, 2], out_butter_hp, "Butterworth HPF")
    show_gray(axes[2, 3], out_gauss_hp, "Gaussian HPF")
    save_figure(fig, output_dir / "70_all_results.png")

    # ----- Metrics: how well do freq low-pass variants approximate the spatial Gaussian? -----
    def mse(a, b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        return float(np.mean((a - b) ** 2))

    def psnr(a, b):
        m = mse(a, b)
        if m < 1e-9:
            return float("inf")
        return 10.0 * float(np.log10(255.0 ** 2 / m))

    metrics = [
        ("rect_lowpass vs spatial_gaussian",
         mse(normalize_uint8(img_lp), spatial_lp), psnr(normalize_uint8(img_lp), spatial_lp)),
        ("gauss_lowpass vs spatial_gaussian",
         mse(out_gauss_lp, spatial_lp), psnr(out_gauss_lp, spatial_lp)),
        ("ideal_lowpass vs spatial_gaussian",
         mse(out_ideal_lp, spatial_lp), psnr(out_ideal_lp, spatial_lp)),
        ("butter_lowpass vs spatial_gaussian",
         mse(out_butter_lp, spatial_lp), psnr(out_butter_lp, spatial_lp)),
        ("rect_highpass vs spatial_laplacian",
         mse(normalize_uint8(img_hp), laplacian), psnr(normalize_uint8(img_hp), laplacian)),
    ]
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("name, MSE, PSNR(dB)\n")
        for name, m_, p_ in metrics:
            f.write(f"{name}, {m_:.3f}, {p_:.3f}\n")

    print(f"All results saved under: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 3: Frequency-domain filtering")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input image path. Defaults to skimage astronaut().")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results.")
    args = parser.parse_args()

    input_path = args.input if args.input else ensure_default_input()
    run(input_path, args.output_dir)


if __name__ == "__main__":
    main()
