import numpy as np
from PIL import Image
import cv2


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_rgb(arr):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _gradient(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx = gx / mag
    ny = gy / mag
    return mag, nx, ny


def _coherence(nx, ny, size=7):
    kernel = np.ones((size, size), np.float32) / (size * size)
    cx = cv2.filter2D(nx, -1, kernel)
    cy = cv2.filter2D(ny, -1, kernel)
    return np.sqrt(cx**2 + cy**2)


def _rho_defects(mag):
    blur = cv2.GaussianBlur(mag, (9, 9), 0)
    diff = mag - blur
    rho = np.maximum(0, -diff)
    rho = rho / (rho.max() + 1e-6)
    return rho


def phi_filter_v2(path, strength=1.0):
    """
    Wersja 2 filtra φ:
    - Λ: koherencja kierunku
    - τ: gradient (przepływ)
    - ρ: defekty (lokalne minima)
    """
    img = _load_rgb(path)
    mag, nx, ny = _gradient(img)

    Lambda = _coherence(nx, ny)
    Tau = mag / (mag.max() + 1e-6)
    Rho = _rho_defects(mag)

    # operator φ = Λ + τ – ρ
    phi = Lambda + Tau - Rho
    phi = phi / (phi.max() + 1e-6)

    phi3 = np.stack([phi]*3, axis=-1)
    out = img * (0.4 + 0.6 * phi3)

    return _save_rgb(out)
