import numpy as np
from PIL import Image
import cv2


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def phi_structure_map(img_path):
    """
    Zwraca trzy mapy:
    - Λ: stabilne struktury (koherencja kierunku)
    - τ: przejścia fazowe (gradient)
    - ρ: defekty (lokalne minima/załamania)
    """
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = _to_gray((img * 255).astype(np.uint8))

    # gradienty
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # Λ — koherencja kierunku
    kernel = np.ones((7, 7), np.float32) / 49.0
    cx = cv2.filter2D(gx, -1, kernel)
    cy = cv2.filter2D(gy, -1, kernel)
    coherence = np.sqrt(cx**2 + cy**2)
    Lambda = _normalize(coherence)

    # τ — przejścia fazowe (gradient)
    Tau = _normalize(mag)

    # ρ — defekty (lokalne minima gradientu)
    blur = cv2.GaussianBlur(mag, (9, 9), 0)
    rho = mag - blur
    Rho = _normalize(np.maximum(0, -rho))

    return Lambda, Tau, Rho
