import numpy as np
from astropy.io import fits
from PIL import Image
import cv2


def _normalize(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-6)
    return arr


def load_fits(path, ext=0):
    """
    Wczytuje dane FITS (NASA/JWST/HST).
    ext – numer rozszerzenia (czasem dane są w HDU 1 lub 2).
    """
    hdul = fits.open(path)
    data = hdul[ext].data
    hdul.close()
    return _normalize(data)


def fits_to_rgb(data):
    """
    Konwersja danych FITS (1 kanał) do RGB.
    """
    img = (data * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _gradient(gray):
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
    return _normalize(rho)


def phi_fits(path, ext=0, mode="phi", strength=1.0):
    """
    Tryby:
        - "phi" – klasyczny filtr φ
        - "lambda" – mapa Λ (koherencja)
        - "tau" – mapa τ (gradient)
        - "rho" – mapa defektów ρ
        - "phi-mix" – φ = Λ + τ – ρ
    """
    data = load_fits(path, ext=ext)
    rgb = fits_to_rgb(data)
    gray = data

    mag, nx, ny = _gradient(gray)
    Lambda = _normalize(_coherence(nx, ny))
    Tau = _normalize(mag)
    Rho = _rho_defects(mag)

    if mode == "lambda":
        out = Lambda
    elif mode == "tau":
        out = Tau
    elif mode == "rho":
        out = Rho
    elif mode == "phi-mix":
        phi = Lambda + Tau - Rho
        out = _normalize(phi)
    else:  # "phi"
        phi = Lambda + Tau - Rho
        out = _normalize(phi)

    out_rgb = (out * 255).astype(np.uint8)
    return Image.fromarray(out_rgb)
