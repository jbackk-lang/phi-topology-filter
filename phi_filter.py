import numpy as np
from PIL import Image
import cv2


def _to_array(img_path):
    img = Image.open(img_path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _from_array(arr):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _local_gradient_field(img):
    """
    Prosty wektorowy opis zmian jasności – baza pod Λ–τ–ρ.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx = gx / mag
    ny = gy / mag
    return mag, nx, ny


def _phi_operator(mag, nx, ny, strength=1.0):
    """
    Miejsce na operator φ – na razie: wzmocnienie struktur o spójnym kierunku.
    """
    # lokalna koherencja kierunku (przybliżenie Λ)
    kernel = np.ones((5, 5), np.float32) / 25.0
    cx = cv2.filter2D(nx, -1, kernel)
    cy = cv2.filter2D(ny, -1, kernel)
    coherence = np.sqrt(cx**2 + cy**2)

    # φ jako stabilizator: wzmacniamy miejsca o wysokiej koherencji
    phi_mask = (coherence ** strength)
    phi_mask = phi_mask / (phi_mask.max() + 1e-6)

    return phi_mask


def phi_filter_image(img_path, mode="lambda-tau-rho", strength=1.0):
    """
    Główna funkcja filtra φ.

    mode:
        - "lambda-tau-rho" – podbicie struktur Λ–τ–ρ
        - w przyszłości: inne tryby (np. mapa defektów ρ)
    """
    img = _to_array(img_path)
    mag, nx, ny = _local_gradient_field(img)
    phi_mask = _phi_operator(mag, nx, ny, strength=strength)

    # nakładamy φ na jasność
    phi_mask_3 = np.stack([phi_mask]*3, axis=-1)
    out = img * (0.5 + 0.5 * phi_mask_3)

    return _from_array(out)
