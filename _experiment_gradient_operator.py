"""
_experiment_gradient_operator.py -- SPRAWDZENIE (nie produkcyjny
skrypt): czy zmiana operatora gradientu zmniejsza znany, udokumentowany
efekt anizotropii Sobela 3x3 (gorsza dokladnosc kierunku na
krawedziach/liniach UKOSNYCH nizniach poziomych/pionowych), ktory w
_experiment_network_scan.py dawal ok. 19 stopni bledu na przekatnym
skanie (25.6 zamiast oczekiwanych 45), podczas gdy pionowy skan wyszedl
niemal idealnie (-88.6 zamiast -90).

Testowane warianty:
- Sobel ksize=3 (obecny domyslny w phi_core.structure_tensor_coherence)
- Sobel ksize=5 (wiekszy kernel, juz obslugiwany przez istniejacy
  parametr grad_ksize -- NIE wymaga zmian w phi_core.py)
- Scharr (3x3, ale ze wspolczynnikami zoptymalizowanymi pod katem
  ISOTROPII rotacyjnej -- klasyczny, udokumentowany fix na dokladnie
  ten problem, opisany przez samego autora Sobela/Scharra)

NIE modyfikuje phi_core.py -- to tymczasowy eksperyment lokalny,
zeby zdecydowac, ktory wariant (jesli ktorykolwiek) warto uzyc w
network_scan_orientation.py.

Uzycie:
    python _experiment_gradient_operator.py
"""

import numpy as np
import cv2


def _structure_tensor_variant(gray, operator="sobel3", window_sigma=2.5):
    gray = gray.astype(np.float32)
    if operator == "sobel3":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    elif operator == "sobel5":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    elif operator == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    else:
        raise ValueError(operator)

    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), window_sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), window_sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), window_sigma)

    trace = Jxx + Jyy
    diff = Jxx - Jyy
    disc = np.sqrt(diff * diff + 4 * Jxy * Jxy)
    coherence = np.clip(disc / (trace + 1e-9), 0.0, 1.0)
    orientation = 0.5 * np.arctan2(2 * Jxy, diff)
    mag = np.sqrt(gx * gx + gy * gy)
    return coherence, orientation, mag


def _connection_density_image(points_xy, size=200, blur_sigma=1.2):
    img = np.zeros((size, size), dtype=np.float32)
    for x, y in points_xy:
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        img[yi, xi] += 40.0
    return cv2.GaussianBlur(img, (0, 0), blur_sigma)


def _normal_traffic(n=250, size=200, seed=0):
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(0, size, n), rng.uniform(0, size, n)))


def _slow_scan_diagonal(n=60, size=200, seed=1, jitter=1.5):
    rng = np.random.default_rng(seed)
    t = np.linspace(10, size - 10, n)
    port = np.linspace(10, size - 10, n) + rng.normal(0, jitter, n)
    return list(zip(t, port))


def _fast_scan_vertical(n=60, size=200, seed=2, t_center=100.0, jitter=1.5):
    rng = np.random.default_rng(seed)
    t = t_center + rng.normal(0, jitter, n)
    port = np.linspace(10, size - 10, n)
    return list(zip(t, port))


def _mask_around_points(points_xy, size=200, dilate_radius=6):
    pts_img = np.zeros((size, size), dtype=np.uint8)
    for x, y in points_xy:
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        pts_img[yi, xi] = 255
    kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
    return cv2.dilate(pts_img, kernel) > 0


def _line_direction_deg(coherence, orientation, mag, mask, mag_thresh=0.1):
    mag_norm = mag / (mag.max() + 1e-9)
    weight = mask & (mag_norm > mag_thresh)
    if not np.any(weight):
        return None, 0.0
    ang = orientation[weight]
    mean_grad_ang = 0.5 * np.arctan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())
    mean_line_ang = mean_grad_ang + np.pi / 2
    deg = np.degrees(mean_line_ang)
    deg = ((deg + 90) % 180) - 90
    return float(deg), float(coherence[weight].mean())


def main():
    size = 200
    window_sigma = 2.5
    normal_pts = _normal_traffic(n=250, size=size, seed=0)
    diagonal_pts = _slow_scan_diagonal(n=60, size=size, seed=1)
    vertical_pts = _fast_scan_vertical(n=60, size=size, seed=2)

    diag_mask = _mask_around_points(diagonal_pts, size=size, dilate_radius=5)
    vert_mask = _mask_around_points(vertical_pts, size=size, dilate_radius=6)

    img_diag = _connection_density_image(normal_pts + diagonal_pts, size=size)
    img_vert = _connection_density_image(normal_pts + vertical_pts, size=size)

    print("=== Porownanie operatorow gradientu: bias kierunku na linii UKOSNEJ vs PIONOWEJ ===")
    print("Oczekiwane: przekatna ~45 stopni (skan rosnacy z czasem), pionowa ~90 stopni.\n")

    for op in ("sobel3", "sobel5", "scharr"):
        coh_d, ori_d, mag_d = _structure_tensor_variant(img_diag, operator=op, window_sigma=window_sigma)
        coh_v, ori_v, mag_v = _structure_tensor_variant(img_vert, operator=op, window_sigma=window_sigma)

        deg_d, coh_mean_d = _line_direction_deg(coh_d, ori_d, mag_d, diag_mask)
        deg_v, coh_mean_v = _line_direction_deg(coh_v, ori_v, mag_v, vert_mask)

        err_d = abs(deg_d - 45.0) if deg_d is not None else None
        err_v = abs(abs(deg_v) - 90.0) if deg_v is not None else None

        print(f"--- {op} ---")
        print(f"  przekatna: kierunek={deg_d:.1f} stopni (blad={err_d:.1f}), koherencja={coh_mean_d:.3f}")
        print(f"  pionowa:   kierunek={deg_v:.1f} stopni (blad={err_v:.1f}), koherencja={coh_mean_v:.3f}")
        print()


if __name__ == "__main__":
    main()
