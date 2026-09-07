"""
_experiment_line_vs_dots.py -- SPRAWDZENIE: czy blad ~19 stopni na
przekatnej (z _experiment_network_scan.py) pochodzi z ALGORYTMU
(structure tensor / Sobel), czy z KONSTRUKCJI syntetycznego testu?

Hipoteza: skan byl rysowany jako rzadkie, POJEDYNCZE kropki (60 kropek
na linii dlugosci ~254px, odstep ~4.2px), lekko rozmyte (blur_sigma=1.2).
Kazda kropka osobno jest w przyblizeniu izotropowym blobem -- jej WLASNY
gradient wskazuje promieniscie od jej srodka, NIE wzdluz linii skanu.
"Kierunek linii" moze wylonic sie tylko wtedy, gdy sasiednie kropki
wystarczajaco sie zlewaja (window_sigma zbierania momentow >= odstep
miedzy kropkami) -- w przeciwnym razie usredniony kierunek to mieszanka
promienistych gradientow pojedynczych kropek, nie prawdziwy kierunek
linii. To bylby artefakt TESTU, nie wada structure tensor.

Test: ta sama przekatna (10,10)-(190,190), narysowana na 3 sposoby:
1. rzadkie kropki (jak w oryginalnym eksperymencie)
2. gestsze kropki (4x wiecej, mniejszy odstep)
3. CIAGLA linia (cv2.line, antyaliasing) -- brak przerw w ogole

Jesli (3) daje kierunek bliski 45 stopni, a (1) nie -- potwierdzone,
ze to byl artefakt rzadkiego probkowania, nie problem z phi_core.

Uzycie:
    python _experiment_line_vs_dots.py
"""

import numpy as np
import cv2

from phi_core import structure_tensor_coherence


def _sparse_dots(n=60, size=200, seed=1, jitter=1.5):
    rng = np.random.default_rng(seed)
    t = np.linspace(10, size - 10, n)
    port = np.linspace(10, size - 10, n) + rng.normal(0, jitter, n)
    img = np.zeros((size, size), dtype=np.float32)
    for x, y in zip(t, port):
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        img[yi, xi] += 40.0
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _dense_dots(n=240, size=200, seed=1, jitter=1.5):
    rng = np.random.default_rng(seed)
    t = np.linspace(10, size - 10, n)
    port = np.linspace(10, size - 10, n) + rng.normal(0, jitter, n)
    img = np.zeros((size, size), dtype=np.float32)
    for x, y in zip(t, port):
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        img[yi, xi] += 40.0
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _continuous_line(size=200):
    img = np.zeros((size, size), dtype=np.float32)
    cv2.line(img, (10, 10), (size - 10, size - 10), color=255, thickness=2, lineType=cv2.LINE_AA)
    return img


def _mask_from_image(img, thresh=5.0, dilate_radius=4):
    pts = (img > thresh).astype(np.uint8) * 255
    kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
    return cv2.dilate(pts, kernel) > 0


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
    window_sigma = 2.5
    variants = {
        "1. rzadkie kropki (60, odstep ~4.2px)": _sparse_dots(),
        "2. gestsze kropki (240, odstep ~1.05px)": _dense_dots(),
        "3. ciagla linia (cv2.line, antyaliasing)": _continuous_line(),
    }

    print("=== Ciagla linia vs kropki: skad bierze sie blad ~19 stopni na przekatnej? ===")
    print("Oczekiwany kierunek: 45 stopni.\n")

    for name, img in variants.items():
        mask = _mask_from_image(img)
        coherence, orientation, mag = structure_tensor_coherence(img, window_sigma=window_sigma)
        deg, coh = _line_direction_deg(coherence, orientation, mag, mask)
        err = abs(deg - 45.0) if deg is not None else None
        print(f"--- {name} ---")
        print(f"  kierunek={deg:.1f} stopni (blad={err:.1f}), koherencja={coh:.3f}")
        print()


if __name__ == "__main__":
    main()
