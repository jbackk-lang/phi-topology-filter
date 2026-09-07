"""
_experiment_multiscale.py -- SPRAWDZENIE (nie produkcyjny skrypt): czy
pojedynczy `window_sigma` w structure_tensor_coherence() gubi struktury
o innej szerokosci/skali niz ta, do ktorej zostal dobrany -- i czy
kombinacja WIELU skal (jak w Frangi vesselness albo multi-scale Canny --
policz odpowiedz przy kilku sigma, wez NAJLEPSZA per piksel) faktycznie
lapie obie jednoczesnie.

Test: obraz z DWIEMA rownoleglymi liniami o BARDZO roznej grubosci
(cienka: 2px, gruba: 14px) -- imituje np. dwa lineamenty geologiczne o
roznej skali, albo skan portow o dwoch roznych "predkosciach" (waski
=szybki, szeroki pasek = wolny/rozmyty w czasie).

Metryka: dla kazdej z dwoch linii osobno, jaki % jej pikseli ma
lineament_score powyzej progu -- przy window_sigma dobranym pod cienka
linie, pod gruba linie, i przy kombinacji obu skal (max per piksel).

Uzycie:
    python _experiment_multiscale.py
"""

import numpy as np
import cv2

from phi_core import structure_tensor_coherence, lineament_score


def _two_lines_image(size=200, thin_w=2, thick_w=14, gap=60):
    img = np.zeros((size, size), dtype=np.float32)
    y_thin = size // 2 - gap // 2
    y_thick = size // 2 + gap // 2
    cv2.line(img, (10, y_thin), (size - 10, y_thin), color=255, thickness=thin_w, lineType=cv2.LINE_AA)
    cv2.line(img, (10, y_thick), (size - 10, y_thick), color=255, thickness=thick_w, lineType=cv2.LINE_AA)
    return img, y_thin, y_thick


def _coverage(score, y_center, half_band, x_range=(15, 185), thresh=0.3):
    """% pikseli w pasie [y_center-half_band, y_center+half_band] x
    x_range, ktore maja score > thresh -- miara 'czy linia zostala
    wykryta na calej dlugosci'."""
    y0, y1 = y_center - half_band, y_center + half_band
    x0, x1 = x_range
    band = score[y0:y1, x0:x1]
    return float((band > thresh).mean())


def main():
    size = 200
    img, y_thin, y_thick = _two_lines_image(size=size)

    sigmas = {
        "maly (sigma=1.5, dobrany pod cienka linie)": 1.5,
        "duzy (sigma=6.0, dobrany pod gruba linie)": 6.0,
    }

    print("=== Pokrycie cienkiej i grubej linii przy roznych window_sigma ===\n")

    scores = {}
    for name, sigma in sigmas.items():
        coherence, orientation, mag = structure_tensor_coherence(img, window_sigma=sigma)
        score = lineament_score(coherence, mag)
        scores[sigma] = score
        cov_thin = _coverage(score, y_thin, half_band=8)
        cov_thick = _coverage(score, y_thick, half_band=8)
        print(f"--- {name} ---")
        print(f"  pokrycie CIENKIEJ linii (2px):  {cov_thin*100:5.1f}%")
        print(f"  pokrycie GRUBEJ linii (14px):   {cov_thick*100:5.1f}%")
        print()

    # multi-scale: max score per piksel z obu skal powyzej
    multiscale_score = np.maximum(scores[1.5], scores[6.0])
    cov_thin_ms = _coverage(multiscale_score, y_thin, half_band=8)
    cov_thick_ms = _coverage(multiscale_score, y_thick, half_band=8)
    print("--- multi-scale (max z sigma=1.5 i sigma=6.0 per piksel) ---")
    print(f"  pokrycie CIENKIEJ linii (2px):  {cov_thin_ms*100:5.1f}%")
    print(f"  pokrycie GRUBEJ linii (14px):   {cov_thick_ms*100:5.1f}%")


if __name__ == "__main__":
    main()
