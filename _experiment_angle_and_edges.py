"""
_experiment_angle_and_edges.py -- SPRAWDZENIE (nie produkcyjny skrypt):
dwa osobne "warunki brzegowe", ktore moga tlumaczyc blad ~19 stopni na
przekatnej z poprzednich eksperymentow:

(A) BRZEG KATA: czy blad orientacji jest specyficzny dla 45 stopni, czy
    to ogolny efekt kwantyzacji na siatce pikseli (znany, udokumentowany
    problem: dyskretna siatka pikseli wprowadza systematyczny blad
    "schodkowy" w estymacji kierunku gradientu, ktory zwykle jest
    NAJWIEKSZY blisko 45 stopni i najmniejszy blisko 0/90 stopni --
    dokladnie to widzielismy: pionowa (90st) niemal idealna, przekatna
    (45st) najgorsza). Test: ciagla linia (cv2.line) pod katami
    0, 15, 30, 45, 60, 75, 90 stopni, zawsze W SRODKU obrazu (z dala od
    brzegu obrazu), zeby odizolowac ten efekt od (B).

(B) BRZEG OBRAZU: czy bliskosc KRAWEDZI CALEGO OBRAZU (gdzie
    cv2.GaussianBlur musi dopelniac piksele poza obrazem -- domyslnie
    BORDER_REFLECT_101) dodaje dodatkowy blad. Test: ta sama linia pod
    45 stopni, raz w pelni wysrodkowana, raz przesunieta blisko brzegu
    obrazu (odlegosc od krawedzi porownywalna z window_sigma).

Uzycie:
    python _experiment_angle_and_edges.py
"""

import numpy as np
import cv2

from phi_core import structure_tensor_coherence


def _draw_line_centered(size, angle_deg, length, thickness=2):
    """Rysuje ciagla linie o zadanym kacie, wysrodkowana w obrazie."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size / 2.0, size / 2.0
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad) * length / 2, np.sin(rad) * length / 2
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(img, p1, p2, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def _draw_line_near_edge(size, angle_deg, length, offset_from_edge, thickness=2):
    """Ta sama linia, ale przesunieta tak, ze jej srodek jest blisko
    krawedzi obrazu (odleglosc = offset_from_edge pikseli od brzegu)."""
    img = np.zeros((size, size), dtype=np.float32)
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad) * length / 2, np.sin(rad) * length / 2
    cx, cy = offset_from_edge, size / 2.0
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(img, p1, p2, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def _mask_from_image(img, thresh=5.0, dilate_radius=4, size=None):
    pts = (img > thresh).astype(np.uint8) * 255
    kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
    mask = cv2.dilate(pts, kernel) > 0
    return mask


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


def _expected_line_deg(raw_deg):
    """Sprowadza kat linii do tej samej konwencji [-90,90) co wynik
    (kierunek linii jest zdefiniowany mod 180 stopni)."""
    d = ((raw_deg + 90) % 180) - 90
    return d


def main():
    size = 200
    window_sigma = 2.5

    print("=== (A) Blad kierunku w funkcji KATA (linia zawsze wysrodkowana, z dala od brzegu) ===\n")
    for angle in (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165):
        img = _draw_line_centered(size, angle, length=140)
        mask = _mask_from_image(img)
        coherence, orientation, mag = structure_tensor_coherence(img, window_sigma=window_sigma)
        deg, coh = _line_direction_deg(coherence, orientation, mag, mask)
        expected = _expected_line_deg(angle)
        err = None
        if deg is not None:
            # blad kolowy (bo kierunek jest mod 180)
            raw_err = abs(deg - expected)
            err = min(raw_err, 180 - raw_err)
        print(f"  kat zadany={angle:4d} st. (oczekiwany={expected:6.1f}) -> "
              f"zmierzony={deg:6.1f} st., blad={err:5.1f} st., koherencja={coh:.3f}")

    print("\n=== (B) Ta sama linia 45 stopni: wysrodkowana vs blisko brzegu obrazu ===\n")
    img_center = _draw_line_centered(size, 45, length=140)
    img_edge = _draw_line_near_edge(size, 45, length=60, offset_from_edge=12)

    for name, img in (("wysrodkowana", img_center), ("blisko brzegu (odleglosc 12px)", img_edge)):
        mask = _mask_from_image(img)
        coherence, orientation, mag = structure_tensor_coherence(img, window_sigma=window_sigma)
        deg, coh = _line_direction_deg(coherence, orientation, mag, mask)
        err = abs(deg - 45.0) if deg is not None else None
        print(f"  {name}: kierunek={deg:.1f} st. (blad={err:.1f}), koherencja={coh:.3f}")


if __name__ == "__main__":
    main()
