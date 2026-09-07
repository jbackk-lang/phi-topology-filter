"""
_experiment_network_scan.py -- SPRAWDZENIE, nie produkcyjny skrypt
domenowy: czy structure_tensor_coherence() faktycznie odroznia wzorce
skanowania portow (znane z literatury: wolny skan portow = przekatna
linia na wykresie czas-vs-port, szybki skan = pionowa linia) od
normalnego ruchu sieciowego (rozrzucone, losowe polaczenia)?

Uzasadnienie tego eksperymentu -- realny precedens w literaturze:
- "Interactive Visualization for Network and Port Scan Detection"
  (Muelder/Ma i in.) -- wolne skany portow pojawiaja sie jako punkty
  ulozone PO PRZEKATNEJ na wykresie (czas, port); szybkie skany jako
  linia PIONOWA.
- Patenty USPTO nt. "network terrain" (np. US7873046) licza gradient
  wektorowy per-wymiar ruchu sieciowego wprost po to, zeby wzmocnic
  wzorce ulozone wzdluz jednej osi -- koncepcyjnie to samo co structure
  tensor, tylko bez formalnej koherencji/kierunku.

POPRAWKA v2 (po pierwszym uruchomieniu): pierwsza wersja mierzyla
kierunek/koherencje w STALYM kole (promien 25px) wokol srodka obrazu.
Dla WOLNEGO skanu to byl blad metodologiczny -- przekatna rozciaga sie
na caly obraz 200x200, wiec stale kolo w srodku lapalo tylko maly,
mocno zaszumiony przez normalny ruch fragment linii, nie cala linie.
To dawalo zanizona/przesunieta ocene kierunku (-76 stopni zamiast
oczekiwanych ok. -45). Naprawione: maska analizy teraz podaza za
FAKTYCZNYMI punktami skanu (dylatacja wokol prawdziwych wspolrzednych
skanu), wiec pokrywa cala linie, nie jeden maly wycinek. Dodatkowo
liczona jest baseline na TEJ SAMEJ masce dla samego normalnego ruchu
(bez skanu) -- uczciwe porownanie jablko-do-jablka: ile koherencji
"dokladla" sama obecnosc skanu w dokladnie tym samym miejscu obrazu.

Druga poprawka: raportowany jest zarowno surowy kat z
structure_tensor_coherence() (to jest kierunek DOMINUJACEGO GRADIENTU,
prostopadly do widocznej linii -- zweryfikowane recznie: dla czysto
pionowej linii Jxx=0 daje formule orientation=90 stopni, czyli
gradient poziomy -- ZGADZA SIE, bo pionowa linia zmienia sie najszybciej
w kierunku POZIOMYM), jak i wyprowadzony z niego kierunek SAMEJ LINII
skanu (orientation + 90 stopni, mod 180) -- to drugie jest bardziej
intuicyjne do porownania z opisem z literatury ("przekatna" = ~45/-45
stopni linii, "pionowa" = ~90 stopni linii).

Ten skrypt NIE laczy sie z prawdziwym ruchem sieciowym ani z
TIMDR-Security-Module (dziala na szeregu czasowym 1D, nie na obrazie)
-- to czysto syntetyczny test numeryczny.

Uzycie:
    python _experiment_network_scan.py
"""

import numpy as np
import cv2

from phi_core import structure_tensor_coherence, lineament_score


def _connection_density_image(points_xy, size=200, blur_sigma=1.2):
    """points_xy: lista (x, y) w [0, size). Zwraca obraz gestosci (kazdy
    punkt = mala kropka), lekko rozmyty gaussem."""
    img = np.zeros((size, size), dtype=np.float32)
    for x, y in points_xy:
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        img[yi, xi] += 40.0
    img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    return img


def _normal_traffic(n=250, size=200, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, size, n)
    ys = rng.uniform(0, size, n)
    return list(zip(xs, ys))


def _slow_scan_diagonal(n=60, size=200, seed=1, jitter=1.5):
    """Wolny skan: port rosnie mniej-wiecej liniowo z czasem -- w
    literaturze: ulozenie po przekatnej. Kierunek LINII (nie gradientu)
    tej przekatnej to +45 stopni (x i y rosna razem)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(10, size - 10, n)
    port = np.linspace(10, size - 10, n) + rng.normal(0, jitter, n)
    return list(zip(t, port))


def _fast_scan_vertical(n=60, size=200, seed=2, t_center=100.0, jitter=1.5):
    """Szybki skan: wiele portow w waskim oknie czasowym -- w
    literaturze: linia pionowa. Kierunek LINII to 90 stopni."""
    rng = np.random.default_rng(seed)
    t = t_center + rng.normal(0, jitter, n)
    port = np.linspace(10, size - 10, n)
    return list(zip(t, port))


def _mask_around_points(points_xy, size=200, dilate_radius=6):
    """Maska pokrywajaca CALA linie skanu (dylatacja wokol prawdziwych
    punktow skanu), zamiast jednego stalego kola w srodku obrazu --
    to jest poprawka v2, patrz naglowek pliku."""
    pts_img = np.zeros((size, size), dtype=np.uint8)
    for x, y in points_xy:
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        pts_img[yi, xi] = 255
    kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
    dilated = cv2.dilate(pts_img, kernel)
    return dilated > 0


def _stats_on_mask(coherence, orientation, mag, mask, mag_thresh=0.1):
    mag_norm = mag / (mag.max() + 1e-9)
    weight = mask & (mag_norm > mag_thresh)
    if not np.any(weight):
        return {"mean_coherence": 0.0, "grad_orientation_deg": None,
                "line_orientation_deg": None, "n_px": 0}
    mean_coh = float(coherence[weight].mean())
    ang = orientation[weight]
    # usrednienie przez podwojny kat (orientacja zdefiniowana mod pi)
    mean_grad_ang = 0.5 * np.arctan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())
    mean_line_ang = mean_grad_ang + np.pi / 2
    # znormalizuj do [-90, 90) stopni
    line_deg = np.degrees(mean_line_ang)
    line_deg = ((line_deg + 90) % 180) - 90
    return {
        "mean_coherence": mean_coh,
        "grad_orientation_deg": float(np.degrees(mean_grad_ang)),
        "line_orientation_deg": float(line_deg),
        "n_px": int(weight.sum()),
    }


def main():
    size = 200
    window_sigma = 2.5

    normal_pts = _normal_traffic(n=250, size=size, seed=0)
    diagonal_pts = _slow_scan_diagonal(n=60, size=size, seed=1)
    vertical_pts = _fast_scan_vertical(n=60, size=size, seed=2, t_center=100.0)

    scans = {
        "WOLNY skan (przekatna, oczekiwany kierunek linii ~+45 stopni)": (diagonal_pts, 5),
        "SZYBKI skan (pionowa linia, oczekiwany kierunek linii ~90 stopni)": (vertical_pts, 6),
    }

    print("=== Eksperyment v2: koherencja ze structure tensor, maska podazajaca za linia skanu ===")
    print(f"(obraz {size}x{size}, os X = czas, os Y = port, window_sigma={window_sigma})\n")

    for name, (scan_pts, dilate_r) in scans.items():
        mask = _mask_around_points(scan_pts, size=size, dilate_radius=dilate_r)

        img_with_scan = _connection_density_image(normal_pts + scan_pts, size=size)
        img_baseline = _connection_density_image(normal_pts, size=size)

        coh_scan, ori_scan, mag_scan = structure_tensor_coherence(img_with_scan, window_sigma=window_sigma)
        coh_base, ori_base, mag_base = structure_tensor_coherence(img_baseline, window_sigma=window_sigma)

        stats_scan = _stats_on_mask(coh_scan, ori_scan, mag_scan, mask)
        stats_base = _stats_on_mask(coh_base, ori_base, mag_base, mask)

        print(f"--- {name} ---")
        print(f"  maska pokrywa {stats_scan['n_px']} pikseli wzdluz calej linii")
        print(f"  koherencja BASELINE (ten sam obszar, sam normalny ruch, BEZ skanu): "
              f"{stats_base['mean_coherence']:.3f}")
        print(f"  koherencja Z SKANEM (ten sam obszar):                              "
              f"{stats_scan['mean_coherence']:.3f}")
        print(f"  kierunek linii skanu (wyprowadzony, orientation+90): {stats_scan['line_orientation_deg']:.1f} stopni")
        print()

    # sam normalny ruch, bez zadnego skanu, jako ogolny punkt odniesienia (caly obraz)
    img_pure = _connection_density_image(normal_pts, size=size)
    coh_pure, _, mag_pure = structure_tensor_coherence(img_pure, window_sigma=window_sigma)
    score_pure = lineament_score(coh_pure, mag_pure)
    print(f"--- sam normalny ruch, caly obraz (ogolny szum tla) ---")
    print(f"  mean lineament_score (caly obraz): {score_pure.mean():.4f}")
    print(f"  mean lineament_score (95 percentyl): {np.percentile(score_pure, 95):.4f}")


if __name__ == "__main__":
    main()
