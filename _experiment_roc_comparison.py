"""
_experiment_roc_comparison.py -- WLASCIWY test (poprzedni byl
metodologicznie niesprawiedliwy, patrz komentarz w
_experiment_resonance_v2.py i uwaga uzytkownika): zamiast porownywac
Vmax i Vfinal=Vmax*(0.5+0.5*RM) przy TYM SAMYM progu bezwzglednym
(co gwarantuje gorszy wynik Vfinal, bo mnoznik <=1 zawsze), dopasuj
prog OSOBNO dla kazdej metody tak, zeby obie dawaly TEN SAM poziom
przepuszczonego szumu (dopasowany punkt pracy) -- dopiero wtedy
porownaj pokrycie prawdziwych struktur. To jest klasyczna analiza ROC:
porownuj czulosc (pokrycie) przy tej samej swoistosci (poziom szumu),
nie przy tym samym surowym progu.

Uzycie:
    python _experiment_roc_comparison.py
"""

import numpy as np
import cv2

from phi_hessian import multi_scale_vesselness, scale_resonance


def _two_lines_with_noise(size=200, thin_w=2, thick_w=14, gap=60, n_noise=40, noise_seed=0):
    img = np.zeros((size, size), dtype=np.float32)
    y_thin = size // 2 - gap // 2
    y_thick = size // 2 + gap // 2
    cv2.line(img, (10, y_thin), (size - 10, y_thin), color=255, thickness=thin_w, lineType=cv2.LINE_AA)
    cv2.line(img, (10, y_thick), (size - 10, y_thick), color=255, thickness=thick_w, lineType=cv2.LINE_AA)
    rng = np.random.default_rng(noise_seed)
    noise_coords = []
    for _ in range(n_noise):
        x = rng.integers(15, size - 15)
        y = rng.integers(15, size - 15)
        if abs(y - y_thin) < 10 or abs(y - y_thick) < 10:
            continue
        r = rng.integers(1, 3)
        cv2.circle(img, (int(x), int(y)), int(r), color=float(rng.uniform(150, 255)), thickness=-1)
        noise_coords.append((int(x), int(y)))
    return img, y_thin, y_thick, noise_coords


def band_mask(size, y_center, half_band=8, x_range=(15, 185)):
    m = np.zeros((size, size), dtype=bool)
    m[y_center - half_band:y_center + half_band, x_range[0]:x_range[1]] = True
    return m


def _threshold_for_target_noise(values_at_noise, target_noise_rate, candidate_thresholds):
    """Znajdz prog (z listy kandydatow, malejaco po czulosci = rosnaco po
    progu) najblizszy zadanemu poziomowi szumu przechodzacego prog."""
    best_t, best_diff = None, None
    for t in candidate_thresholds:
        rate = float(np.mean(values_at_noise > t))
        diff = abs(rate - target_noise_rate)
        if best_diff is None or diff < best_diff:
            best_diff, best_t, best_rate = diff, t, rate
    return best_t, best_rate


def main():
    size = 200
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    vote_threshold = 0.10

    img, y_thin, y_thick, noise_coords = _two_lines_with_noise(size=size)
    thin_mask = band_mask(size, y_thin)
    thick_mask = band_mask(size, y_thick)

    vmax, stack = multi_scale_vesselness(img, sigmas, return_stack=True)
    rm = scale_resonance(stack, threshold=vote_threshold)
    vfinal = vmax * (0.5 + 0.5 * rm)

    noise_vmax = np.array([vmax[y, x] for x, y in noise_coords])
    noise_vfinal = np.array([vfinal[y, x] for x, y in noise_coords])

    candidates_vmax = np.linspace(0.0, vmax.max(), 400)
    candidates_vfinal = np.linspace(0.0, vfinal.max(), 400)

    print("=== Porownanie ROC: pokrycie struktur PRZY DOPASOWANYM poziomie szumu ===\n")
    print(f"{'cel szumu':>10} | {'Vmax prog':>10} {'Vmax szum':>10} {'cienka':>8} {'gruba':>8} || "
          f"{'Vfin prog':>10} {'Vfin szum':>10} {'cienka':>8} {'gruba':>8}")

    for target in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        t_vmax, rate_vmax = _threshold_for_target_noise(noise_vmax, target, candidates_vmax)
        t_vfinal, rate_vfinal = _threshold_for_target_noise(noise_vfinal, target, candidates_vfinal)

        thin_vmax = (vmax[thin_mask] > t_vmax).mean() * 100
        thick_vmax = (vmax[thick_mask] > t_vmax).mean() * 100
        thin_vfinal = (vfinal[thin_mask] > t_vfinal).mean() * 100
        thick_vfinal = (vfinal[thick_mask] > t_vfinal).mean() * 100

        print(f"{target*100:9.0f}% | {t_vmax:10.4f} {rate_vmax*100:9.1f}% {thin_vmax:7.1f}% {thick_vmax:7.1f}% || "
              f"{t_vfinal:10.4f} {rate_vfinal*100:9.1f}% {thin_vfinal:7.1f}% {thick_vfinal:7.1f}%")


if __name__ == "__main__":
    main()
