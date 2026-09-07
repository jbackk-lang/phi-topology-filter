"""
_experiment_resonance_v2.py -- SPRAWDZENIE (nie produkcyjny skrypt):
test poprawki zaproponowanej po _experiment_scale_resonance.py --
zamiast twardego progu rezonansu (>=3 z 8 skal, ktory tlumil cienka
linie do 31.2% pokrycia), uzyc:

1. gestszego probkowania malych sigma (dodaje 0.7, 1.2, 1.8, 2.5 do
   istniejacego zestawu) -- cienka linia dostaje wiecej "glosow".
2. rezonansu jako WAGI (miekkiej), nie twardego progu:
       Vfinal = Vmax * (0.5 + 0.5*RM)
   gdzie RM w [0,1] to odsetek skal z V>T w danym pikselu. Piksel bez
   ZADNEGO wsparcia z sasiednich skal (RM=0) jest stlumiony o polowe
   (mnoznik 0.5), nie wyzerowany -- a piksel z pelnym wsparciem (RM=1)
   zostaje bez zmian (mnoznik 1.0).

Uzycie:
    python _experiment_resonance_v2.py
"""

import numpy as np
import cv2


def _hessian_eigs(gray, sigma):
    gray = gray.astype(np.float64)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
    Ixx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3) * (sigma ** 2)
    Iyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3) * (sigma ** 2)
    Ixy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3) * (sigma ** 2)
    mean = (Ixx + Iyy) / 2.0
    diff = (Ixx - Iyy) / 2.0
    disc = np.sqrt(diff * diff + Ixy * Ixy)
    a = mean - disc
    b = mean + disc
    swap = np.abs(a) > np.abs(b)
    lambda1 = np.where(swap, b, a)
    lambda2 = np.where(swap, a, b)
    return lambda1, lambda2


def frangi_stack(gray, sigmas, beta=0.5):
    all_l1, all_l2, all_S = {}, {}, {}
    for s in sigmas:
        l1, l2 = _hessian_eigs(gray, s)
        all_l1[s], all_l2[s] = l1, l2
        all_S[s] = np.sqrt(l1 ** 2 + l2 ** 2)
    c = 0.5 * max(S.max() for S in all_S.values())
    c = c if c > 1e-9 else 1.0
    result = {}
    for s in sigmas:
        l1, l2, S = all_l1[s], all_l2[s], all_S[s]
        Rb = l1 / (l2 + 1e-12)
        v = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1 - np.exp(-(S ** 2) / (2 * c ** 2)))
        v[l2 > 0] = 0.0
        result[s] = v
    return result


def scale_resonance(stack, threshold):
    sigmas = list(stack.keys())
    votes = np.zeros_like(next(iter(stack.values())))
    for s in sigmas:
        votes += (stack[s] > threshold).astype(np.float64)
    return votes / len(sigmas)


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


def main():
    size = 200
    T = 0.10
    threshold_report = 0.10

    sigmas_v1 = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    sigmas_v2 = sorted(set(sigmas_v1 + [0.7, 1.2, 1.8, 2.5]))

    img, y_thin, y_thick, noise_coords = _two_lines_with_noise(size=size)
    thin_mask = band_mask(size, y_thin)
    thick_mask = band_mask(size, y_thick)
    line_mask = thin_mask | thick_mask
    background_area = ~line_mask

    for label, sigmas in (("v1 (8 skal, oryginalne)", sigmas_v1),
                           ("v2 (12 skal, gestsze male sigma)", sigmas_v2)):
        stack = frangi_stack(img, sigmas)
        vmax = np.maximum.reduce(list(stack.values()))
        rm = scale_resonance(stack, threshold=T)
        vfinal = vmax * (0.5 + 0.5 * rm)

        noise_vmax = np.array([vmax[y, x] for x, y in noise_coords])
        noise_vfinal = np.array([vfinal[y, x] for x, y in noise_coords])

        print(f"=== {label} ===")
        print(f"  pokrycie CIENKIEJ linii (Vmax>{threshold_report}):    "
              f"{(vmax[thin_mask] > threshold_report).mean() * 100:5.1f}%   "
              f"(Vfinal>{threshold_report}): {(vfinal[thin_mask] > threshold_report).mean() * 100:5.1f}%")
        print(f"  pokrycie GRUBEJ linii (Vmax>{threshold_report}):     "
              f"{(vmax[thick_mask] > threshold_report).mean() * 100:5.1f}%   "
              f"(Vfinal>{threshold_report}): {(vfinal[thick_mask] > threshold_report).mean() * 100:5.1f}%")
        print(f"  % szumu > prog:  Vmax={np.mean(noise_vmax > threshold_report) * 100:5.1f}%   "
              f"Vfinal={np.mean(noise_vfinal > threshold_report) * 100:5.1f}%")
        print(f"  % calego tla > prog:  Vmax={(vmax[background_area] > threshold_report).mean() * 100:5.2f}%   "
              f"Vfinal={(vfinal[background_area] > threshold_report).mean() * 100:5.2f}%")
        print()


if __name__ == "__main__":
    main()
