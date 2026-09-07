"""
_experiment_scale_resonance.py -- SPRAWDZENIE (nie produkcyjny skrypt):
czy "rezonans M" (TIMDR: koincydencja >=3 sygnalow w tym samym "czasie",
tu przeniesiona na przestrzen SKALI zamiast czasu) faktycznie dziala
jako filtr antyszumowy dla multi-scale Frangiego, tak jak zaproponowano:

    RM(x,y) = #{sigma_i : V(x,y,sigma_i) > T} / #sigma_i

Hipoteza do sprawdzenia: prawdziwa struktura (linia/naczynie) daje
CIAGLY grzbiet odpowiedzi w kilku SASIEDNICH skalach (bo jej szerokosc
fizycznie "pasuje" do calego zakresu bliskich sigma), podczas gdy
pojedynczy piksel szumu trafia zwykle TYLKO JEDNA skale przypadkiem --
wiec rezonans (odsetek zgadzajacych sie skal) powinien byc WYSOKI na
prawdziwej linii i NISKI na szumie, nawet jesli surowe MAX(V) po
skalach jest podobnie wysokie w obu przypadkach (bo max zawsze bierze
najlepszy przypadek, nie sprawdzajac spojnosci miedzy skalami).

Test: obraz z dwiema liniami (2px, 14px) + DODANY SZUM (losowe jasne
plamki, imitujace zakloacenia/artefakty) -- porownanie:
1. surowy MAX(V) po skalach (bez rezonansu) -- ile pikseli szumu
   przechodzi prog?
2. RM(x,y) (rezonans) -- ile pikseli szumu przechodzi TEN SAM prog?
3. Czy prawdziwe linie NADAL maja wysoki rezonans (nie zostaly
   przypadkiem stlumione)?

Uzycie:
    python _experiment_scale_resonance.py
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
    """RM(x,y) = odsetek skal, dla ktorych V(x,y,sigma) > threshold."""
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
        # unikaj miejsc na samych liniach -- to ma byc SZUM TLA, nie czesc linii
        if abs(y - y_thin) < 10 or abs(y - y_thick) < 10:
            continue
        r = rng.integers(1, 3)
        cv2.circle(img, (int(x), int(y)), int(r), color=float(rng.uniform(150, 255)), thickness=-1)
        noise_coords.append((int(x), int(y)))

    return img, y_thin, y_thick, noise_coords


def main():
    size = 200
    threshold = 0.10
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    img, y_thin, y_thick, noise_coords = _two_lines_with_noise(size=size)
    print(f"Wygenerowano {len(noise_coords)} plamek szumu tla (poza liniami).\n")

    stack = frangi_stack(img, sigmas)
    max_v = np.maximum.reduce(list(stack.values()))
    resonance = scale_resonance(stack, threshold=threshold)

    def band_mask(y_center, half_band=8, x_range=(15, 185)):
        m = np.zeros((size, size), dtype=bool)
        m[y_center - half_band:y_center + half_band, x_range[0]:x_range[1]] = True
        return m

    thin_mask = band_mask(y_thin)
    thick_mask = band_mask(y_thick)
    line_mask = thin_mask | thick_mask

    print("=== Na PRAWDZIWYCH liniach (powinno zostac WYSOKO w obu metodach) ===")
    print(f"  MAX(V)     -- cienka: {max_v[thin_mask].mean():.3f}   gruba: {max_v[thick_mask].mean():.3f}")
    print(f"  rezonans RM -- cienka: {resonance[thin_mask].mean():.3f}   gruba: {resonance[thick_mask].mean():.3f}")

    # POPRAWKA: prog decyzyjny dla REZONANSU nie moze byc tym samym `threshold`
    # co prog per-skala T (ktory ustala, czy dana skala w ogole "glosuje").
    # RM jest UŁAMKIEM glosow (0, 1/8, 2/8, ... 1.0) -- prog 0.10 oznaczal w
    # praktyce "wystarczy 1 z 8 skal", czyli DOKLADNIE to samo co MAX (stad
    # identyczne wyniki powyzej). TIMDR-M wymaga >=3 zgodnych sygnalow z
    # definicji -- wiec prog decyzyjny na REZONANSIE to n_votes>=3, nie
    # jakikolwiek prog na V.
    n_sigmas = len(sigmas)
    resonance_decision_threshold = 3.0 / n_sigmas  # >=3 z 8 skal, zgodnie z TIMDR-M

    print(f"\n(n_sigmas={n_sigmas}, prog decyzyjny rezonansu = 3/{n_sigmas} = {resonance_decision_threshold:.3f})")

    print("\n=== Na PUNKTACH SZUMU TLA (powinno spasc NISKO tylko w rezonansie) ===")
    noise_max_vals = np.array([max_v[y, x] for x, y in noise_coords])
    noise_res_vals = np.array([resonance[y, x] for x, y in noise_coords])
    print(f"  MAX(V)      -- srednia na szumie: {noise_max_vals.mean():.3f}, "
          f"% szumu > prog({threshold}): {(noise_max_vals > threshold).mean() * 100:.1f}%")
    print(f"  rezonans RM -- srednia na szumie: {noise_res_vals.mean():.3f}, "
          f"% szumu z >=3 zgodnych skal: {(noise_res_vals >= resonance_decision_threshold).mean() * 100:.1f}%")

    print("\n=== Falszywie dodatnie poza liniami I poza oznaczonymi punktami szumu (caly szumiacy obszar) ===")
    background_area = ~line_mask
    fp_max = (max_v[background_area] > threshold).mean() * 100
    fp_res = (resonance[background_area] >= resonance_decision_threshold).mean() * 100
    print(f"  % tla > prog -- MAX(V): {fp_max:.2f}%   rezonans RM (>=3 skale): {fp_res:.2f}%")

    print("\n=== Czy prawdziwe linie NADAL przechodza prog rezonansu (>=3 skale)? ===")
    print(f"  cienka: {(resonance[thin_mask] >= resonance_decision_threshold).mean() * 100:.1f}% pikseli OK")
    print(f"  gruba:  {(resonance[thick_mask] >= resonance_decision_threshold).mean() * 100:.1f}% pikseli OK")


if __name__ == "__main__":
    main()
