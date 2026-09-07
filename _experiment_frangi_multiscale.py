"""
_experiment_frangi_multiscale.py -- SPRAWDZENIE (nie produkcyjny
skrypt): prawdziwa wersja multi-scale, oparta na HESJANIE (druga
pochodna, znormalizowana wzgledem skali), a nie na naszej koherencji z
tensora struktury (pierwsza pochodna) -- bo poprzedni eksperyment
(_experiment_multiscale.py) pokazal, ze zwykla koherencja NIE jest
czula na skale (izolowana linia daje ta sama koherencje niezaleznie od
window_sigma), wiec "policz przy kilku sigma, wez max" nic tam nie dawal.

To jest uproszczona wersja filtra Frangiego (Frangi i wsp. 1998,
klasyczna metoda segmentacji naczyn siatkowki/angiografii -- WPROST
zwiazana z nasza domena retina_vessel_orientation.py):

1. Wygladz obraz Gaussem o sigma s.
2. Policz Hesjan (Ixx, Ixy, Iyy) na wygladzonym obrazie.
3. ZNORMALIZUJ wzgledem skali: pomnoz przez s^2 (normalizacja
   Lindeberga dla pochodnej 2. rzedu) -- BEZ TEGO odpowiedzi z roznych
   skal nie sa ze soba porownywalne (silniejsze rozmycie zawsze daje
   mniejsze surowe pochodne).
4. Wartosci wlasne Hesjanu (lambda1, lambda2, |lambda1|<=|lambda2|):
   - Rb = lambda1/lambda2 (blobness -- linia: mala, plama: duza)
   - S  = sqrt(lambda1^2+lambda2^2) (structureness -- sila krawedzi)
   - vesselness = exp(-Rb^2/2beta^2) * (1-exp(-S^2/2c^2)),
     zerowane tam gdzie lambda2>0 (odrzuca ciemne/wklesle struktury,
     zostawia jasne grzbiety/linie na ciemnym tle)
5. Multi-scale: policz vesselness dla kilku sigma, wez MAX per piksel.

Test: to samo co poprzednio -- dwie rownolegle linie o roznej grubosci
(2px i 14px) -- ale tym razem sprawdzamy, czy POJEDYNCZA sigma
FAKTYCZNIE traci jedna z nich (co powinno sie zdarzyc, jesli metoda jest
prawdziwie czula na skale), i czy multi-scale max odzyskuje obie.

Uzycie:
    python _experiment_frangi_multiscale.py
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
    lambda1 = np.where(swap, b, a)   # |lambda1| <= |lambda2|
    lambda2 = np.where(swap, a, b)
    return lambda1, lambda2


def frangi_stack(gray, sigmas, beta=0.5):
    """
    POPRAWKA (po pierwszym, blednym uruchomieniu): `c` (normalizacja
    structureness) NIE moze byc liczone OSOBNO dla kazdej sigma z jej
    wlasnego S.max() -- to samo-normalizuje kazda skale do wlasnej
    relatywnej skali i niszczy WLASNIE TA informacje o bezwzglednym
    spadku/wzroscie odpowiedzi z sigma, ktora ma znaczenie dla
    wykrycia dopasowania skali. Zweryfikowano: z c per-sigma odpowiedz
    na pojedynczej linii rosla MONOTONICZNIE z sigma (0.12 do 0.50, bez
    maksimum) -- co jest sprzeczne z oczekiwanym zachowaniem Frangiego.
    Naprawiono: jedno, STALE `c` liczone raz z globalnego maksimum S po
    WSZYSTKICH skalach naraz (tak jak w referencyjnych implementacjach
    Frangiego -- normalizacja wzgledem calego stosu skal, nie per-skala).

    Zwraca slownik {sigma: vesselness}.
    """
    all_lambda1, all_lambda2, all_S = {}, {}, {}
    for s in sigmas:
        l1, l2 = _hessian_eigs(gray, s)
        all_lambda1[s] = l1
        all_lambda2[s] = l2
        all_S[s] = np.sqrt(l1 ** 2 + l2 ** 2)

    global_s_max = max(S.max() for S in all_S.values())
    c = 0.5 * global_s_max if global_s_max > 1e-9 else 1.0

    result = {}
    for s in sigmas:
        lambda1, lambda2, S = all_lambda1[s], all_lambda2[s], all_S[s]
        Rb = lambda1 / (lambda2 + 1e-12)
        vesselness = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1 - np.exp(-(S ** 2) / (2 * c ** 2)))
        vesselness[lambda2 > 0] = 0.0  # tylko jasne grzbiety na ciemnym tle
        result[s] = vesselness
    return result


def _two_lines_image(size=200, thin_w=2, thick_w=14, gap=60):
    img = np.zeros((size, size), dtype=np.float32)
    y_thin = size // 2 - gap // 2
    y_thick = size // 2 + gap // 2
    cv2.line(img, (10, y_thin), (size - 10, y_thin), color=255, thickness=thin_w, lineType=cv2.LINE_AA)
    cv2.line(img, (10, y_thick), (size - 10, y_thick), color=255, thickness=thick_w, lineType=cv2.LINE_AA)
    return img, y_thin, y_thick


def _band_mean(response, y_center, half_band, x_range=(15, 185)):
    y0, y1 = y_center - half_band, y_center + half_band
    x0, x1 = x_range
    return float(response[y0:y1, x0:x1].mean())


def main():
    size = 200
    img, y_thin, y_thick = _two_lines_image(size=size)

    print("=== Krok 1: sanity check -- odpowiedz Frangiego na SAMEJ cienkiej linii, kilka sigma ===")
    print("(oczekiwanie: odpowiedz powinna miec MAKSIMUM w okolicy sigma ~ szerokosc/2, nie rosnac bez konca)\n")
    single_thin = np.zeros((size, size), dtype=np.float32)
    cv2.line(single_thin, (10, size // 2), (size - 10, size // 2), color=255, thickness=2, lineType=cv2.LINE_AA)
    sanity_sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    sanity_stack = frangi_stack(single_thin, sanity_sigmas)
    for sigma in sanity_sigmas:
        m = _band_mean(sanity_stack[sigma], size // 2, half_band=8)
        print(f"  sigma={sigma:4.1f} -> srednia odpowiedz w pasie linii: {m:.4f}")

    print("\n=== Krok 2: dwie linie o roznej grubosci (2px i 14px), pojedyncze sigma vs multi-scale ===\n")

    sigmas_thin = [0.5, 1.0, 1.5]
    sigmas_thick = [4.0, 6.0, 8.0]
    all_sigmas = sorted(set(sigmas_thin + sigmas_thick))

    responses = frangi_stack(img, all_sigmas)

    print("--- pojedyncze skale ---")
    for s in all_sigmas:
        r = responses[s]
        m_thin = _band_mean(r, y_thin, half_band=8)
        m_thick = _band_mean(r, y_thick, half_band=8)
        print(f"  sigma={s:4.1f}: odpowiedz CIENKA={m_thin:.4f}   odpowiedz GRUBA={m_thick:.4f}")

    multiscale = np.maximum.reduce([responses[s] for s in all_sigmas])
    m_thin_ms = _band_mean(multiscale, y_thin, half_band=8)
    m_thick_ms = _band_mean(multiscale, y_thick, half_band=8)
    print(f"\n  MULTI-SCALE (max z {len(all_sigmas)} skal): odpowiedz CIENKA={m_thin_ms:.4f}   "
          f"odpowiedz GRUBA={m_thick_ms:.4f}")

    best_single_thin = max(_band_mean(responses[s], y_thin, half_band=8) for s in all_sigmas)
    best_single_thick = max(_band_mean(responses[s], y_thick, half_band=8) for s in all_sigmas)
    print(f"\n  (dla porownania: najlepsza POJEDYNCZA skala per linia z osobna: "
          f"cienka={best_single_thin:.4f}, gruba={best_single_thick:.4f})")


if __name__ == "__main__":
    main()
