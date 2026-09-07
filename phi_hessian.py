"""
phi_hessian.py -- DRUGI rdzen matematyczny w tym repo, celowo OSOBNY od
phi_core.py. phi_core.py liczy koherencje/kierunek z tensora struktury
(macierz momentow gradientu, PIERWSZA pochodna) -- odpowiada na pytanie
"w ktora strone biegnie struktura". Ten plik liczy vesselness/ridgeness
z Hesjanu (macierz DRUGIEJ pochodnej, znormalizowana wzgledem skali wg
Lindeberga) -- odpowiada na INNE pytanie: "czy w tym miejscu jest
struktura o SZEROKOSCI pasujacej do danej skali".

DLACZEGO OSOBNY PLIK (nie rozszerzenie phi_core.py): zweryfikowano
eksperymentalnie (2026-09-08, patrz _experiment_multiscale.py w tym
repo), ze koherencja z phi_core.py NIE JEST czula na skale -- izolowana
linia daje TA SAMA koherencje niezaleznie od window_sigma (bo mag/gx/gy
sa liczone z surowego, nieznormalizowanego gradientu, niezaleznie od
window_sigma, ktory wplywa tylko na rozmycie momentow Jxx/Jyy/Jxy).
"Policz przy kilku sigma, wez max" na phi_core.structure_tensor_coherence
nic wiec nie daje. Prawdziwa czulosc na skale wymaga INNEJ matematyki
(Hesjan + normalizacja Lindeberga), nie tylko innego parametru -- stad
osobny plik, nie funkcja dodana do phi_core.py.

ZWALIDOWANE EKSPERYMENTALNIE (patrz _experiment_frangi_multiscale.py,
_experiment_scale_resonance.py, _experiment_resonance_v2.py w tym repo):

1. Pojedyncza skala, poprawnie znormalizowana (mnozenie przez sigma^2,
   normalizacja Lindeberga rzedu gamma=1): odpowiedz na IZOLOWANEJ linii
   2px ma prawdziwe MAKSIMUM w funkcji sigma (szczyt ~sigma=3, nie rosnie
   bez konca) -- W PRZECIWIENSTWIE do koherencji z phi_core.
2. KRYTYCZNE dla poprawnosci: stala `c` (normalizacja structureness w
   formule vesselness) MUSI byc policzona RAZ, z globalnego maksimum
   structureness po WSZYSTKICH skalach w stosie -- NIE osobno per
   skala. Pierwsza (bledna) probka z c per-skala dawala odpowiedz
   rosnaca MONOTONICZNIE z sigma (brak maksimum) -- bo samo-normalizacja
   per skala niszczy wlasnie informacje o dopasowaniu skali, ktora
   chcemy zmierzyc. `multi_scale_vesselness()` ponizej liczy `c`
   poprawnie (raz, globalnie).
3. Multi-scale (max z kilku sigma per piksel) FAKTYCZNIE lapie struktury
   o roznej szerokosci jednoczesnie: test z linia 2px i linia 14px --
   zadna POJEDYNCZA sigma z testowanego zakresu nie dawala dobrej
   odpowiedzi na obu naraz (np. sigma=1.5: cienka=0.116, gruba=0.092;
   sigma=8: gruba=0.586, cienka=0.080), multi-scale max dal cienka=0.184,
   gruba=0.589 -- lepiej niz jakakolwiek pojedyncza skala dla obu.

UCZCIWE OGRANICZENIE -- `scale_resonance()` (TIMDR-M, koincydencja w
przestrzeni skali zamiast czasu -- pomysl uzytkownika, konsultowany i
przetestowany): dziala jako filtr antyszumowy (zmierzono: % szumu tla
przechodzacego prog spadl z 40.6% do 15.6% przy progu ">=3 z 8 zgodnych
skal"), ALE kosztem czulosci na WASKIE struktury -- ten sam test pokazal
spadek pokrycia cienkiej linii (2px) z 42.8% do 31.2%, podczas gdy gruba
linia (14px) zostala nietknieta (93.8% w obu przypadkach). Proba naprawy
(gestsze probkowanie malych sigma) NIE dala mierzalnej roznicy w tym
tescie. Miekkie wazenie (Vfinal=Vmax*(0.5+0.5*RM) zamiast twardego progu)
z definicji nie moze dac WYZSZEGO pokrycia niz Vmax (mnoznik <=1) --
porownywanie obu przy tym samym progu bezwzglednym jest metodologicznie
niesprawiedliwe (mierzy tylko efekt przeskalowania, nie realna wartosc
rezonansu); uczciwa ocena wymaga dopasowanych punktow pracy (ROC), co
NIE zostalo jeszcze zrobione. Dlatego `scale_resonance()` jest tu
dostepna, ale oznaczona jako EKSPERYMENTALNA -- do dalszego strojenia
na realnych danych (np. retina_vessel_orientation.py), nie do uzycia
jako gotowy, domyslny filtr.
"""

import numpy as np
import cv2


def _hessian_eigs(gray, sigma):
    """Wartosci wlasne Hesjanu (Ixx, Ixy, Iyy) po wygladzeniu Gaussem
    (sigma) i normalizacji Lindeberga (mnozenie przez sigma^2, rzad
    gamma=1 dla pochodnej 2. rzedu) -- BEZ tej normalizacji odpowiedzi
    z roznych skal nie sa ze soba porownywalne (patrz naglowek modulu).
    Zwraca (lambda1, lambda2) z |lambda1| <= |lambda2|."""
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


def multi_scale_vesselness(gray, sigmas, beta=0.5, return_stack=False):
    """
    Vesselness Frangiego (bright ridge na ciemnym tle), na kilku skalach
    naraz, z poprawnie policzona (RAZ, globalnie) stala normalizujaca
    `c` -- patrz pkt. 2 w naglowku modulu, to jest KRYTYCZNE dla
    poprawnosci multi-scale, nie kosmetyczny szczegol.

    sigmas: iterowalna kolekcja sigma do przetestowania. Dobierz zakres
    tak, zeby obejmowal ~polowe oczekiwanej szerokosci najwezszej i
    najszerszej struktury, ktora chcesz wykryc (np. dla naczyn
    siatkowki o szerokosci 2-14px: sigmas ~ [1, 1.5, 2, 3, 4, 6, 8]).

    return_stack: jesli True, zwraca tez slownik {sigma: vesselness} --
    potrzebny dla scale_resonance().

    Zwraca (vmax, stack) gdzie vmax = max(vesselness) per piksel po
    wszystkich sigma. Jesli return_stack=False, stack=None.
    """
    sigmas = list(sigmas)
    all_l1, all_l2, all_S = {}, {}, {}
    for s in sigmas:
        l1, l2 = _hessian_eigs(gray, s)
        all_l1[s], all_l2[s] = l1, l2
        all_S[s] = np.sqrt(l1 ** 2 + l2 ** 2)

    global_s_max = max(S.max() for S in all_S.values())
    c = 0.5 * global_s_max if global_s_max > 1e-9 else 1.0

    stack = {}
    for s in sigmas:
        l1, l2, S = all_l1[s], all_l2[s], all_S[s]
        Rb = l1 / (l2 + 1e-12)
        v = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1 - np.exp(-(S ** 2) / (2 * c ** 2)))
        v[l2 > 0] = 0.0  # tylko jasne grzbiety na ciemnym tle
        stack[s] = v

    vmax = np.maximum.reduce(list(stack.values()))
    return vmax, (stack if return_stack else None)


def scale_resonance(stack, threshold=0.10):
    """
    EKSPERYMENTALNE -- patrz "UCZCIWE OGRANICZENIE" w naglowku modulu.

    TIMDR-M rezonans przeniesiony z czasu na przestrzen skali:
    RM(x,y) = odsetek skal, dla ktorych vesselness(x,y,sigma) > threshold.
    Wysoki RM = wiele sasiednich skal "widzi" te sama strukture (typowe
    dla prawdziwej, szerokiej-w-skali struktury); niski RM = tylko
    pojedyncza skala zareagowala (typowe dla szumu).

    Wymaga `stack` ze `multi_scale_vesselness(..., return_stack=True)`.

    ZMIERZONY KOMPROMIS (patrz naglowek modulu): filtruje ~2.6x wiecej
    szumu (40.6%->15.6% w tescie), ale kosztem pokrycia waskich struktur
    (spadek 42.8%->31.2% na linii 2px w tym samym tescie). Uzywaj jako
    DODATKOWA informacje (np. do sortowania/priorytetyzacji kandydatow),
    nie jako twardy filtr odrzucajacy, dopoki nie skalibrowane na
    realnych danych.
    """
    sigmas = list(stack.keys())
    votes = np.zeros_like(next(iter(stack.values())))
    for s in sigmas:
        votes += (stack[s] > threshold).astype(np.float64)
    return votes / len(sigmas)
