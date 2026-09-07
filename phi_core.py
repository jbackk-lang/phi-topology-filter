"""
phi_core.py -- wspolny rdzen matematyczny filtra phi (Lambda-tau-rho).

BUGFIX (2026-09-05): _gradient / _coherence / _rho_defects istnialy
wczesniej jako TRZY niezalezne kopie w phi_filter_v2.py, phi_fits.py i
phi_map.py -- i zdazyly sie rozjechac, mimo identycznych nazw i
docstringow:

  - phi_filter_v2.py i phi_fits.py liczyly Lambda (koherencje kierunku)
    ze ZNORMALIZOWANEGO (jednostkowego) wektora gradientu (nx, ny) --
    wynik zalezy WYLACZNIE od tego, jak zgodne kierunkowo sa lokalne
    gradienty, nie od ich sily.
  - phi_map.py liczyla "Lambda" z SUROWEGO (nieznormalizowanego)
    gradientu (gx, gy) -- co w praktyce jest wygladzona SILA
    gradientu, calkiem inna wielkosc, mimo tej samej nazwy i tego
    samego opisu ("koherencja kierunku") w docstringu.

Ten plik jest teraz JEDYNYM miejscem, gdzie ta logika jest
zdefiniowana -- phi_filter_v2.py, phi_fits.py i phi_map.py importuja
stad, zamiast trzymac wlasne kopie. Kanoniczna definicja Lambda to ta
ZNORMALIZOWANA wersja (v2/fits), bo tylko ona faktycznie mierzy
koherencje KIERUNKU, zgodnie z tym, co obiecuje README i docstringi.

phi_filter.py (wersja 1, starsza, inne API: `_phi_operator` z
wlasnym `strength`-jako-wykladnik) NIE zostal tu wciagniety -- to
osobny, wczesniejszy plik, nie czesc tej konkretnej trzy-kopiowej
duplikacji.

KONSOLIDACJA (2026-09-08): `structure_tensor_coherence()` /
`lineament_score()` / `orientation_to_rgb()` przeniesione tu z
geo_fault_lines.py (byly tam JEDYNA kopia, wiec to nie jest bugfix
rozjazdu jak wyzej -- po prostu ten sam wzorzec porzadkowania: kazda
wersja koherencji ma teraz jedno miejsce definicji). geo_fault_lines.py
importuje je stad. Dwie METODY koherencji nadal wspolistnieja
CELOWO, to NIE duplikat:
  - coherence_lambda() (wyzej) -- usrednia znormalizowane wektory
    gradientu w oknie prostokatnym. Szybsza aproksymacja, uzywana przez
    phi_filter_v2/phi_fits/phi_map do ogolnej wizualizacji struktury.
  - structure_tensor_coherence() (nizej) -- prawdziwy structure tensor
    (wartosci wlasne macierzy momentow Jxx/Jyy/Jxy, rozmytych Gaussem),
    z ktorego liczona jest ZARAZ koherencja I kierunek lokalnej
    struktury. To jest metoda faktycznie uzywana w teledetekcji/analizie
    tekstury (ENVI/SNAP/ArcGIS) do ekstrakcji struktur liniowych --
    uskokow geologicznych, rzek, drog, wlokien tkanki w mikroskopii,
    naczyn siatkowki, grzbietow linii papilarnych. Dokladniejsza, ale
    wolniejsza (rozmycie Gaussa zamiast prostego usredniania w oknie)
    i dodatkowo daje `orientation`, ktorego coherence_lambda() w ogole
    nie liczy.
Zadna domena (geologia, biometria, mikroskopia, ...) nie wymaga wlasnej
kopii tej matematyki -- rozni sie TYLKO preprocessingiem obrazu
wejsciowego (kanal barwny, kontrast, dobor `window_sigma` do skali
struktury), nie samym liczeniem coherence/orientation.
"""

import numpy as np
import cv2


def gradient_field(gray):
    """
    Zwraca (mag, nx, ny) dla obrazu w skali szarosci `gray`:
    - mag: magnitude gradientu (Sobel), zawsze > 0 (+1e-6)
    - nx, ny: znormalizowany (jednostkowy) kierunek gradientu w kazdym
      pikselu -- to jest wejscie do coherence_lambda(), NIE do flow_tau().
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx = gx / mag
    ny = gy / mag
    return mag, nx, ny


def normalize(x):
    """Min-max rozciagniecie do [0,1] (per-obraz, nie globalnie)."""
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def coherence_lambda(nx, ny, size=7):
    """
    Lambda -- lokalna koherencja KIERUNKU gradientu. Usrednia wektory
    JEDNOSTKOWE (nx, ny) w oknie size x size, a nie surowy gradient --
    to jedyny sposob, zeby wynik mierzyl zgodnosc kierunku niezaleznie
    od sily krawedzi (patrz BUGFIX w naglowku pliku). Wartosc z
    definicji miesci sie w [0,1] (usrednienie wektorow jednostkowych
    nie moze przekroczyc dlugosci 1) -- wywolujacy zwykle i tak
    dodatkowo przepuszcza wynik przez normalize() dla kontrastu.
    """
    kernel = np.ones((size, size), np.float32) / (size * size)
    cx = cv2.filter2D(nx, -1, kernel)
    cy = cv2.filter2D(ny, -1, kernel)
    return np.sqrt(cx**2 + cy**2)


def flow_tau(mag):
    """Tau -- znormalizowana magnitude gradientu (przeplyw/przejscie)."""
    return mag / (mag.max() + 1e-6)


def defects_rho(mag, blur_ksize=9):
    """
    Rho -- defekty jako lokalne MINIMA magnitude gradientu (miejsca,
    gdzie krawedz nagle slabnie wzgledem otoczenia). Nie jest to
    dosl0wny curl(grad(I)) z README (ktory byl by tozsamosciowo zero z
    tozsamosci wektorowej ∇x∇f=0) -- to inna, praktyczna heurystyka;
    patrz zastrzezenie w README dopisane rownolegle z ta poprawka.
    """
    blur = cv2.GaussianBlur(mag, (blur_ksize, blur_ksize), 0)
    diff = mag - blur
    rho = np.maximum(0, -diff)
    return normalize(rho)


def phi_composite(Lambda, Tau, Rho):
    """phi = Lambda + Tau - Rho, znormalizowane do [0,1]."""
    phi = Lambda + Tau - Rho
    return normalize(phi)


# ---------------------------------------------------------------------
# Structure tensor: druga, dokladniejsza metoda koherencji + kierunek.
# Przeniesione z geo_fault_lines.py (KONSOLIDACJA, patrz naglowek pliku).
# ---------------------------------------------------------------------

def structure_tensor_coherence(gray, grad_ksize=3, window_sigma=4.0):
    """
    Liczy prawdziwy structure tensor i zwraca (coherence, orientation, mag).

    coherence:   [0,1], 0 = izotropowe (brak dominujacego kierunku),
                 1 = idealnie liniowe (jeden, spojny kierunek)
    orientation: kat lokalnego kierunku struktury, w radianach
                 [-pi/2, pi/2]
    mag:         magnitude gradientu (surowa, nieznormalizowana)

    grad_ksize   -- rozmiar kernela Sobela do liczenia gradientu
    window_sigma -- sigma rozmycia Gaussa uzywanego do "zbierania"
                    momentow Jxx/Jyy/Jxy z sasiedztwa (wiekszy =
                    gladsza, bardziej regionalna ocena kierunku) --
                    to jest jedyny parametr, ktory warto dostroic per
                    domena (np. mniejszy dla cienkich wlokien tkanki
                    pod mikroskopem, wiekszy dla lineamentow satelitarnych).

    BUGFIX (2026-09-05, odkryty w geo_fault_lines.py przy pierwszym
    napisaniu tej funkcji): pierwsza wersja liczyla
    `disc = sqrt(diff^2 + 4*Jxy^2) + eps` (epsilon DODANY do licznika)
    -- w plaskich/bez-gradientowych obszarach (Jxx=Jyy=Jxy=0) dawalo to
    coherence = eps/(0+eps) = 1.0, czyli "idealna koherencja" tam, gdzie
    w ogole nie ma zadnej struktury. Naprawione: epsilon jest TYLKO w
    mianowniku (`trace + eps`), licznik zostaje czystym `disc` bez
    dodatku -- wtedy plaski obszar (disc=0, trace=0) daje poprawnie
    coherence=0, nie 1. Sprawdzone recznie na trzech przypadkach: plaski
    obszar -> 0, czysty pojedynczy gradient (rampa) -> 1, izotropowy szum
    -> ~0 (bo diff i Jxy usredniaja sie do ~0, trace zostaje dodatnie).
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=grad_ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=grad_ksize)

    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), window_sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), window_sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), window_sigma)

    trace = Jxx + Jyy
    diff = Jxx - Jyy
    disc = np.sqrt(diff * diff + 4 * Jxy * Jxy)   # BEZ epsilon -- patrz BUGFIX wyzej

    coherence = disc / (trace + 1e-9)              # epsilon TYLKO w mianowniku
    coherence = np.clip(coherence, 0.0, 1.0)

    orientation = 0.5 * np.arctan2(2 * Jxy, diff)  # [-pi/2, pi/2]

    mag = np.sqrt(gx * gx + gy * gy)

    return coherence, orientation, mag


def lineament_score(coherence, mag):
    """Wysoki tylko tam, gdzie jest JEDNOCZESNIE silna krawedz I spojny
    kierunek -- kandydat na lineament (uskok/naczynie/wlokno/grzbiet),
    nie tylko teksture. Nazwa pochodzi od pierwszego zastosowania
    (geologia), ale sama funkcja jest domenowo-neutralna."""
    mag_norm = mag / (mag.max() + 1e-9)
    return coherence * mag_norm


def orientation_to_rgb(orientation, coherence, mag):
    """
    Standardowa wizualizacja pola orientacji: H = kierunek struktury,
    S = koherencja (jak bardzo 'liniowa' jest struktura), V = magnitude
    gradientu (jasnosc = sila krawedzi).
    """
    hue = ((orientation + np.pi / 2) / np.pi * 179).astype(np.uint8)  # OpenCV H w [0,179]
    sat = np.clip(coherence * 255, 0, 255).astype(np.uint8)
    val = np.clip((mag / (mag.max() + 1e-9)) * 255, 0, 255).astype(np.uint8)
    hsv = cv2.merge([hue, sat, val])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
