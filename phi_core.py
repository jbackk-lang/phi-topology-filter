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
