import numpy as np
from PIL import Image
import cv2

from phi_core import gradient_field, coherence_lambda, flow_tau, defects_rho, normalize


def phi_structure_map(img_path):
    """
    Zwraca trzy mapy:
    - Λ: stabilne struktury (koherencja kierunku)
    - τ: przejścia fazowe (gradient)
    - ρ: defekty (lokalne minima/załamania)

    BUGFIX (2026-09-05): Λ liczyła się tu wcześniej z SUROWEGO,
    nieznormalizowanego gradientu (gx, gy) -- w praktyce to była
    wygładzona SIŁA gradientu, nie koherencja KIERUNKU, mimo
    identycznej nazwy/opisu co w phi_filter_v2.py i phi_fits.py, które
    poprawnie normalizowały kierunek przed uśrednieniem. Teraz wszystkie
    trzy pliki wołają tę samą, kanoniczną funkcję z phi_core.py -- patrz
    tam pełny opis błędu.
    """
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    mag, nx, ny = gradient_field(gray)

    Lambda = normalize(coherence_lambda(nx, ny))
    Tau = flow_tau(mag)
    Rho = defects_rho(mag)

    return Lambda, Tau, Rho
