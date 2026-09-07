import numpy as np
from astropy.io import fits
from PIL import Image

from phi_core import gradient_field, coherence_lambda, flow_tau, defects_rho, normalize, phi_composite


def _normalize(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-6)
    return arr


def load_fits(path, ext=0):
    """
    Wczytuje dane FITS (NASA/JWST/HST).
    ext – numer rozszerzenia (czasem dane są w HDU 1 lub 2).
    """
    hdul = fits.open(path)
    data = hdul[ext].data
    hdul.close()
    return _normalize(data)


def fits_to_rgb(data):
    """
    Konwersja danych FITS (1 kanał) do RGB.
    """
    img = (data * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def phi_fits(path, ext=0, mode="phi", strength=1.0):
    """
    Tryby:
        - "phi" – klasyczny filtr φ
        - "lambda" – mapa Λ (koherencja)
        - "tau" – mapa τ (gradient)
        - "rho" – mapa defektów ρ
        - "phi-mix" – φ = Λ + τ – ρ

    Λ/τ/ρ licza sie teraz przez wspolny phi_core.py (patrz BUGFIX w
    phi_core.py -- byly 3 rozjezdzajace sie kopie tej logiki, ta byla
    juz poprawna/kanoniczna, teraz jest tylko re-eksportowana).
    """
    data = load_fits(path, ext=ext)
    rgb = fits_to_rgb(data)
    gray = data

    mag, nx, ny = gradient_field(gray)
    Lambda = normalize(coherence_lambda(nx, ny))
    Tau = flow_tau(mag)
    Rho = defects_rho(mag)

    if mode == "lambda":
        out = Lambda
    elif mode == "tau":
        out = Tau
    elif mode == "rho":
        out = Rho
    else:  # "phi" / "phi-mix"
        phi = phi_composite(Lambda, Tau, Rho)
        out = np.clip(phi, 0, 1) ** max(strength, 1e-6)

    out_rgb = (out * 255).astype(np.uint8)
    return Image.fromarray(out_rgb)
