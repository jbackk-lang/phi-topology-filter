import numpy as np
from PIL import Image
import cv2

from phi_core import gradient_field, coherence_lambda, flow_tau, defects_rho, normalize, phi_composite


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_rgb(arr):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def phi_filter_v2(path, mode="phi-mix", strength=1.0):
    """
    Wersja 2 filtra phi -- Lambda/Tau/Rho licza sie teraz przez wspolny
    phi_core.py (patrz BUGFIX w phi_core.py: byly 3 rozjezdzajace sie
    kopie tej logiki).

    mode:
        - "lambda" / "tau" / "rho" -- pojedyncza mapa (jak w phi_fits.py)
        - "phi" / "phi-mix" (domyslnie) -- pelny kompozyt Lambda+Tau-Rho

    BUGFIX (2026-09-05), dwie rzeczy:
    - `mode` wczesniej NIE ISTNIAL w sygnaturze tej funkcji, mimo ze
      phi_batch.py wolal ja z parametrem `mode` uzywanym do nazwy pliku
      wyjsciowego -- wiec np. "zdjecie_lambda.jpg" zawieralo w
      rzeczywistosci pelny kompozyt phi, nie mape Lambda. Teraz `mode`
      dziala tak samo dla obrazow RGB jak juz dzialalo dla FITS.
    - `strength` bylo w sygnaturze, ale nigdzie w ciele funkcji nie
      bylo uzywane -- nie mialo zadnego efektu niezaleznie od wartosci.
      Teraz dziala jako wykladnik na koncowym kompozycie phi (analogicznie
      do tego, jak `strength` byl uzyty w phi_filter.py v1), TYLKO dla
      trybu phi/phi-mix -- pojedyncze mapy lambda/tau/rho zostaja
      surowymi diagnostykami, bez wzmocnienia.
    """
    img = _load_rgb(path)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mag, nx, ny = gradient_field(gray)

    Lambda = normalize(coherence_lambda(nx, ny))
    Tau = flow_tau(mag)
    Rho = defects_rho(mag)

    if mode == "lambda":
        out_map = Lambda
    elif mode == "tau":
        out_map = Tau
    elif mode == "rho":
        out_map = Rho
    else:  # "phi" / "phi-mix"
        phi = phi_composite(Lambda, Tau, Rho)
        out_map = np.clip(phi, 0, 1) ** max(strength, 1e-6)

    out_map3 = np.stack([out_map] * 3, axis=-1)
    out = img * (0.4 + 0.6 * out_map3)

    return _save_rgb(out)
