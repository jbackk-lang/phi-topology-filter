"""
fingerprint_ridge_orientation.py -- ta sama koherencja ze structure
tensor (phi_core.py:structure_tensor_coherence), zastosowana do map
orientacji linii papilarnych (fingerprint ridge orientation).

To jest DOKLADNIE ten sam wzorzec co geo_fault_lines.py -- inna domena,
ta sama matematyka. Jedyna roznica: `window_sigma` dobrany do skali
linii papilarnych (odstep miedzy grzbietami linii przy typowej
rozdzielczosci skanera ~500dpi to ok. 8-10 pikseli -- wiec okno
zbierania momentow gradientu musi byc WYRAZNIE mniejsze niz przy
lineamentach geologicznych, gdzie struktury sa duzo szersze).

Zastosowanie: mapa orientacji grzbietow linii papilarnych jest
standardowym pierwszym krokiem w klasycznych (nie-neuronowych)
pipeline'ach fingerprint recognition (np. przed ekstrakcja minucji,
albo do oceny jakosci odcisku) -- niska koherencja lokalnie wskazuje
na punkty osobliwe (core/delta) albo minucje (rozgalezienia/konce
linii), gdzie kierunek grzbietu nie jest dobrze zdefiniowany.
`fingerprint_quality_map()` ponizej wprost wykorzystuje to: obszary
niskiej koherencji sa oznaczane jako kandydaci na osobliwosci/szum,
nie jako "zle dane" do odrzucenia.
"""

import numpy as np
import cv2
from PIL import Image

from phi_core import structure_tensor_coherence, lineament_score, orientation_to_rgb


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return img.astype(np.float32)


def analyze_fingerprint(image_path, grad_ksize=3, window_sigma=2.5, low_coherence_threshold=0.25):
    """
    window_sigma=2.5 (vs 4.0 domyslne dla geo_fault_lines) -- dobrane do
    typowego odstepu miedzy grzbietami linii papilarnych (~8-10px przy
    500dpi); zbyt duze okno usredni sasiednie grzbiety o roznym
    kierunku (np. w poblizu core/delta) i sztucznie zawyzy koherencje
    tam, gdzie w rzeczywistosci jest osobliwosc.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    gray = _to_gray(img)

    coherence, orientation, mag = structure_tensor_coherence(
        gray, grad_ksize=grad_ksize, window_sigma=window_sigma
    )
    score = lineament_score(coherence, mag)
    orientation_rgb = orientation_to_rgb(orientation, coherence, mag)

    # Kandydaci na osobliwosci (core/delta) / minucje: silna krawedz
    # (jest grzbiet), ale NISKA koherencja kierunku (kierunek niestabilny
    # lokalnie) -- odwrotnosc kryterium lineament_score.
    mag_norm = mag / (mag.max() + 1e-9)
    singularity_candidates = (coherence < low_coherence_threshold) & (mag_norm > 0.15)

    overlay = img.copy()
    overlay[singularity_candidates] = [255, 210, 0]

    coherence_img = (np.clip(coherence, 0, 1) * 255).astype(np.uint8)

    return {
        "coherence": coherence_img,
        "orientation_rgb": orientation_rgb,
        "score": score,
        "singularity_map": singularity_candidates,
        "overlay": overlay,
    }


def _synthetic_fingerprint(size=400, seed=0, ridge_spacing=9.0, core_strength=18.0):
    """
    Syntetyczny wzor grzbietow linii papilarnych: rownolegle faliste
    linie (sinusoida wzdluz osi x) zaburzone lokalnym "core"-podobnym
    zniekształceniem (promieniowe pole przesuniecia wokol jednego
    punktu) -- imituje typowy uklad grzbietow z jedna osobliwoscia.
    Jawnie SYNTETYCZNE -- nie prawdziwy odcisk palca.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size * 0.5, size * 0.55

    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    # Pole przesuniecia fazy grzbietow malejace z odlegloscia od "core"
    warp = core_strength * (dy / r) * np.exp(-r / (size * 0.28))

    phase = (yy + warp) / ridge_spacing * 2 * np.pi
    ridges = 0.5 + 0.5 * np.sin(phase)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.03, (size, size)).astype(np.float32)

    img = np.clip(ridges + noise, 0, 1) * 255.0
    return img.astype(np.uint8)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analizuję prawdziwy plik: {path}")
    else:
        print("Brak podanego pliku -- generuję SYNTETYCZNY wzor linii papilarnych")
        print("(demo, NIE prawdziwy odcisk palca).")
        synth = _synthetic_fingerprint()
        path = "synthetic_fingerprint.png"
        Image.fromarray(synth).convert("RGB").save(path)
        print(f"Zapisano syntetyczny wzor demonstracyjny: {path}")

    out = analyze_fingerprint(path)
    stem = os.path.splitext(path)[0]

    Image.fromarray(out["coherence"]).save(f"{stem}_COHERENCE.png")
    Image.fromarray(out["orientation_rgb"]).save(f"{stem}_ORIENTATION.png")
    Image.fromarray(out["overlay"]).save(f"{stem}_SINGULARITIES.png")

    print("Zapisano:")
    print(f"  {stem}_COHERENCE.png     -- mapa koherencji grzbietow (0=chaos, 1=spojne linie)")
    print(f"  {stem}_ORIENTATION.png   -- kolor=kierunek grzbietu, jasnosc=sila krawedzi")
    print(f"  {stem}_SINGULARITIES.png -- oryginal z zaznaczonymi kandydatami na core/delta/minucje (zolty)")
