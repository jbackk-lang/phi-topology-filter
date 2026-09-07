"""
geo_fault_lines.py -- wlasciwa (rygorystyczna) koherencja ze structure
tensor, zastosowana do wykrywania linijnych struktur (lineamentow) w
zdjeciach satelitarnych / DEM: uskokow geologicznych, rzek, drog.

KONSOLIDACJA (2026-09-08): `structure_tensor_coherence()` /
`lineament_score()` / `orientation_to_rgb()` przeniesione do
phi_core.py (byla to jedyna kopia tej matematyki, wiec przeniesienie
nie naprawia rozjazdu -- to ten sam porzadek co reszta modulow tego
repo, ktore importuja wspolny rdzen z phi_core zamiast trzymac wlasna
kopie). Ten plik zostaje CIENKI: importuje gotowe funkcje i uzywa ich
do konkretnego zastosowania (lineamenty geologiczne + demo). Pelny
docstring matematyki/bugfixa -- patrz phi_core.py:structure_tensor_coherence.

Roznica wzgledem coherence_lambda() (phi_core.py): to jest szybka
aproksymacja (usrednienie znormalizowanych wektorow gradientu w oknie),
dobra do ogolnej wizualizacji struktury. `structure_tensor_coherence()`
liczy PRAWDZIWY structure tensor (macierz momentow drugiego rzedu
gradientu: Jxx, Jyy, Jxy), z ktorego koherencja i KIERUNEK liczone sa z
wartosci wlasnych -- to jest metoda faktycznie uzywana w
geologii/teledetekcji do ekstrakcji lineamentow (structure tensor /
orientation coherence, klasyczna technika z analizy tekstury i
fingerprint recognition).
"""

import numpy as np
import cv2
from PIL import Image

from phi_core import structure_tensor_coherence, lineament_score, orientation_to_rgb


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return img.astype(np.float32)


def detect_lineaments(image_path, grad_ksize=3, window_sigma=4.0, score_threshold=0.35):
    img = np.array(Image.open(image_path).convert("RGB"))
    gray = _to_gray(img)

    coherence, orientation, mag = structure_tensor_coherence(
        gray, grad_ksize=grad_ksize, window_sigma=window_sigma
    )
    score = lineament_score(coherence, mag)

    orientation_rgb = orientation_to_rgb(orientation, coherence, mag)

    overlay = img.copy()
    overlay[score > score_threshold] = [255, 40, 40]

    coherence_img = (np.clip(coherence, 0, 1) * 255).astype(np.uint8)

    return {
        "coherence": coherence_img,
        "orientation_rgb": orientation_rgb,
        "score": score,
        "overlay": overlay,
    }


def _synthetic_fault_terrain(size=400, seed=0, angle_deg=30.0, step=25.0):
    """
    Syntetyczny teren z widocznym uskokiem: szum bazowy (teren) + ostre
    przesuniecie wysokosci wzdluz jednej linii pod katem -- imituje scarp
    uskokowy. UZYWANE TYLKO gdy nie podano prawdziwego zdjecia -- jawnie
    oznaczone jako syntetyczne, NIE prawdziwe dane satelitarne/DEM.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size), dtype=np.float32)
    for octave, amp in [(4, 40), (8, 20), (16, 10), (32, 5)]:
        noise = rng.normal(0, 1, (octave, octave)).astype(np.float32)
        noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        base += amp * noise

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = np.deg2rad(angle_deg)
    d = xx * np.cos(angle) + yy * np.sin(angle) - size * 0.5
    fault = np.where(d > 0, step, -step)

    terrain = base + fault
    terrain = terrain - terrain.min()
    terrain = terrain / (terrain.max() + 1e-9) * 255.0
    return terrain.astype(np.uint8)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analizuję prawdziwy plik: {path}")
    else:
        print("Brak podanego pliku -- generuję SYNTETYCZNY teren z uskokiem (demo,")
        print("NIE prawdziwe dane satelitarne).")
        synth = _synthetic_fault_terrain()
        path = "synthetic_fault_terrain.png"
        Image.fromarray(synth).convert("RGB").save(path)
        print(f"Zapisano syntetyczny teren demonstracyjny: {path}")

    out = detect_lineaments(path)
    stem = os.path.splitext(path)[0]

    Image.fromarray(out["coherence"]).save(f"{stem}_COHERENCE.png")
    Image.fromarray(out["orientation_rgb"]).save(f"{stem}_ORIENTATION.png")
    Image.fromarray(out["overlay"]).save(f"{stem}_LINEAMENTS.png")

    print("Zapisano:")
    print(f"  {stem}_COHERENCE.png   -- mapa koherencji (0=izotropowe, 1=liniowe)")
    print(f"  {stem}_ORIENTATION.png -- kolor=kierunek, jasność=siła krawędzi")
    print(f"  {stem}_LINEAMENTS.png  -- oryginał z zaznaczonymi kandydatami na uskoki (na czerwono)")
