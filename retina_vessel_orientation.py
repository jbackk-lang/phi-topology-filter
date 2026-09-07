"""
retina_vessel_orientation.py -- ta sama koherencja ze structure tensor
(phi_core.py:structure_tensor_coherence), zastosowana do map orientacji
naczyn siatkowki (retinal fundus imaging).

Ten sam wzorzec co geo_fault_lines.py / fingerprint_ridge_orientation.py
-- inna domena, ta sama matematyka. Roznice specyficzne dla siatkowki:

- Naczynia sa GRUBSZE i bardziej TORTUOUS (kreta) niz linie papilarne,
  wiec `window_sigma` jest wieksze niz w fingerprint_ridge_orientation.py
  (3.5 vs 2.5), ale mniejsze niz przy lineamentach geologicznych (4.0),
  bo naczynia wciaz sa waskie wzgledem calego pola widzenia.
- Prawdziwe zdjecia dna oka (fundus) maja okragle pole widzenia (FOV) na
  czarnym tle -- `_fov_mask()` odtwarza to i wyklucza obszar poza FOV z
  wyniku (poza FOV nie ma zadnej struktury, tylko krawedz maski, ktora
  bez tego zostalaby falszywie wykryta jako "idealnie koherentna linia").
- `vessel_score` (alias lineament_score) jest tu uzywany jako przyblizenie
  "vesselness" -- w realnych pipeline'ach do tego zadania czesciej uzywa
  sie filtra Frangiego (dedykowanego pod tubularne struktury), ale
  orientacja z structure tensor jest legitymnie uzywana rownolegle, np.
  do sledzenia lokalnego przebiegu naczynia (tracking wzdluz kierunku)
  albo klasyfikacji tetnica/zyla po ciaglosci orientacji.
"""

import numpy as np
import cv2
from PIL import Image

from phi_core import structure_tensor_coherence, lineament_score, orientation_to_rgb


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return img.astype(np.float32)


def _fov_mask(shape, margin=0.02):
    """Okragla maska pola widzenia jak w prawdziwym zdjeciu dna oka --
    reszta obrazu (poza kolem) jest czarna i NIE powinna wchodzic do
    wyniku (krawedz maski wyglada jak idealnie koherentna linia,
    a to artefakt, nie naczynie)."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) / 2.0 * (1.0 - margin)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def analyze_retina(image_path, grad_ksize=3, window_sigma=3.5, vessel_threshold=0.30):
    img = np.array(Image.open(image_path).convert("RGB"))
    gray = _to_gray(img)
    fov = _fov_mask(gray.shape)

    coherence, orientation, mag = structure_tensor_coherence(
        gray, grad_ksize=grad_ksize, window_sigma=window_sigma
    )
    vessel_score = lineament_score(coherence, mag)
    vessel_score[~fov] = 0.0
    coherence = np.where(fov, coherence, 0.0)

    orientation_rgb = orientation_to_rgb(orientation, coherence, mag)
    orientation_rgb[~fov] = 0

    overlay = img.copy()
    overlay[~fov] = 0
    overlay[fov & (vessel_score > vessel_threshold)] = [255, 40, 40]

    coherence_img = (np.clip(coherence, 0, 1) * 255).astype(np.uint8)

    return {
        "coherence": coherence_img,
        "orientation_rgb": orientation_rgb,
        "vessel_score": vessel_score,
        "overlay": overlay,
        "mean_vessel_coherence": float(coherence[fov & (vessel_score > vessel_threshold)].mean())
        if np.any(fov & (vessel_score > vessel_threshold)) else 0.0,
    }


def _branch(img, x, y, angle, length, thickness, depth, rng, segments=5):
    """Rekurencyjnie rysuje jedno drzewo naczyniowe: kazda galaz to
    lekko kreta polilinia (drift kata przy kazdym segmencie), na koncu
    z pewnym prawdopodobienstwem dzieli sie na 2 cieńsze, krotsze
    galezie pod losowym katem -- imituje bifurkacje naczyn siatkowki."""
    if depth <= 0 or thickness < 1.0 or length < 8:
        return

    seg_len = length / segments
    cur_x, cur_y, cur_angle = x, y, angle
    for _ in range(segments):
        cur_angle += rng.normal(0, 0.12)
        nx = cur_x + seg_len * np.cos(cur_angle)
        ny = cur_y + seg_len * np.sin(cur_angle)
        cv2.line(img, (int(cur_x), int(cur_y)), (int(nx), int(ny)),
                  color=255, thickness=max(1, int(round(thickness))))
        cur_x, cur_y = nx, ny

    if rng.random() < 0.85:
        for sign in (-1.0, 1.0):
            if rng.random() < 0.75:
                branch_angle = cur_angle + sign * rng.uniform(0.35, 0.9)
                _branch(img, cur_x, cur_y, branch_angle,
                        length * rng.uniform(0.55, 0.75),
                        thickness * rng.uniform(0.55, 0.75),
                        depth - 1, rng, segments=segments)


def _synthetic_retina(size=400, seed=0, n_main_branches=6, return_mask=False):
    """Syntetyczne 'drzewo' naczyniowe od centralnego 'tarczy nerwu
    wzrokowego' (optic disc) w okragłym FOV -- jawnie SYNTETYCZNE, nie
    prawdziwe zdjecie dna oka.

    return_mask=True: dodatkowo zwraca PRAWDZIWA maske naczyn (kanwa
    PRZED rozmyciem/polaczeniem z tlem/dorysowaniem tarczy -- czysty
    ground truth "tu jest narysowane naczynie, tu nie") -- do uzycia w
    testach ROC/precision-recall, gdzie potrzebna jest znana prawda, nie
    tylko przyblizenie pasmem wokol linii."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size), dtype=np.uint8)
    disc_x, disc_y = size * 0.42, size * 0.5

    for i in range(n_main_branches):
        angle = 2 * np.pi * i / n_main_branches + rng.normal(0, 0.15)
        _branch(canvas, disc_x, disc_y, angle,
                length=size * rng.uniform(0.32, 0.42),
                thickness=rng.uniform(3.5, 5.5),
                depth=4, rng=rng)

    vessel_mask = canvas > 0

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    background = np.full((size, size), 60, dtype=np.uint8)
    fundus = np.maximum(background, canvas)

    fov = _fov_mask((size, size))
    fundus[~fov] = 0
    vessel_mask = vessel_mask & fov
    cv2.circle(fundus, (int(disc_x), int(disc_y)), 10, 200, -1)

    if return_mask:
        return fundus, vessel_mask
    return fundus


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analizuję prawdziwy plik: {path}")
    else:
        print("Brak podanego pliku -- generuję SYNTETYCZNE drzewo naczyniowe")
        print("(demo, NIE prawdziwe zdjecie dna oka).")
        synth = _synthetic_retina()
        path = "synthetic_retina.png"
        Image.fromarray(synth).convert("RGB").save(path)
        print(f"Zapisano syntetyczne dno oka demonstracyjne: {path}")

    out = analyze_retina(path)
    stem = os.path.splitext(path)[0]

    Image.fromarray(out["coherence"]).save(f"{stem}_COHERENCE.png")
    Image.fromarray(out["orientation_rgb"]).save(f"{stem}_ORIENTATION.png")
    Image.fromarray(out["overlay"]).save(f"{stem}_VESSELS.png")

    print("Zapisano:")
    print(f"  {stem}_COHERENCE.png   -- mapa koherencji (0=tlo/rozgalezienie, 1=prosty odcinek naczynia)")
    print(f"  {stem}_ORIENTATION.png -- kolor=kierunek naczynia, jasnosc=sila krawedzi")
    print(f"  {stem}_VESSELS.png     -- oryginal z zaznaczonymi kandydatami na naczynia (czerwony)")
    print(f"  srednia koherencja wzdluz wykrytych naczyn: {out['mean_vessel_coherence']:.3f}")
