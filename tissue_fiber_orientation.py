"""
tissue_fiber_orientation.py -- ta sama koherencja ze structure tensor
(phi_core.py:structure_tensor_coherence), zastosowana do map orientacji
wlokien tkanki w mikroskopii (kolagen, wlokna miesniowe, itp.).

Ten sam wzorzec co geo_fault_lines.py / fingerprint_ridge_orientation.py
/ retina_vessel_orientation.py -- inna domena, ta sama matematyka.
Roznica: `window_sigma` jest tu NAJMNIEJSZE ze wszystkich czterech
domen (1.8) -- wlokna pod mikroskopem sa cienkie i gesto upakowane,
wieksze okno zlaloby sasiednie wlokna o roznym kierunku w jedna
usrednioną orientacje.

To jest realna, uznana technika w biomechanice/histologii: "alignment
index" / "coherency" liczony ze structure tensor jest dokladnie tym, co
robia narzedzia typu OrientationJ (wtyczka do Fiji/ImageJ) i CT-FIRE
przy ocenie stopnia uporzadkowania wlokien kolagenu w tkance (np. w
badaniach nowotworow, gdzie wysoce uporzadkowane wlokna kolagenu wokol
guza sa markerem inwazyjnosci) -- `mean_alignment_index` ponizej to
dokladnie ta wielkosc: srednia koherencja w polu widzenia, [0,1], gdzie
wyzsza wartosc = wieksze globalne uporzadkowanie wlokien.
"""

import numpy as np
import cv2
from PIL import Image

from phi_core import structure_tensor_coherence, lineament_score, orientation_to_rgb


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return img.astype(np.float32)


def analyze_tissue_fibers(image_path, grad_ksize=3, window_sigma=1.8, fiber_threshold=0.35):
    img = np.array(Image.open(image_path).convert("RGB"))
    gray = _to_gray(img)

    coherence, orientation, mag = structure_tensor_coherence(
        gray, grad_ksize=grad_ksize, window_sigma=window_sigma
    )
    fiber_score = lineament_score(coherence, mag)
    orientation_rgb = orientation_to_rgb(orientation, coherence, mag)

    overlay = img.copy()
    overlay[fiber_score > fiber_threshold] = [40, 220, 255]

    coherence_img = (np.clip(coherence, 0, 1) * 255).astype(np.uint8)

    # Alignment index (OrientationJ/CT-FIRE-style): srednia koherencja
    # tam, gdzie w ogole jest wlokno (silna krawedz), nie na calym tle.
    mag_norm = mag / (mag.max() + 1e-9)
    fiber_pixels = mag_norm > 0.15
    alignment_index = float(coherence[fiber_pixels].mean()) if np.any(fiber_pixels) else 0.0

    return {
        "coherence": coherence_img,
        "orientation_rgb": orientation_rgb,
        "score": fiber_score,
        "overlay": overlay,
        "mean_alignment_index": alignment_index,
    }


def _synthetic_fiber_tissue(size=400, seed=0, fiber_spacing=6.0, warp_octaves=((6, 14.0), (12, 6.0))):
    """
    Syntetyczna teksura wlokien: rownolegle linie (jak w
    fingerprint_ridge_orientation.py) zaburzone WIELOOKTAWOWYM, gladkim
    polem losowym (nie pojedynczym punktowym 'core' jak przy odcisku
    palca) -- imituje faliste, lokalnie-ale-nie-globalnie uporzadkowane
    wlokna tkanki (np. kolagen wokol regionu o innym uporzadkowaniu).
    Jawnie SYNTETYCZNE -- nie prawdziwy obraz mikroskopowy.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    warp = np.zeros((size, size), dtype=np.float32)
    for octave, amp in warp_octaves:
        noise = rng.normal(0, 1, (octave, octave)).astype(np.float32)
        noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        warp += amp * noise

    phase = (yy + warp) / fiber_spacing * 2 * np.pi
    fibers = 0.5 + 0.5 * np.sin(phase)

    noise = rng.normal(0, 0.04, (size, size)).astype(np.float32)
    img = np.clip(fibers + noise, 0, 1) * 255.0
    return img.astype(np.uint8)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analizuję prawdziwy plik: {path}")
    else:
        print("Brak podanego pliku -- generuję SYNTETYCZNA teksture wlokien tkanki")
        print("(demo, NIE prawdziwy obraz mikroskopowy).")
        synth = _synthetic_fiber_tissue()
        path = "synthetic_tissue_fibers.png"
        Image.fromarray(synth).convert("RGB").save(path)
        print(f"Zapisano syntetyczna teksture demonstracyjna: {path}")

    out = analyze_tissue_fibers(path)
    stem = os.path.splitext(path)[0]

    Image.fromarray(out["coherence"]).save(f"{stem}_COHERENCE.png")
    Image.fromarray(out["orientation_rgb"]).save(f"{stem}_ORIENTATION.png")
    Image.fromarray(out["overlay"]).save(f"{stem}_FIBERS.png")

    print("Zapisano:")
    print(f"  {stem}_COHERENCE.png -- mapa koherencji lokalnej (0=chaos, 1=spojne wlokno)")
    print(f"  {stem}_ORIENTATION.png -- kolor=kierunek wlokna, jasnosc=sila krawedzi")
    print(f"  {stem}_FIBERS.png -- oryginal z zaznaczonymi wloknami o wysokiej koherencji (cyjan)")
    print(f"  alignment index (srednia koherencja wzdluz wlokien): {out['mean_alignment_index']:.3f}")
