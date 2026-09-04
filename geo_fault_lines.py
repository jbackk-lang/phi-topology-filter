"""
geo_fault_lines.py -- wlasciwa (rygorystyczna) koherencja ze structure
tensor, zastosowana do wykrywania linijnych struktur (lineamentow) w
zdjeciach satelitarnych / DEM: uskokow geologicznych, rzek, drog.

Roznica wzgledem phi_core.py:
- coherence_lambda() w phi_core.py usrednia ZNORMALIZOWANE wektory
  gradientu w oknie -- szybka aproksymacja, dobra do ogolnej
  wizualizacji struktury w phi-topology-filter.
- Ten plik liczy PRAWDZIWY structure tensor (macierz momentow
  drugiego rzedu gradientu: Jxx, Jyy, Jxy), z ktorego koherencja i
  KIERUNEK liczone sa z wartosci wlasnych -- to jest metoda faktycznie
  uzywana w geologii/teledetekcji do ekstrakcji lineamentow (structure
  tensor / orientation coherence, klasyczna technika z analizy
  tekstury i fingerprint recognition).

Wyjscie funkcji structure_tensor_coherence():
- coherence:   [0,1], 0 = izotropowe (brak dominujacego kierunku),
               1 = idealnie liniowe (jeden, spojny kierunek)
- orientation: kat lokalnego kierunku struktury, w radianach
               [-pi/2, pi/2]
- mag:         magnitude gradientu (surowa, nieznormalizowana)

BUGFIX odkryty przy pisaniu tego pliku (2026-09-05): pierwsza wersja
liczyla `disc = sqrt(diff^2 + 4*Jxy^2) + eps` (epsilon DODANY do
licznika) -- w plaskich/bez-gradientowych obszarach (Jxx=Jyy=Jxy=0)
dawalo to coherence = eps/(0+eps) = 1.0, czyli "idealna koherencja"
tam, gdzie w ogole nie ma zadnej struktury. Naprawione: epsilon jest
TYLKO w mianowniku (`trace + eps`), licznik zostaje czystym `disc` bez
dodatku -- wtedy plaski obszar (disc=0, trace=0) daje poprawnie
coherence=0, nie 1. Sprawdzone recznie na trzech przypadkach: plaski
obszar -> 0, czysty pojedynczy gradient (rampa) -> 1, izotropowy szum
-> ~0 (bo diff i Jxy usredniaja sie do ~0, trace zostaje dodatnie).
"""

import numpy as np
import cv2
from PIL import Image


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return img.astype(np.float32)


def structure_tensor_coherence(gray, grad_ksize=3, window_sigma=4.0):
    """
    Liczy prawdziwy structure tensor i zwraca (coherence, orientation, mag).

    grad_ksize   -- rozmiar kernela Sobela do liczenia gradientu
    window_sigma -- sigma rozmycia Gaussa uzywanego do "zbierania"
                    momentow Jxx/Jyy/Jxy z sasiedztwa (wiekszy =
                    gladsza, bardziej regionalna ocena kierunku)
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=grad_ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=grad_ksize)

    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), window_sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), window_sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), window_sigma)

    trace = Jxx + Jyy
    diff = Jxx - Jyy
    disc = np.sqrt(diff * diff + 4 * Jxy * Jxy)   # BEZ epsilon -- patrz BUGFIX w naglowku

    coherence = disc / (trace + 1e-9)              # epsilon TYLKO w mianowniku
    coherence = np.clip(coherence, 0.0, 1.0)

    orientation = 0.5 * np.arctan2(2 * Jxy, diff)  # [-pi/2, pi/2]

    mag = np.sqrt(gx * gx + gy * gy)

    return coherence, orientation, mag


def lineament_score(coherence, mag):
    """Wysoki tylko tam, gdzie jest JEDNOCZESNIE silna krawedz I spojny
    kierunek -- kandydat na lineament, nie tylko teksture."""
    mag_norm = mag / (mag.max() + 1e-9)
    return coherence * mag_norm


def orientation_to_rgb(orientation, coherence, mag):
    """
    Standardowa wizualizacja pola orientacji: H = kierunek lineamentu,
    S = koherencja (jak bardzo 'liniowa' jest struktura), V = magnitude
    gradientu (jasnosc = sila krawedzi).
    """
    hue = ((orientation + np.pi / 2) / np.pi * 179).astype(np.uint8)  # OpenCV H w [0,179]
    sat = np.clip(coherence * 255, 0, 255).astype(np.uint8)
    val = np.clip((mag / (mag.max() + 1e-9)) * 255, 0, 255).astype(np.uint8)
    hsv = cv2.merge([hue, sat, val])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


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
