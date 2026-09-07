"""
network_scan_orientation.py -- ta sama koherencja ze structure tensor
(phi_core.py:structure_tensor_coherence()), zastosowana do wykrywania
skanowania portow w ruchu sieciowym, po przetworzeniu zdarzen polaczen
(czas, port) na obraz gestosci (os X = czas, os Y = port).

Ugruntowanie w literaturze (nie wymyslone na poczekaniu):
- "Interactive Visualization for Network and Port Scan Detection"
  (Muelder/Ma i in.) -- wolne skany portow ukladaja sie po PRZEKATNEJ
  na wykresie (czas, port), szybkie skany jako PIONOWA linia.
- Patenty USPTO o "network terrain" (np. US7873046) licza gradient
  wektorowy ruchu sieciowego wprost po to, zeby wzmocnic wzorce ulozone
  wzdluz jednej osi -- koncepcyjnie to samo co structure tensor.

ZBADANE EKSPERYMENTALNIE (2026-09-08, przed napisaniem tego pliku --
patrz `_experiment_network_scan.py`, `_experiment_gradient_operator.py`,
`_experiment_line_vs_dots.py`, `_experiment_angle_and_edges.py` w tym
repo, zostawione jako dokumentacja procesu):

1. Sam algorytm structure tensor NIE ma bledu kierunku: na ciaglej
   linii wynik jest dokladny co do stopnia dla katow 0-165 stopni
   (blad <=1.9 stopnia, wylacznie z rasteryzacji linii na siatce
   pikseli), niezaleznie od operatora gradientu (Sobel3/Sobel5/Scharr
   dawaly praktycznie ten sam wynik) i niezaleznie od bliskosci brzegu
   obrazu.
2. Renderowanie zdarzen jako POJEDYNCZYCH, IZOLOWANYCH kropek (bez
   laczenia sasiednich w czasie) dawalo bledny kierunek (~19 stopni
   bledu na przekatnym skanie) -- bo kazda kropka z osobna ma wlasny,
   w przyblizeniu izotropowy gradient, niezwiazany z kierunkiem calej
   linii skanu; potrzeba kilku kropek zlanych w ciagly slad, zeby
   kierunek mial sens.
3. Sygnal KOHERENCJI (nie kierunku) byl solidny w obu wariantach: sam
   normalny ruch w danym miejscu obrazu ~0.09-0.12, z obecnoscia skanu
   w tym samym miejscu ~0.35-0.46 (wzrost 2.9x-5.2x) -- to jest glowny,
   wiarygodny sygnal detekcyjny.

Naprawka zastosowana ponizej: `build_connection_trace()` LACZY
zdarzenia sasiadujace W CZASIE odcinkiem linii (jesli odstep czasu
<= max_gap), zamiast rysowac izolowane kropki -- odtwarza to ciagly
slad, tak jak w zwalidowanym tescie (pkt 1 powyzej). Normalny ruch
(losowe, niepowiazane porty) polaczony w ten sam sposob NIE tworzy
spojnego kierunku (kolejne odcinki wskazuja w przypadkowych
kierunkach, usredniaja sie do niskiej koherencji) -- tylko prawdziwy
skan (port rosnacy/malejacy w miare uplywu czasu) daje wysoka, stabilna
koherencje wzdluz spojnego kierunku.

UCZCIWE OGRANICZENIE: to narzedzie zaklada, ze wejsciowe zdarzenia juz
naleza do JEDNEGO zrodla/potoku (np. jeden adres IP zrodlowy) -- nie
rozdziela samo wielu niezaleznych hostow zmieszanych w jednym
strumieniu. W realnym wdrozeniu trzeba by wczesniej pogrupowac zdarzenia
po IP zrodlowym/docelowym. To narzedzie wspiera wizualna/ilosciowa
ocene, nie jest gotowym systemem IDS.
"""

import numpy as np
import cv2

from phi_core import structure_tensor_coherence, lineament_score, orientation_to_rgb


def build_connection_trace(events, size=200, max_gap=None, thickness=1):
    """
    events: iterowalna kolekcja (timestamp, port) -- surowe wartosci,
    dowolna skala (nie musza byc w [0, size)).

    max_gap: maksymalny odstep czasu (w jednostkach WEJSCIOWYCH, nie w
    pikselach) miedzy kolejnymi (posortowanymi czasowo) zdarzeniami, dla
    ktorego sa one rysowane jako POLACZONY odcinek linii, a nie osobne
    kropki -- patrz uzasadnienie w naglowku modulu. Domyslnie (None):
    5% calego zakresu czasu wejscia; dostroic per realne zrodlo danych
    (np. do typowego odstepu miedzy kolejnymi probami tego samego
    hosta).

    Zwraca obraz gestosci polaczen (size x size, os X=czas, os Y=port).
    """
    events = sorted(events, key=lambda e: e[0])
    img = np.zeros((size, size), dtype=np.float32)
    if len(events) == 0:
        return img

    ts = np.array([e[0] for e in events], dtype=np.float64)
    ports = np.array([e[1] for e in events], dtype=np.float64)

    t_min, t_max = ts.min(), ts.max()
    p_min, p_max = ports.min(), ports.max()
    t_span = max(t_max - t_min, 1e-9)
    p_span = max(p_max - p_min, 1e-9)

    if max_gap is None:
        max_gap = 0.05 * t_span

    def to_px(t, p):
        x = (t - t_min) / t_span * (size - 1)
        y = (p - p_min) / p_span * (size - 1)
        return int(round(x)), int(round(y))

    prev_px, prev_t = None, None
    for t, p in zip(ts, ports):
        px = to_px(t, p)
        if prev_px is not None and (t - prev_t) <= max_gap:
            cv2.line(img, prev_px, px, color=60.0, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img, px, radius=max(1, thickness), color=60.0, thickness=-1)
        prev_px, prev_t = px, t

    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _interpret_direction(deg):
    """Zgrubna, opisowa interpretacja zmierzonego kierunku linii --
    tylko pomoc przy czytaniu wyniku, nie twardy klasyfikator."""
    if deg is None:
        return "brak wystarczajaco silnego wzorca do oceny kierunku"
    a = abs(deg)
    if a > 70:
        return "kierunek bliski PIONU -- typowe dla SZYBKIEGO skanu (wiele portow w krotkim czasie)"
    if a < 20:
        return "kierunek bliski POZIOMU -- ten sam port/podobna aktywnosc rozciagnieta w czasie"
    return "kierunek UKOSNY -- typowe dla WOLNEGO, metodycznego skanu portow"


def analyze_network_traffic(image_or_events, size=200, grad_ksize=3, window_sigma=2.5,
                             scan_threshold=0.30, max_gap=None):
    """
    Przyjmuje albo gotowy obraz gestosci (np. z build_connection_trace,
    albo wlasny), albo liste zdarzen (t, port) -- wtedy sam buduje
    obraz przez build_connection_trace().
    """
    if isinstance(image_or_events, np.ndarray):
        img = image_or_events.astype(np.float32)
    else:
        img = build_connection_trace(image_or_events, size=size, max_gap=max_gap)

    coherence, orientation, mag = structure_tensor_coherence(
        img, grad_ksize=grad_ksize, window_sigma=window_sigma
    )
    score = lineament_score(coherence, mag)
    orientation_rgb = orientation_to_rgb(orientation, coherence, mag)

    scan_mask = score > scan_threshold
    mag_norm = mag / (mag.max() + 1e-9)
    weight = scan_mask & (mag_norm > 0.1)

    line_direction_deg = None
    mean_scan_coherence = 0.0
    if np.any(weight):
        mean_scan_coherence = float(coherence[weight].mean())
        ang = orientation[weight]
        mean_grad_ang = 0.5 * np.arctan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())
        mean_line_ang = mean_grad_ang + np.pi / 2
        deg = np.degrees(mean_line_ang)
        line_direction_deg = float(((deg + 90) % 180) - 90)

    return {
        "coherence": (np.clip(coherence, 0, 1) * 255).astype(np.uint8),
        "orientation_rgb": orientation_rgb,
        "score": score,
        "scan_mask": scan_mask,
        "mean_scan_coherence": mean_scan_coherence,
        "line_direction_deg": line_direction_deg,
        "interpretation": _interpret_direction(line_direction_deg),
        "raw_image": img,
    }


def _synthetic_normal_traffic(n=250, size=200, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, size, n)
    port = rng.uniform(0, size, n)
    return list(zip(t, port))


def _synthetic_slow_scan(n=60, size=200, seed=1, jitter=1.5):
    rng = np.random.default_rng(seed)
    t = np.linspace(10, size - 10, n)
    port = np.linspace(10, size - 10, n) + rng.normal(0, jitter, n)
    return list(zip(t, port))


if __name__ == "__main__":
    size = 200
    print("Brak podanych zdarzen -- generuje SYNTETYCZNY ruch: normalny ruch")
    print("+ nalozony wolny skan portow po przekatnej (demo, NIE prawdziwe dane).")

    events = _synthetic_normal_traffic(size=size) + _synthetic_slow_scan(size=size)
    out = analyze_network_traffic(events, size=size)

    from PIL import Image
    stem = "synthetic_network_scan"

    raw_vis = np.clip(out["raw_image"] / (out["raw_image"].max() + 1e-9) * 255, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(raw_vis, cv2.COLOR_GRAY2RGB)
    overlay[out["scan_mask"]] = [255, 40, 40]

    Image.fromarray(out["coherence"]).save(f"{stem}_COHERENCE.png")
    Image.fromarray(out["orientation_rgb"]).save(f"{stem}_ORIENTATION.png")
    Image.fromarray(overlay).save(f"{stem}_SCAN_CANDIDATES.png")

    print("Zapisano:")
    print(f"  {stem}_COHERENCE.png       -- mapa koherencji (0=przypadkowy ruch, 1=spojny slad)")
    print(f"  {stem}_ORIENTATION.png     -- kolor=kierunek, jasnosc=sila sygnalu")
    print(f"  {stem}_SCAN_CANDIDATES.png -- oryginal z zaznaczonymi kandydatami na skan (czerwony)")
    print(f"  srednia koherencja w regionie skanu: {out['mean_scan_coherence']:.3f}")
    print(f"  zmierzony kierunek linii: {out['line_direction_deg']}")
    print(f"  interpretacja: {out['interpretation']}")
