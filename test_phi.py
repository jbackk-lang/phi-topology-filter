"""
test_phi.py -- podstawowe testy sanity dla phi-topology-filter.

Repo nie mialo wczesniej ZADNYCH testow. Testy dzialaja na syntetycznych
obrazach (numpy), nie wymagaja zadnych plikow z examples/.

Wymagania: numpy, opencv-python, Pillow (jak w requirements.txt).
Uruchomienie: python3 test_phi.py   (z poziomu tego folderu)
"""

import numpy as np

from phi_core import (
    gradient_field,
    coherence_lambda,
    flow_tau,
    defects_rho,
    phi_composite,
)


def test_plaski_obraz():
    """Plaski obraz (brak krawedzi) -> gradient dokladnie ~0 wszedzie
    (Sobel stalego pola daje 0 z definicji, nie tylko w przyblizeniu)."""
    gray = np.full((50, 50), 128, dtype=np.float32)
    mag, nx, ny = gradient_field(gray)
    print(f"Test 1 (plaski obraz): max(mag)={mag.max():.6f} (oczekiwano ~1e-6)")
    assert np.allclose(mag, 1e-6, atol=1e-4), "plaski obraz powinien miec gradient ~0"
    print("  -> OK\n")


def test_krawedz_pionowa_pik_tau():
    """Ostra pionowa krawedz -> Tau (magnitude gradientu) ma wyrazny
    pik dokladnie w kolumnie krawedzi -- pilnuje, ze 'przejscie fazowe'
    faktycznie lokalizuje GDZIE jest krawedz, nie tylko ZE jest."""
    gray = np.zeros((60, 60), dtype=np.float32)
    gray[:, 30:] = 255.0
    mag, nx, ny = gradient_field(gray)
    tau = flow_tau(mag)

    col_energy = tau.sum(axis=0)
    pik = int(np.argmax(col_energy))
    print(f"Test 2 (pik Tau): kolumna {pik} (oczekiwano ok. 30)")
    assert 27 <= pik <= 33, "pik Tau powinien byc w miejscu krawedzi"
    print("  -> OK\n")


def test_lambda_ograniczona_0_1():
    """coherence_lambda usrednia wektory JEDNOSTKOWE -> z nierownosci
    trojkata wynik nie moze przekroczyc 1.0, dla ZADNEGO wejscia. To
    test wlasnosci matematycznej, nie tylko obserwacji na przykladzie --
    pilnuje, zeby nikt nie wrocil do liczenia koherencji z surowego
    (nieznormalizowanego) gradientu, dokladnie tak jak bylo wczesniej
    w phi_map.py (patrz BUGFIX w phi_core.py)."""
    rng = np.random.default_rng(7)
    gray = rng.uniform(0, 255, size=(80, 80)).astype(np.float32)
    mag, nx, ny = gradient_field(gray)
    lam = coherence_lambda(nx, ny)
    print(f"Test 3 (Lambda ograniczona): max={lam.max():.4f}, min={lam.min():.4f}")
    assert lam.max() <= 1.0 + 1e-4, "Lambda z definicji nie moze przekroczyc 1.0"
    assert lam.min() >= 0.0 - 1e-4
    print("  -> OK\n")


def test_lambda_rampa_vs_szum():
    """Idealna rampa (staly kierunek gradientu wszedzie) -> Lambda
    bliska 1.0 w srodku obrazu (usrednienie identycznych wektorow
    jednostkowych = ten sam wektor, dlugosc dokladnie 1 -- wynik
    matematycznie pewny, nie tylko przyblizony). Czysty szum -> Lambda
    wyraznie nizsza, bo losowe kierunki czesciowo sie znosza."""
    x = np.linspace(0, 255, 80, dtype=np.float32)
    rampa = np.tile(x, (80, 1))
    mag, nx, ny = gradient_field(rampa)
    lam_rampa = coherence_lambda(nx, ny)[10:-10, 10:-10].mean()  # bez marginesu filtra

    rng = np.random.default_rng(1)
    szum = rng.uniform(0, 255, size=(80, 80)).astype(np.float32)
    mag2, nx2, ny2 = gradient_field(szum)
    lam_szum = coherence_lambda(nx2, ny2).mean()

    print(f"Test 4 (rampa vs szum): Lambda(rampa)={lam_rampa:.4f}, Lambda(szum)={lam_szum:.4f}")
    assert lam_rampa > 0.9, "idealna rampa powinna miec koherencje bliska 1.0"
    assert lam_rampa > lam_szum, "rampa powinna miec wyzsza koherencje niz czysty szum"
    print("  -> OK\n")


def test_rho_wykrywa_zalamanie():
    """Pojedyncza silna krawedz w plaskim tle -> rho (lokalne minimum
    magnitude gradientu po rozmyciu) powinno wykryc niezerowa strukture
    w otoczeniu krawedzi (miejsca, gdzie sygnal jest plaski, ale
    sasiedztwo krawedzi 'podnosi' oczekiwana/rozmyta magnitude)."""
    gray = np.zeros((60, 60), dtype=np.float32)
    gray[:, 30:] = 255.0
    mag, nx, ny = gradient_field(gray)
    rho = defects_rho(mag)
    print(f"Test 5 (rho): max(rho)={rho.max():.4f}")
    assert rho.max() > 0, "rho powinno wykryc cos w otoczeniu jedynej krawedzi w obrazie"
    print("  -> OK\n")


def test_phi_composite_w_zakresie():
    """phi = Lambda + Tau - Rho, znormalizowane -> zawsze w [0,1]."""
    gray = np.zeros((40, 40), dtype=np.float32)
    gray[:, 20:] = 200.0
    mag, nx, ny = gradient_field(gray)
    lam = coherence_lambda(nx, ny)
    tau = flow_tau(mag)
    rho = defects_rho(mag)
    phi = phi_composite(lam, tau, rho)
    print(f"Test 6 (phi_composite): min={phi.min():.4f}, max={phi.max():.4f}")
    assert phi.min() >= -1e-4 and phi.max() <= 1.0 + 1e-4
    print("  -> OK\n")


def test_jeden_wspolny_rdzen():
    """Regresja przeciw powrotowi trzech niezaleznych kopii tej samej
    logiki: phi_filter_v2, phi_fits i phi_map musza importowac
    DOKLADNIE ta sama funkcje coherence_lambda z phi_core -- test
    tozsamosci obiektu (is), nie tylko rownosci wynikow. Rownosc
    wynikow przeszlaby nawet gdyby ktos wkleil nowa, niezalezna kopie z
    identycznym kodem dzis -- a dokladnie tak (kopia zamiast wspolnego
    importu) doszlo do rozjazdu, ktory ten refaktor naprawil."""
    import phi_filter_v2
    import phi_fits
    import phi_map
    import phi_core

    assert phi_filter_v2.coherence_lambda is phi_core.coherence_lambda
    assert phi_fits.coherence_lambda is phi_core.coherence_lambda
    assert phi_map.coherence_lambda is phi_core.coherence_lambda
    print("Test 7 (wspolny rdzen): phi_filter_v2/phi_fits/phi_map dziela ta sama coherence_lambda -> OK\n")


if __name__ == "__main__":
    print("=== Test phi-topology-filter ===\n")
    test_plaski_obraz()
    test_krawedz_pionowa_pik_tau()
    test_lambda_ograniczona_0_1()
    test_lambda_rampa_vs_szum()
    test_rho_wykrywa_zalamanie()
    test_phi_composite_w_zakresie()
    test_jeden_wspolny_rdzen()
    print("Wszystkie testy zaliczone -- phi-topology-filter dziala poprawnie.")
