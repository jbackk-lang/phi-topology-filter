## 🔗 Wszystkie modele i repozytoria
Pełna lista projektów znajduje się na stronie:
https://jbackk-lang.github.io
---

# phi-topology-filter  
Topologiczny filtr **φ** oparty na geometrii **Λ–τ–ρ** do analizy obrazów kosmicznych (FITS/JPG/PNG).

Tak — to jest realna matematyka na pikselach, nie coś udawanego: Λ liczy się z rzeczywistej koherencji kierunku gradientu, τ z magnitude gradientu, ρ z lokalnych minimów po rozmyciu Gaussa. Na zdjęciach o dużej ilości struktury (mgławice, galaktyki, cokolwiek z wieloma krawędziami) to naturalnie wygląda efektownie, bo tam jest dużo materiału do wyłapania.

---

## 🔥 Co to jest filtr φ?

Zwykłe filtry pokazują **kontrast**.  
Filtr φ pokazuje **strukturę przestrzeni**:

- **Λ** – stabilne struktury (kolumny, włókna, rdzenie),
- **τ** – przepływy i przejścia fazowe (gradienty, halo),
- **ρ** – defekty (miejsca narodzin materii, załamania skrętu),
- **φ** – operator równowagi: φ = Λ + τ – ρ.

Celem jest uzyskanie wizualizacji, która pokazuje **stadium materii i przestrzeni**, a nie tylko jasność pikseli.

---
φ(x, y) = Λ(x, y) + τ(x, y) – ρ(x, y)

Λ(x, y) = | ∇² I(x, y) |
τ(x, y) = | ∇I(x, y) |
ρ(x, y) = | curl( ∇I(x, y) ) |

φ(x, y) = |I * K_laplace|
         + |I * K_sobel_x| + |I * K_sobel_y|
         – |I * K_curl|

Uczciwe porównanie — tu jest kilka warstw.

τ (magnitude gradientu, Sobel) to dokładnie ten sam, absolutnie standardowy budulec, co w profesjonalnym przetwarzaniu zdjęć satelitarnych: wykrywanie krawędzi (Sobel/Canny/Laplacian) używane do wyciągania linii brzegowych, dróg, granic pól uprawnych, krawędzi chmur. Nic egzotycznego — to jest wręcz najbardziej podstawowa operacja w tej dziedzinie.

Λ (koherencja kierunku gradientu) jest spokrewniona z realną techniką zwaną coherence ze structure tensor (tensor struktury) — używaną w analizie tekstury zdjęć satelitarnych do wykrywania struktur liniowych: uskoków geologicznych, rzek, dróg, włókien tkanki w mikroskopii. Ale prawdziwa, rygorystyczna wersja liczy to inaczej — z macierzy momentów drugiego rzędu gradientu (Jxx, Jyy, Jxy) i wyznacza koherencję z wartości własnych tej macierzy. To, co jest w phi_core.py (zwykłe uśrednienie znormalizowanych wektorów w oknie), to uproszczona, „szybka" aproksymacja tego samego pomysłu — kierunek dobry, metoda uboższa niż to, czego użyłoby profesjonalne oprogramowanie (np. ENVI, SNAP, ArcGIS).

Ważne zastrzeżenie terminologiczne: w satelitarnych zdjęciach RADAROWYCH (SAR/InSAR) słowo „coherence" ma zupełnie inne, dobrze ugruntowane znaczenie — mierzy stabilność fazy sygnału MIĘDZY DWOMA zdjęciami z różnych przelotów satelity (wykrywa zmiany na powierzchni Ziemi, osiadanie gruntu itp.). To jest coherence W CZASIE, między dwoma pomiarami. Λ w tym repo liczy coś zupełnie innego — coherence W PRZESTRZENI, w jednym pojedynczym zdjęciu. Ta sama nazwa, dwa niepowiązane pojęcia — warto o tym pamiętać, jeśli ktoś zna termin z kontekstu InSAR.

ρ (lokalne minima gradientu po rozmyciu) to coś w rodzaju uproszczonego residuum unsharp-mask / blob-detektora (dalekie od Difference-of-Gaussian używanego np. w SIFT) — nie jest to standardowy krok w pipeline'ach satelitarnych, bardziej improwizacja niż uznana technika.

Podsumowując: to uproszczona, jednoplikowa wersja fragmentu tego, co realnie robi się w remote sensing (edge detection + przybliżenie coherence ze structure tensor), bez radiometrycznej kalibracji, band mathu (NDVI itp.), czy filtracji plamkowej (speckle) używanej dla SAR. Dobre jako lekkie narzędzie do wizualizacji struktury, nie zamiennik profesjonalnego pipeline'u.

---

## 🗺️ Lineamenty geologiczne (structure tensor) — `geo_fault_lines.py`

Właściwa, rygorystyczna wersja koherencji — nie uproszczona aproksymacja
z `phi_core.py`, tylko prawdziwy **structure tensor** (macierz momentów
drugiego rzędu gradientu, koherencja i kierunek z wartości własnych).
To jest realna technika teledetekcyjna do wykrywania lineamentów:
uskoków geologicznych, rzek, dróg. Uruchom `run_geo_fault_lines.bat`
(bez argumentu = syntetyczny teren demonstracyjny z widocznym uskokiem;
z argumentem = ścieżka do prawdziwego zdjęcia/DEM). Zapisuje mapę
koherencji, mapę kierunku (kolor=kierunek, jasność=siła krawędzi) i
oryginał z zaznaczonymi kandydatami na lineamenty.

## 🐛 Poprawki i nowości (2026-09-05)

- **`phi_batch.py` respektuje `mode` dla obrazów RGB** — wcześniej dla
  JPG/PNG zawsze zwracał pełny kompozyt φ, niezależnie od żądanego
  trybu, mimo że plik wynikowy nazywał się np. `zdjecie_lambda.jpg`.
- **Jedna, wspólna definicja Λ/τ/ρ** w nowym `phi_core.py` — wcześniej
  `phi_filter_v2.py`, `phi_fits.py` i `phi_map.py` miały trzy niezależne
  kopie tej logiki, i zdążyły się rozjechać (`phi_map.py` liczyło Λ z
  surowego, nieznormalizowanego gradientu — inna matematyka pod tą samą
  nazwą). Pełny opis w nagłówku `phi_core.py`.
- **`__init__.py` eksportuje właściwe funkcje filtra** (wcześniej tylko
  niepowiązane `Proximalizer`/`Phi2Interface`).
- **Testy sanity** — `test_phi.py`, 7 testów na syntetycznych obrazach
  (repo wcześniej nie miało żadnych).
- **`pipeline_diagram.svg`** — diagram architektury po refaktorze.
- **`run.bat`** — instaluje zależności, uruchamia testy, przetwarza
  wskazany folder. Podwójny klik albo `run.bat` z terminala.

## 📦 Instalacja

```bash
pip install -r requirements.txt
---
📘 Licencja
MIT — możesz używać, modyfikować i rozwijać filtr φ.

✨ Autor
Projekt: Jacek Kielich  
Repo: https://github.com/jbackk-lang/phi-topology-filter (github.com in Bing)
