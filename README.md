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

---

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
