# phi-topology-filter  
Topologiczny filtr **φ** oparty na geometrii **Λ–τ–ρ** do analizy obrazów kosmicznych (FITS/JPG/PNG).

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

## 📦 Instalacja

```bash
pip install -r requirements.txt
---
📘 Licencja
MIT — możesz używać, modyfikować i rozwijać filtr φ.

✨ Autor
Projekt: Jacek Kielich  
Repo: https://github.com/jbackk-lang/phi-topology-filter (github.com in Bing)
