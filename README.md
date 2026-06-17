# phi-topology-filter  Topologiczny filtr (idealny do zdjęć kosmosu)

Topologiczny filtr **φ** oparty na geometrii **Λ–τ–ρ** do analizy obrazów kosmicznych (FITS/JPG/PNG).

## Idea

Zwykłe filtry pokazują **kontrast**.  
Ten filtr próbuje pokazać **strukturę przestrzeni**:

- **Λ** – stabilne struktury (kolumny, włókna, rdzenie),
- **τ** – przejścia fazowe (halo, rezonanse, gradienty),
- **ρ** – defekty (miejsca narodzin materii, załamania skrętu),
- **φ** – punkt równowagi między Λ i τ (stan „porządkujący” przestrzeń).

Celem jest uzyskanie wizualizacji, która lepiej oddaje **stadium materii i przestrzeni**, a nie tylko jasność pikseli.

## Instalacja

```bash
pip install -r requirements.txt
from phi_filter import phi_filter_image

out_img = phi_filter_image("input.jpg", mode="lambda-tau-rho")
out_img.save("output_phi.jpg")
