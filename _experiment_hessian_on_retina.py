"""
_experiment_hessian_on_retina.py -- test phi_hessian.py na prawdziwym
celu (nie kolejnej syntetycznej linii): syntetyczne drzewo naczyniowe z
retina_vessel_orientation.py, ktore ma NATURALNIE zmienna szerokosc
(glowne galezie grube 3.5-5.5px, kolejne generacje x0.55-0.75 per
poziom, do 4 poziomow glebokosci -- czyli realny zakres szerokosci w
JEDNYM obrazie, dokladnie problem, do ktorego multi-scale ma sluzyc).

Porownanie:
1. multi_scale_vesselness() z phi_hessian.py (Hesjan, wieloskalowy)
2. vessel_score z analyze_retina() (koherencja z phi_core, pojedyncza
   skala window_sigma=3.5, juz istniejaca)

Nie zeby jedno "wygralo" -- to dwie rozne miary (kierunek/koherencja vs
dopasowanie-do-szerokosci) -- ale sprawdzamy, czy Hesjan faktycznie daje
sensowna, uzyteczna mape na PRAWDZIWYM (nie sztucznie skonstruowanym)
obrazie, nie tylko na dwoch idealnych rownoleglych liniach.

Uzycie:
    python _experiment_hessian_on_retina.py
"""

import numpy as np
from PIL import Image

from phi_hessian import multi_scale_vesselness, scale_resonance
from retina_vessel_orientation import _synthetic_retina, analyze_retina, _fov_mask


def main():
    size = 400
    print("Generuje syntetyczne drzewo naczyniowe (retina_vessel_orientation._synthetic_retina)...")
    fundus = _synthetic_retina(size=size)
    Image.fromarray(fundus).convert("RGB").save("_tmp_retina_for_hessian_test.png")

    fov = _fov_mask((size, size))

    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    print(f"Licze multi_scale_vesselness na {len(sigmas)} skalach: {sigmas} ...")
    vmax, stack = multi_scale_vesselness(fundus, sigmas, return_stack=True)
    vmax[~fov] = 0.0

    resonance = scale_resonance(stack, threshold=0.10)
    resonance[~fov] = 0.0

    print("Licze analyze_retina() (koherencja z phi_core, dla porownania) ...")
    out_coherence = analyze_retina("_tmp_retina_for_hessian_test.png")

    # zapisz wizualizacje
    vmax_img = np.clip(vmax / (vmax.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(vmax_img).save("_tmp_retina_HESSIAN_VESSELNESS.png")

    resonance_img = np.clip(resonance * 255, 0, 255).astype(np.uint8)
    Image.fromarray(resonance_img).save("_tmp_retina_HESSIAN_RESONANCE.png")

    # prosta metryka: ile pikseli w FOV ma silna odpowiedz w kazdej metodzie
    vessel_like_hessian = (vmax > 0.10) & fov
    vessel_like_coherence = (out_coherence["vessel_score"] > 0.30) & fov

    print("\n=== Wyniki (w obrebie FOV) ===")
    print(f"  Hesjan: piksele z vesselness>0.10: {vessel_like_hessian.sum()} "
          f"({vessel_like_hessian.mean()*100:.2f}% FOV)")
    print(f"  Koherencja (phi_core): piksele z vessel_score>0.30: {vessel_like_coherence.sum()} "
          f"({vessel_like_coherence.mean()*100:.2f}% FOV)")

    overlap = vessel_like_hessian & vessel_like_coherence
    only_hessian = vessel_like_hessian & ~vessel_like_coherence
    only_coherence = vessel_like_coherence & ~vessel_like_hessian
    print(f"  Wspolne piksele (obie metody sie zgadzaja): {overlap.sum()}")
    print(f"  Tylko Hesjan wykryl: {only_hessian.sum()}")
    print(f"  Tylko koherencja wykryla: {only_coherence.sum()}")

    print("\nZapisano: _tmp_retina_for_hessian_test.png, _tmp_retina_HESSIAN_VESSELNESS.png, "
          "_tmp_retina_HESSIAN_RESONANCE.png")


if __name__ == "__main__":
    main()
