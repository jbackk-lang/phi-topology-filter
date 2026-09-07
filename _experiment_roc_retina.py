"""
_experiment_roc_retina.py -- WLASCIWY test ROC (dopasowany poziom
falszywych alarmow, nie ten sam prog bezwzgledny -- patrz
_experiment_roc_comparison.py) powtorzony na PRAWDZIWYM celu: syntetyczne
drzewo naczyniowe ze zmienna szerokoscia (retina_vessel_orientation.py),
nie dwie sztuczne rownolegle linie. To jest bardziej informatywny test,
bo:
- szerokosc naczyn jest naturalnie zmienna (grube przy tarczy, cienkie
  na koncach galezi) -- dokladnie problem, do ktorego multi-scale ma
  sluzyc,
- mamy PRAWDZIWA maske ground-truth (canvas sprzed rozmycia/tla,
  `_synthetic_retina(..., return_mask=True)`) -- wiec TPR/FPR sa liczone
  z rzeczywistej prawdy, nie z przyblizenia pasmem wokol linii.

Porownuje 3 metody przy DOPASOWANYM poziomie falszywych alarmow (FPR):
1. Vmax -- surowy multi_scale_vesselness (bez rezonansu)
2. Vfinal -- Vmax*(0.5+0.5*RM) (miekkie wazenie rezonansem)
3. vessel_score -- koherencja z phi_core (analyze_retina, istniejaca
   metoda), dla pelnego obrazu.

Uzycie:
    python _experiment_roc_retina.py
"""

import numpy as np

from phi_hessian import multi_scale_vesselness, scale_resonance
from retina_vessel_orientation import _synthetic_retina, analyze_retina, _fov_mask


def _threshold_for_target_fpr(values_fov_nonvessel, target_fpr, candidates):
    best_t, best_diff, best_rate = None, None, None
    for t in candidates:
        rate = float(np.mean(values_fov_nonvessel > t))
        diff = abs(rate - target_fpr)
        if best_diff is None or diff < best_diff:
            best_diff, best_t, best_rate = diff, t, rate
    return best_t, best_rate


def main():
    size = 400
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    vote_threshold = 0.10

    print("Generuje syntetyczne drzewo naczyniowe z ground-truth maska...")
    fundus, vessel_mask = _synthetic_retina(size=size, return_mask=True)
    fov = _fov_mask((size, size))
    nonvessel_fov = fov & ~vessel_mask

    from PIL import Image
    Image.fromarray(fundus).convert("RGB").save("_tmp_retina_roc.png")

    print(f"Vessel mask: {vessel_mask.sum()} pikseli ({vessel_mask.sum()/fov.sum()*100:.2f}% FOV)")

    print("Licze multi_scale_vesselness + scale_resonance...")
    vmax, stack = multi_scale_vesselness(fundus, sigmas, return_stack=True)
    rm = scale_resonance(stack, threshold=vote_threshold)
    vfinal = vmax * (0.5 + 0.5 * rm)

    print("Licze analyze_retina (koherencja phi_core, dla porownania)...")
    out_coh = analyze_retina("_tmp_retina_roc.png")
    vessel_score = out_coh["vessel_score"]

    methods = {
        "Vmax (Hesjan, bez rezonansu)": vmax,
        "Vfinal (Hesjan + rezonans jako waga)": vfinal,
        "vessel_score (koherencja phi_core)": vessel_score,
    }

    print("\n=== ROC: recall (TPR) na prawdziwych naczyniach, przy dopasowanym FPR ===\n")
    header = f"{'cel FPR':>8} |"
    for name in methods:
        header += f" {name[:28]:>28} |"
    print(header)

    for target_fpr in (0.01, 0.02, 0.05, 0.10, 0.20):
        row = f"{target_fpr*100:6.0f}% |"
        for name, values in methods.items():
            candidates = np.linspace(0.0, float(values.max()), 500)
            t, achieved_fpr = _threshold_for_target_fpr(values[nonvessel_fov], target_fpr, candidates)
            tpr = float(np.mean(values[vessel_mask] > t)) * 100
            row += f" FPR={achieved_fpr*100:4.1f}% TPR={tpr:5.1f}%      |"
        print(row)


if __name__ == "__main__":
    main()
