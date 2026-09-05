# BUGFIX (2026-09-05): __init__.py eksportował wcześniej WYŁĄCZNIE
# Proximalizer/Phi2Interface -- dwie klasy z osobnego, niepowiązanego
# mini-eksperymentu (symboliczna kompresja listy liczb). Właściwe
# funkcje filtra phi (Lambda-tau-rho) nie były w ogóle eksportowane z
# pakietu -- `import phi_topology_filter` nie dawał dostępu do
# niczego, co robi filtr obrazów.

# Importy bezwzgledne (nie relatywne) -- konsekwentnie z reszta repo
# (phi_core.py, phi_batch.py, test_phi.py, examples/*.py wszystkie uzywaja
# `from phi_core import ...` itp., zakladajac plaski katalog na sys.path,
# nie pakiet). Wersja z `.` (relative import) psula kolekcje testow przez
# pytest: __init__.py jest importowany bez kontekstu pakietu (nazwa katalogu
# "phi-topology-filter" zawiera myslnik, wiec nie jest prawidlowa nazwa
# modulu Pythona), co dawalo
# "ImportError: attempted relative import with no known parent package".
from proximalizer import Proximalizer
from phi2_interface import Phi2Interface

from phi_core import (
    gradient_field,
    coherence_lambda,
    flow_tau,
    defects_rho,
    phi_composite,
    normalize,
)
from phi_filter import phi_filter_image
from phi_filter_v2 import phi_filter_v2
from phi_fits import phi_fits, load_fits, fits_to_rgb
from phi_map import phi_structure_map
from phi_batch import process_directory

__all__ = [
    # niepowiązany mini-eksperyment (osobne API, patrz examples/example_usage.py)
    "Proximalizer",
    "Phi2Interface",
    # rdzeń matematyczny filtra phi
    "gradient_field",
    "coherence_lambda",
    "flow_tau",
    "defects_rho",
    "phi_composite",
    "normalize",
    # filtr phi -- obrazy RGB i FITS
    "phi_filter_image",
    "phi_filter_v2",
    "phi_fits",
    "load_fits",
    "fits_to_rgb",
    "phi_structure_map",
    "process_directory",
]
