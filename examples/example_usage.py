from proximalizer import Proximalizer
from phi2_interface import Phi2Interface

# przykładowa funkcja φ₂
def phi2(x):
    return [v // 2 for v in x]  # symboliczna kompresja

phi = Phi2Interface(phi2)
prox = Proximalizer(phi)

A = [10, 20, 30]
B = [12, 18, 33]

delta = prox.prox(A, B)
print("Δ =", delta)
