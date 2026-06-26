class Proximalizer:
    def __init__(self, phi_layer):
        self.phi = phi_layer

    def rel(self, A, B):
        # relacja relacyjna – suma różnic
        return sum(abs(a - b) for a, b in zip(A, B))

    def prox(self, A, B):
        A1 = self.phi(A)
        B1 = self.phi(B)
        d0 = self.rel(A, B)
        d1 = self.rel(A1, B1)
        return d1 - d0
