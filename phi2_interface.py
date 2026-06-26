class Phi2Interface:
    def __init__(self, phi2_fn):
        self.phi2 = phi2_fn

    def __call__(self, data):
        return self.phi2(data)
