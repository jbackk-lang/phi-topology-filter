from phi_map import phi_structure_map
from PIL import Image
import numpy as np

INPUT = "pillars.jpg"

def save_map(arr, name):
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(name)

def main():
    print("Generuję mapy Λ–τ–ρ...")
    L, T, R = phi_structure_map(INPUT)

    save_map(L, "lambda_map.jpg")
    save_map(T, "tau_map.jpg")
    save_map(R, "rho_map.jpg")

    print("Zapisano: lambda_map.jpg, tau_map.jpg, rho_map.jpg")

if __name__ == "__main__":
    main()
