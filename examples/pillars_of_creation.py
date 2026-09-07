from phi_filter import phi_filter_image
from pathlib import Path

# Ścieżka do obrazu wejściowego
INPUT = "pillars.jpg"   # podmień na własny plik
OUTPUT = "pillars_phi.jpg"

def main():
    print("Przetwarzam obraz przez filtr φ...")
    out = phi_filter_image(INPUT, mode="lambda-tau-rho", strength=1.0)
    out.save(OUTPUT)
    print(f"Zapisano wynik: {OUTPUT}")

if __name__ == "__main__":
    main()
