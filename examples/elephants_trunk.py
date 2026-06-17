from phi_filter import phi_filter_image
from pathlib import Path

INPUT = "elephant.jpg"   # podmień na własny plik
OUTPUT = "elephant_phi.jpg"

def main():
    print("Filtr φ na Mgławicy Trąba Słonia...")
    out = phi_filter_image(INPUT, mode="lambda-tau-rho", strength=1.2)
    out.save(OUTPUT)
    print(f"Wynik zapisany jako: {OUTPUT}")

if __name__ == "__main__":
    main()
