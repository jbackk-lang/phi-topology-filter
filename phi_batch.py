import os
from pathlib import Path
from phi_filter_v2 import phi_filter_v2
from phi_fits import phi_fits

SUPPORTED_IMG = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
SUPPORTED_FITS = [".fits", ".fit", ".fts"]


def process_directory(input_dir, output_dir="output", mode="phi", strength=1.0):
    """
    Przetwarza cały katalog obrazów:
    - FITS → phi_fits
    - JPG/PNG → phi_filter_v2

    mode:
        "phi", "lambda", "tau", "rho", "phi-mix"
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    files = list(input_dir.iterdir())
    if not files:
        print("Brak plików w katalogu.")
        return

    print(f"Przetwarzam katalog: {input_dir}")
    print(f"Tryb: {mode}")

    for f in files:
        ext = f.suffix.lower()

        # FITS
        if ext in SUPPORTED_FITS:
            print(f"[FITS] {f.name}")
            out = phi_fits(str(f), mode=mode)
            out.save(output_dir / f"{f.stem}_{mode}.jpg")

        # Obrazy RGB
        elif ext in SUPPORTED_IMG:
            print(f"[IMG]  {f.name}")
            out = phi_filter_v2(str(f), strength=strength)
            out.save(output_dir / f"{f.stem}_{mode}.jpg")

        else:
            print(f"[SKIP] {f.name} — nieobsługiwany format.")

    print(f"Zakończono. Wyniki zapisane w: {output_dir}")
