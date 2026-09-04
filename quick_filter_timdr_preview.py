"""
quick_filter_timdr_preview.py -- prawdziwa matematyka Lambda/tau/rho z
phi_core.py, flow:

  1) okienko: wybierz plik wejściowy
  2) okienko: wybierz FOLDER, w którym zapisać wyniki ("Zapisz jako" ->
     folder, nie pojedynczy plik)
  3) liczy Λ/τ/ρ/φ i zapisuje 5 plików (oryginał + 4 mapy) do TEGO
     JEDNEGO folderu
  4) na koniec otwiera ten folder w Eksploratorze Windows -- Explorer
     ma wbudowany podgląd miniaturek, więc nie trzeba własnego okienka
     do przeglądania obrazów.

Uruchom przez run_quick_filter_timdr_preview.bat.
"""

import os
import sys

import numpy as np
import cv2
from PIL import Image

try:
    from tkinter import Tk, filedialog, messagebox
except ImportError:
    Tk = None
    filedialog = None
    messagebox = None

from phi_core import (
    gradient_field,
    coherence_lambda,
    flow_tau,
    defects_rho,
    phi_composite,
    normalize,
)


# -----------------------------
# 1. Wybór pliku wejściowego
# -----------------------------
def wybierz_plik():
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Wybierz zdjęcie astronomiczne",
        filetypes=[
            ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("Wszystkie pliki", "*.*"),
        ],
    )
    root.destroy()
    return filename


# -----------------------------
# 2. Wybór folderu zapisu (okienko "Zapisz jako" -> folder)
# -----------------------------
def wybierz_folder_zapisu(domyslny_folder):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Wybierz folder, w którym zapisać wyniki filtra",
        initialdir=domyslny_folder,
        mustexist=True,
    )
    root.destroy()
    return folder


def _to_gray(img_rgb_float):
    return cv2.cvtColor((img_rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _map_to_image(map01):
    arr = np.clip(map01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _composite_to_image(img_rgb_float, map01):
    map3 = np.stack([map01] * 3, axis=-1)
    out = img_rgb_float * (0.4 + 0.6 * map3)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# -----------------------------
# 3. Otwieranie folderu wynikowego -- Windows ma wbudowany podgląd
# -----------------------------
def otworz_w_eksploratorze(folder):
    if sys.platform == "win32":
        os.startfile(folder)
    else:
        import subprocess
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, folder])


# -----------------------------
# 4. Główna funkcja
# -----------------------------
def main():
    if Tk is None:
        print("BLAD: modul tkinter nie jest dostepny w tej instalacji Pythona.")
        print("Zainstaluj Pythona z python.org (tkinter jest w standardowym")
        print("instalatorze) albo doinstaluj pakiet tk dla swojej dystrybucji.")
        return

    path = wybierz_plik()
    if not path:
        print("Nie wybrano pliku.")
        return
    print("Wybrano plik:", path)

    out_dir = wybierz_folder_zapisu(os.path.dirname(path))
    if not out_dir:
        print("Nie wybrano folderu zapisu.")
        return
    print("Folder zapisu:", out_dir)

    img = Image.open(path).convert("RGB")
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = _to_gray(img_arr)

    mag, nx, ny = gradient_field(gray)
    Lambda = normalize(coherence_lambda(nx, ny))
    Tau = flow_tau(mag)
    Rho = defects_rho(mag)
    Phi = phi_composite(Lambda, Tau, Rho)

    stem = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1] or ".jpg"

    do_zapisania = [
        ("ORYGINAL", img),
        ("LAMBDA", _map_to_image(Lambda)),
        ("TAU", _map_to_image(Tau)),
        ("RHO", _map_to_image(Rho)),
        ("PHI_TOTAL", _composite_to_image(img_arr, Phi)),
    ]

    saved = []
    for label, img_out in do_zapisania:
        out_path = os.path.join(out_dir, f"{stem}_{label}{ext}")
        img_out.save(out_path)
        saved.append(out_path)

    print("Zapisano:")
    for p in saved:
        print("  ", p)

    try:
        root = Tk()
        root.withdraw()
        messagebox.showinfo(
            "Gotowe",
            f"Zapisano {len(saved)} plików w:\n{out_dir}\n\nOtwieram folder w Eksploratorze...",
        )
        root.destroy()
    except Exception:
        pass

    otworz_w_eksploratorze(out_dir)


if __name__ == "__main__":
    main()
