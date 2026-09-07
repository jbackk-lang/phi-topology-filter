"""
phi_hessian.py -- DRUGI rdzen matematyczny w tym repo, celowo OSOBNY od
phi_core.py. phi_core.py liczy koherencje/kierunek z tensora struktury
(macierz momentow gradientu, PIERWSZA pochodna) -- odpowiada na pytanie
"w ktora strone biegnie struktura". Ten plik liczy vesselness/ridgeness
z Hesjanu (macierz DRUGIEJ pochodnej, znormalizowana wzgledem skali wg
Lindeberga) -- odpowiada na INNE pytanie: "czy w tym miejscu jest
struktura o SZEROKOSCI pasujacej do danej skali".

DLACZEGO OSOBNY PLIK (nie rozszerzenie phi_core.py): zweryfikowano
eksperymentalnie (2026-09-08, patrz _experiment_multiscale.py w tym
repo), ze koherencja z phi_core.py NIE JEST czula na skale -- izolowana
linia daje TA SAMA koherencje niezaleznie od window_sigma (bo mag/gx/gy
sa liczone z surowego, nieznormalizowanego gradientu, niezaleznie od
window_sigma, ktory wplywa tylko na rozmycie momentow Jxx/Jyy/Jxy).
"Policz przy kilku sigma, wez max" na phi_core.structure_tensor_coherence
nic wiec nie daje. Prawdziwa czulosc na skale wymaga INNEJ matematyki
(Hesjan + normalizacja Lindeberga), nie tylko innego parametru -- stad
osobny plik, nie funkcja dodana do phi_core.py.

ZWALIDOWANE EKSPERYMENTALNIE (patrz _experiment_frangi_multiscale.py,
_experiment_scale_resonance.py, _experiment_resonance_v2.py w tym repo):

1. Pojedyncza skala, poprawnie znormalizowana (mnozenie przez sigma^2,
   normalizacja Lindeberga rzedu gamma=1): odpowiedz na IZOLOWANEJ linii
   2px ma prawdziwe MAKSIMUM w funkcji sigma (szczyt ~sigma=3, nie rosnie
   bez konca) -- W PRZECIWIENSTWIE do koherencji z phi_core.
2. KRYTYCZNE dla poprawnosci: stala `c` (normalizacja structureness w
   formule vesselness) MUSI byc policzona RAZ, z globalnego maksimum
   structureness po WSZYSTKICH skalach w stosie -- NIE osobno per
   skala. Pierwsza (bledna) probka z c per-skala dawala odpowiedz
   rosnaca MONOTONICZNIE z sigma (brak maksimum) -- bo samo-normalizacja
   per skala niszczy wlasnie informacje o dopasowaniu skali, ktora
   chcemy zmierzyc. `multi_scale_vesselness()` ponizej liczy `c`
   poprawnie (raz, globalnie).
3. Multi-scale (max z kilku sigma per piksel) FAKTYCZNIE lapie struktury
   o roznej szerokosci jednoczesnie: test z linia 2px i linia 14px --
   zadna POJEDYNCZA sigma z testowanego zakresu nie dawala dobrej
   odpowiedzi na obu naraz (np. sigma=1.5: cienka=0.116, gruba=0.092;
   sigma=8: gruba=0.586, cienka=0.080), multi-scale max dal cienka=0.184,
   gruba=0.589 -- lepiej niz jakakolwiek pojedyncza skala dla obu.
4. WALIDACJA NA REALNYM CELU (_experiment_roc_retina.py): test ROC
   (recall na prawdziwej masce naczyn, przy dopasowanym FPR) na
   syntetycznym drzewie naczyniowym ze ZMIENNA szerokoscia (nie sztuczne
   dwie linie) -- multi_scale_vesselness druzgocaco wygrywa z istniejaca
   koherencja z phi_core (analyze_retina, window_sigma=3.5 stale):
   przy FPR=1%: Vmax=95.6% recall vs koherencja=32.7%; przy FPR=2%:
   97.3% vs 59.3%; koherencja W OGOLE nie dochodzi do 100% recall nawet
   przy FPR=20% (plaskowyz na 86.7%) -- pojedyncza stala skala fizycznie
   nie widzi czesci drzewa o innej szerokosci, niezaleznie od progu.
   To jest najmocniejszy, bo na najbardziej realistycznym celu,
   argument za multi-scale Hesjanem w tym repo.

UCZCIWE OGRANICZENIE -- `scale_resonance()` (TIMDR-M, koincydencja w
przestrzeni skali zamiast czasu -- pomysl uzytkownika, konsultowany i
przetestowany, W TRZECH KOLEJNYCH TURACH, az do wlasciwej metodologii):

- Tura 1 (naiwna): "% szumu przechodzacego prog" spadalo z 40.6% do
  15.6% przy progu ">=3 z 8 zgodnych skal" -- ale kosztem pokrycia
  cienkiej linii (42.8%->31.2%). Wyglada jak realny kompromis.
- Tura 2 (proba naprawy): miekkie wazenie
  (Vfinal=Vmax*(0.5+0.5*RM) zamiast twardego progu ">=3 skale") i/lub
  gestsze probkowanie malych sigma. Gestsze sigma: brak mierzalnej
  roznicy. Miekkie wazenie: PRZY TYM SAMYM progu bezwzglednym co Vmax,
  co jest metodologicznie NIESPRAWIEDLIWE -- Vfinal=Vmax*waga z
  waga<=1 z definicji nie moze dac wyzszego pokrycia niz Vmax przy
  jednym wspolnym progu, wiec ten test niczego nie dowodzil.
- Tura 3 (test ROC, wlasciwy -- zaproponowany przez uzytkownika):
  dopasuj prog OSOBNO dla Vmax i dla Vfinal tak, zeby OBA dawaly TEN
  SAM poziom przepuszczonego szumu (dopasowany punkt pracy), DOPIERO
  wtedy porownaj pokrycie struktur. WYNIK: przy kazdym z 6 testowanych
  poziomow szumu (5%-40%) Vmax i Vfinal daja PRAKTYCZNIE IDENTYCZNE
  pokrycie (cienka: roznice <=0.8pp, gruba: dokladnie 93.8% w obu, na
  kazdym poziomie) -- ZERO realnej przewagi rezonansu jako wagi na
  SZTUCZNYM celu (dwie proste rownolegle linie). Tura 1 mierzyla
  wylacznie artefakt przeskalowania, nie prawdziwy efekt. Wyjasnienie:
  RM liczone jest z TEGO SAMEGO stosu odpowiedzi co Vmax, wiec jest z
  nim silnie skorelowane -- wazenie skorelowana wielkoscia nie zmienia
  istotnie kolejnosci rankingowej pikseli, stad identyczna krzywa ROC.
- Tura 4 (ten sam test ROC, ale na REALNYM celu -- syntetyczne drzewo
  naczyniowe ze zmienna szerokoscia, `_experiment_roc_retina.py`,
  ground-truth maska z generatora): tu Vfinal daje MALY, ale
  KONSEKWENTNY plusik nad samym Vmax (recall przy FPR=1%: 96.3% vs
  95.6%; FPR=2%: 97.6% vs 97.3%). Prawdopodobnie na bardziej zlozonej
  strukturze (rozgalezienia, zmienna orientacja, nie tylko prosta
  linia) korelacja RM-Vmax jest niedoskonala, wiec rezonans dokada
  odrobine realnej informacji -- ale efekt jest maly (<1pp), nie
  przesadzac z jego znaczeniem.
  (Patrz _experiment_scale_resonance.py, _experiment_resonance_v2.py,
  _experiment_roc_comparison.py, _experiment_roc_retina.py w tym repo
  -- wszystkie cztery tury zachowane jako dokumentacja procesu.)

WNIOSEK: `scale_resonance()` NIE poprawia detekcji jako waga/filtr
polaczony z Vmax (zweryfikowane wlasciwym testem ROC) -- ale (patrz
`_experiment_hessian_on_retina.py`) daje uzyteczna, INNA informacje:
na prawdziwym drzewie naczyniowym RM zaznacza KRAWEDZIE/KONTURY
szerokich naczyn (nizszy rezonans w centrum, wyzszy na brzegu), wiec
nadaje sie jako wskaznik krawedzi/szerokosci, NIE jako samodzielny ani
polaczony detektor obecnosci struktury. Funkcja zostaje w module z tym
jawnym zastrzezeniem -- do uzycia swiadomego, nie jako domyslny filtr.
"""

import numpy as np
import cv2


def _hessian_eigs(gray, sigma):
    """Wartosci wlasne Hesjanu (Ixx, Ixy, Iyy) po wygladzeniu Gaussem
    (sigma) i normalizacji Lindeberga (mnozenie przez sigma^2, rzad
    gamma=1 dla pochodnej 2. rzedu) -- BEZ tej normalizacji odpowiedzi
    z roznych skal nie sa ze soba porownywalne (patrz naglowek modulu).
    Zwraca (lambda1, lambda2) z |lambda1| <= |lambda2|."""
    gray = gray.astype(np.float64)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)

    Ixx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3) * (sigma ** 2)
    Iyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3) * (sigma ** 2)
    Ixy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3) * (sigma ** 2)

    mean = (Ixx + Iyy) / 2.0
    diff = (Ixx - Iyy) / 2.0
    disc = np.sqrt(diff * diff + Ixy * Ixy)
    a = mean - disc
    b = mean + disc

    swap = np.abs(a) > np.abs(b)
    lambda1 = np.where(swap, b, a)
    lambda2 = np.where(swap, a, b)
    return lambda1, lambda2


def multi_scale_vesselness(gray, sigmas, beta=0.5, return_stack=False):
    """
    Vesselness Frangiego (bright ridge na ciemnym tle), na kilku skalach
    naraz, z poprawnie policzona (RAZ, globalnie) stala normalizujaca
    `c` -- patrz pkt. 2 w naglowku modulu, to jest KRYTYCZNE dla
    poprawnosci multi-scale, nie kosmetyczny szczegol.

    sigmas: iterowalna kolekcja sigma do przetestowania. Dobierz zakres
    tak, zeby obejmowal ~polowe oczekiwanej szerokosci najwezszej i
    najszerszej struktury, ktora chcesz wykryc (np. dla naczyn
    siatkowki o szerokosci 2-14px: sigmas ~ [1, 1.5, 2, 3, 4, 6, 8]).

    return_stack: jesli True, zwraca tez slownik {sigma: vesselness} --
    potrzebny dla scale_resonance().

    Zwraca (vmax, stack) gdzie vmax = max(vesselness) per piksel po
    wszystkich sigma. Jesli return_stack=False, stack=None.
    """
    sigmas = list(sigmas)
    all_l1, all_l2, all_S = {}, {}, {}
    for s in sigmas:
        l1, l2 = _hessian_eigs(gray, s)
        all_l1[s], all_l2[s] = l1, l2
        all_S[s] = np.sqrt(l1 ** 2 + l2 ** 2)

    global_s_max = max(S.max() for S in all_S.values())
    c = 0.5 * global_s_max if global_s_max > 1e-9 else 1.0

    stack = {}
    for s in sigmas:
        l1, l2, S = all_l1[s], all_l2[s], all_S[s]
        Rb = l1 / (l2 + 1e-12)
        v = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1 - np.exp(-(S ** 2) / (2 * c ** 2)))
        v[l2 > 0] = 0.0  # tylko jasne grzbiety na ciemnym tle
        stack[s] = v

    vmax = np.maximum.reduce(list(stack.values()))
    return vmax, (stack if return_stack else None)


def scale_resonance(stack, threshold=0.10):
    """
    EKSPERYMENTALNE -- patrz "UCZCIWE OGRANICZENIE" w naglowku modulu.

    TIMDR-M rezonans przeniesiony z czasu na przestrzen skali:
    RM(x,y) = odsetek skal, dla ktorych vesselness(x,y,sigma) > threshold.
    Wysoki RM = wiele sasiednich skal "widzi" te sama strukture (typowe
    dla prawdziwej, szerokiej-w-skali struktury); niski RM = tylko
    pojedyncza skala zareagowala (typowe dla szumu).

    Wymaga `stack` ze `multi_scale_vesselness(..., return_stack=True)`.

    NIE UZYWAJ jako wagi/filtra polaczonego z vesselness (np.
    Vmax*(0.5+0.5*RM)) w celu poprawy detekcji -- WLASCIWY test ROC
    (dopasowany poziom szumu, nie ten sam prog bezwzgledny) pokazal
    ZERO realnej roznicy wzgledem samego Vmax (patrz "UCZCIWE
    OGRANICZENIE" w naglowku modulu, tura 3). Wczesniejszy pozorny
    "kompromis" (40.6%->15.6% szumu) byl artefaktem niesprawiedliwego
    porownania przy jednym progu, nie prawdziwym efektem.

    UZYWAJ NATOMIAST jako niezaleznej informacji o KRAWEDZI/SZEROKOSCI
    struktury (zweryfikowane na prawdziwym drzewie naczyniowym w
    _experiment_hessian_on_retina.py) -- np. do szacowania grubosci
    naczynia albo segmentacji konturu, nie do samej decyzji "czy to w
    ogole jest naczynie".

    DODATKOWA OBSERWACJA (test na _experiment_hessian_on_retina.py,
    prawdziwe -- nie syntetyczna linia -- drzewo naczyniowe): RM daje
    PUSTE KONTURY na grubych naczyniach, nie wypelnione linie -- srodek
    szerokiego naczynia ma NIZSZY rezonans niz jego brzegi (lokalnie
    "plaski"/mniej grzbietowy w centrum, bardziej grzbietowy przy
    krawedzi, gdzie intensywnosc realnie opada). Wniosek: RM w obecnej
    formie nadaje sie jako WSKAZNIK KRAWEDZI/SZEROKOSCI naczynia (np. do
    szacowania grubosci, albo segmentacji konturu), NIE jako samodzielny
    detektor "czy tu w ogole jest naczynie" -- do tego lepszy jest
    surowy `vmax` z multi_scale_vesselness (ktory wypelnia cale
    naczynie, nie tylko brzegi).
    """
    sigmas = list(stack.keys())
    votes = np.zeros_like(next(iter(stack.values())))
    for s in sigmas:
        votes += (stack[s] > threshold).astype(np.float64)
    return votes / len(sigmas)
