@echo off
setlocal

echo ==================================================
echo   phi-topology-filter :: quick_filter_timdr_preview.py
echo   (okienko z podgladem Original -^> Lambda -^> tau
echo    -^> rho -^> phi, Poprzedni/Nastepny)
echo ==================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo BLAD: nie znaleziono "python" w PATH. Zainstaluj Pythona 3
    echo ^(python.org^) i zaznacz "Add python.exe to PATH" przy instalacji.
    pause
    exit /b 1
)

echo -- Sprawdzam / instaluje zaleznosci (numpy, opencv, Pillow) --
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo BLAD: nie udalo sie zainstalowac zaleznosci z requirements.txt.
    pause
    exit /b 1
)
echo OK
echo.

python quick_filter_timdr_preview.py

echo.
pause
