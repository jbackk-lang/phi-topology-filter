@echo off
setlocal

echo ==================================================
echo   phi-topology-filter :: geo_fault_lines.py
echo   (koherencja ze structure tensor -- lineamenty)
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

if "%~1"=="" (
    echo Brak podanego pliku -- uzyje syntetycznego terenu demonstracyjnego.
    python geo_fault_lines.py
) else (
    python geo_fault_lines.py "%~1"
)

echo.
pause
