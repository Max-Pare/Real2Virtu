@echo off
rem Sets up Real2Virtu on a new Windows machine. Run from a normal Command Prompt:
rem     setup.bat                 everything
rem     setup.bat --skip-system   Python packages and COLMAP only (no winget installs)
rem Safe to run again: steps that are already done are skipped.
setlocal
cd /d "%~dp0"

rem Signed python.org build (Python Software Foundation certificate), taken from the
rem official NuGet package. Unsigned Python builds get blocked by Smart App Control.
set "PY_VERSION=3.12.10"
set "PY_SHA256=0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8"
set "PY_DIR=tools\python312_signed"

if exist "%PY_DIR%\python.exe" goto run

echo == python ==
echo Downloading Python %PY_VERSION% ...
if not exist tools mkdir tools
set "PKG=tools\python-%PY_VERSION%.nupkg"
set "TMP_DIR=tools\_python_unpack"
curl.exe -fsSL -o "%PKG%" "https://www.nuget.org/api/v2/package/python/%PY_VERSION%"
if errorlevel 1 goto fail_download
certutil -hashfile "%PKG%" SHA256 | findstr /i /x "%PY_SHA256%" >nul
if errorlevel 1 goto fail_hash
if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%"
mkdir "%TMP_DIR%"
tar.exe -xf "%PKG%" -C "%TMP_DIR%" tools
if errorlevel 1 goto fail_unpack
move "%TMP_DIR%\tools" "%PY_DIR%" >nul
if errorlevel 1 goto fail_unpack
rmdir /s /q "%TMP_DIR%"
del "%PKG%"

:run
"%PY_DIR%\python.exe" tools\setup_env.py %*
exit /b %errorlevel%

:fail_download
echo Could not download Python %PY_VERSION% from nuget.org.
exit /b 1

:fail_hash
echo The downloaded Python package does not match the expected SHA-256; not using it.
del "%PKG%"
exit /b 1

:fail_unpack
echo Could not unpack the Python package into %PY_DIR%.
exit /b 1
