@echo off
REM Double-click: installs everything needed and builds the Windows installer (.exe).
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 || goto :err
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 || goto :err
echo.
echo Done. Installer is in: src-tauri\target\release\bundle\nsis
start "" "src-tauri\target\release\bundle\nsis"
exit /b 0
:err
echo Build failed. Read the messages above.
pause
exit /b 1
