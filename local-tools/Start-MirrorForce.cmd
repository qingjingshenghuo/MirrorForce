@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-MirrorForce.ps1" %*
set "result=%errorlevel%"
if /i "%~1"=="-CheckOnly" exit /b %result%
pause
exit /b %result%
