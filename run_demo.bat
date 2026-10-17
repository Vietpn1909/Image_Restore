@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM   Chay demo Phuc che anh chan dung cu VN
REM   - Mo demo Gradio (local: http://127.0.0.1:7860)
REM   - Tao link public qua Cloudflare (https://...trycloudflare.com)

set "PY=C:\Users\Admin\miniconda3\envs\vnrestore\python.exe"
set "PROJ=C:\Users\Admin\Desktop\FaceRestore"
set "CF=%PROJ%\cloudflared.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJ%"

echo   DEMO PHUC CHE ANH CHAN DUNG CU VN
echo.

REM --- Kiem tra file can thiet ---
if not exist "%PY%" ( echo [LOI] Khong thay Python env vnrestore tai "%PY%" & pause & exit /b 1 )
if not exist "%CF%" ( echo [LOI] Khong thay cloudflared.exe tai "%CF%" & pause & exit /b 1 )

REM --- Giai phong port 7860 neu dang bi chiem ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"127.0.0.1:7860 .*LISTENING"') do (
  echo Tat tien trinh cu dang giu port 7860 ^(PID %%p^)...
  taskkill /F /PID %%p >nul 2>&1
)

REM --- Chay demo trong cua so rieng ---
echo Khoi dong demo Gradio ^(cua so rieng^)...
start "VN Restore - Demo" "%PY%" -u scripts\demo_gradio.py

REM --- Cho port 7860 san sang (toi da ~60s) ---
echo Cho server len...
set /a tries=0
:waitport
timeout /t 2 /nobreak >nul
netstat -ano | findstr /r /c:"127.0.0.1:7860 .*LISTENING" >nul
if errorlevel 1 (
  set /a tries+=1
  if !tries! lss 30 goto waitport
  echo [LOI] Server chua len sau 60s. Xem cua so "VN Restore - Demo" de biet loi.
  pause
  exit /b 1
)
echo Server da san sang: http://127.0.0.1:7860
echo.
echo   Dang tao LINK PUBLIC...
echo   Link https://...trycloudflare.com se hien ben duoi vai giay.
echo   COPY link do gui nguoi khac de test.
echo.

REM --- Mo tunnel public ngay trong cua so nay ---
"%CF%" tunnel --url http://localhost:7860

echo.
echo Tunnel da dong. Demo local van chay o cua so "VN Restore - Demo".
echo Dong cua so do de tat han demo.
pause
