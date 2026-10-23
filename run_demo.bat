@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM   Chay demo Phuc che anh chan dung cu VN
REM   Cach dung:  run_demo.bat          -> chi chay local (http://127.0.0.1:7860)
REM               run_demo.bat public   -> them link public qua Cloudflare (https://...trycloudflare.com)

set "PROJ=%~dp0"
set "PY=%PROJ%venv\Scripts\python.exe"
set "PORT=7860"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJ%"

echo   DEMO PHUC CHE ANH CHAN DUNG CU VN
echo.

REM --- Kiem tra file can thiet ---
if not exist "%PY%" ( echo [LOI] Chua co venv. Chay setup_env.bat truoc. & pause & exit /b 1 )
if not exist "checkpoints\stage3\best.pth" echo [CANH BAO] Thieu checkpoints\stage3\best.pth - xem checkpoints\README.md
if not exist "checkpoints\pretrained\GFPGANv1.4.pth" echo [CANH BAO] Thieu checkpoints\pretrained\GFPGANv1.4.pth

REM --- cloudflared: uu tien PATH, sau do thu muc du an ---
set "CF="
for /f "delims=" %%c in ('where cloudflared 2^>nul') do if not defined CF set "CF=%%c"
if not defined CF if exist "%PROJ%cloudflared.exe" set "CF=%PROJ%cloudflared.exe"

REM --- Giai phong port neu dang bi chiem ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"127.0.0.1:%PORT% .*LISTENING"') do (
  echo Tat tien trinh cu dang giu port %PORT% ^(PID %%p^)...
  taskkill /F /PID %%p >nul 2>&1
)

REM --- Chay demo trong cua so rieng ---
echo Khoi dong demo Gradio ^(cua so rieng^)...
start "VN Restore - Demo" "%PY%" -u scripts\demo_gradio.py --port %PORT%

REM --- Cho port san sang (toi da ~90s, lan dau load model lau hon) ---
echo Cho server len...
set /a tries=0
:waitport
timeout /t 2 /nobreak >nul
netstat -ano | findstr /r /c:"127.0.0.1:%PORT% .*LISTENING" >nul
if errorlevel 1 (
  set /a tries+=1
  if !tries! lss 45 goto waitport
  echo [LOI] Server chua len sau 90s. Xem cua so "VN Restore - Demo" de biet loi.
  pause
  exit /b 1
)
echo Server da san sang: http://127.0.0.1:%PORT%
start "" "http://127.0.0.1:%PORT%"

if /i not "%~1"=="public" (
  echo.
  echo Demo dang chay o cua so "VN Restore - Demo". Dong cua so do de tat demo.
  echo Muon tao link public cho nguoi khac: run_demo.bat public
  pause
  exit /b 0
)

if not defined CF ( echo [LOI] Khong tim thay cloudflared ^(PATH hoac thu muc du an^). & pause & exit /b 1 )
echo.
echo   Dang tao LINK PUBLIC...
echo   Link https://...trycloudflare.com se hien ben duoi vai giay.
echo   COPY link do gui nguoi khac de test.
echo.

REM --- Mo tunnel public ngay trong cua so nay ---
"%CF%" tunnel --url http://localhost:%PORT%

echo.
echo Tunnel da dong. Demo local van chay o cua so "VN Restore - Demo".
echo Dong cua so do de tat han demo.
pause
