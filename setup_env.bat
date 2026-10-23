@echo off
setlocal
chcp 65001 >nul

REM   Dung moi truong venv cho du an (Windows, GPU NVIDIA RTX 50 / CUDA 12.8)
REM   Cach dung:  setup_env.bat           -> tao venv neu chua co
REM               setup_env.bat --force   -> xoa venv cu va tao lai

set "PROJ=%~dp0"
set "PY=%PROJ%venv\Scripts\python.exe"
set "PYTHONUTF8=1"
cd /d "%PROJ%"

py -3.11 --version >nul 2>&1
if errorlevel 1 ( echo [LOI] Can Python 3.11 ^(lenh "py -3.11"^). & exit /b 1 )

if exist venv (
  if /i "%~1"=="--force" (
    echo Xoa venv cu...
    rmdir /s /q venv
  ) else (
    echo [LOI] Da co venv. Chay "setup_env.bat --force" de tao lai.
    exit /b 1
  )
)

echo [0/4] Tao venv Python 3.11...
py -3.11 -m venv venv || exit /b 1
"%PY%" -m pip install --upgrade pip setuptools wheel || exit /b 1

echo [1/4] PyTorch + CUDA 12.8 ^(ho tro RTX 50^)...
"%PY%" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 || exit /b 1

echo [2/4] basicsr 1.4.2 ^(setup.py can torch nen tat build isolation^)...
"%PY%" -m pip install --no-build-isolation basicsr==1.4.2 || exit /b 1

echo [3/4] requirements.txt...
"%PY%" -m pip install -r requirements.txt || exit /b 1

echo [4/4] Va loi import torchvision.transforms.functional_tensor cua basicsr...
"%PY%" -c "import importlib.util as u, pathlib as p; f = p.Path(u.find_spec('basicsr').submodule_search_locations[0], 'data', 'degradations.py'); t = f.read_text(encoding='utf-8'); f.write_text(t.replace('torchvision.transforms.functional_tensor', 'torchvision.transforms.functional'), encoding='utf-8'); print('  da va', f)" || exit /b 1

echo.
echo Kiem tra:
"%PY%" -c "import torch; print('  torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
"%PY%" -m pip check
echo.
echo Xong. Chay demo bang run_demo.bat
