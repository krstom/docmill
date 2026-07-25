@echo off
rem Windows twin of download_dependencies.sh: fetch docmill's runtime
rem dependencies into models\ and .pdfium\lib\ so a native (non-WSL) build
rem runs out of the box:
rem
rem     cargo build --release
rem     target\release\docmill document.docx
rem
rem Everything lands relative to the repo root (the binary resolves models\
rem and .pdfium\lib next to the CWD or the executable, no env vars needed).
rem Models come from the same sources the sh script uses; pdfium.dll comes
rem from pdfium-binaries (the docling.rs release only hosts the Linux .so).
rem The PP-OCRv5 det/rec conversion needs python + pip (paddle2onnx); when
rem python is missing it is skipped with a warning and the v3 fallback is
rem used — much weaker on screenshots.
rem
rem Usage: scripts\install\download_dependencies.bat [--no-pdf] [--no-tableformer] [--no-v5] [--force]
rem Needs curl.exe and tar.exe (both ship with Windows 10 1803+).
setlocal enabledelayedexpansion

set "BASE_URL=https://github.com/docling-project/docling.rs/releases/download/models-v1"
if not "%DOCLING_RS_MODELS_URL%"=="" set "BASE_URL=%DOCLING_RS_MODELS_URL%"
set "PADDLE_HF=https://huggingface.co/PaddlePaddle"
set "PDFIUM_URL=https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"

set WITH_PDF=1
set WITH_TF=1
set WITH_V5=1
set FORCE=0
:parse
if "%~1"=="" goto parsed
if "%~1"=="--no-pdf"         set WITH_PDF=0
if "%~1"=="--no-tableformer" set WITH_TF=0
if "%~1"=="--no-v5"          set WITH_V5=0
if "%~1"=="--force"          set FORCE=1
if "%~1"=="--help" (
  echo usage: download_dependencies.bat [--no-pdf] [--no-tableformer] [--no-v5] [--force]
  exit /b 0
)
shift
goto parse
:parsed

rem repo root = two levels above this script
cd /d "%~dp0..\.."
if not exist models mkdir models

if "%WITH_PDF%"=="0" goto ocr_v3

echo fetching PDF pipeline assets from %BASE_URL%
if not exist .pdfium\lib mkdir .pdfium\lib
if exist .pdfium\lib\pdfium.dll if "%FORCE%"=="0" (
  echo   = .pdfium\lib\pdfium.dll ^(already present^)
) else (
  echo   ^> .pdfium\lib\pdfium.dll
  curl -fsSL -o .pdfium\pdfium.tgz "%PDFIUM_URL%" || goto fail
  tar -xzf .pdfium\pdfium.tgz -C .pdfium bin/pdfium.dll || goto fail
  move /y .pdfium\bin\pdfium.dll .pdfium\lib\pdfium.dll >nul
  rmdir .pdfium\bin 2>nul
  del .pdfium\pdfium.tgz
)
call :fetch "%BASE_URL%/layout_heron.onnx" models\layout_heron.onnx || goto fail
call :fetch_opt "%BASE_URL%/layout_heron_int8.onnx" models\layout_heron_int8.onnx

if "%WITH_TF%"=="0" goto ocr_v3
if not exist models\tableformer mkdir models\tableformer
call :fetch "%BASE_URL%/encoder.onnx" models\tableformer\encoder.onnx || goto fail
call :fetch_opt "%BASE_URL%/encoder.onnx.data" models\tableformer\encoder.onnx.data
call :fetch "%BASE_URL%/decoder.onnx" models\tableformer\decoder.onnx || goto fail
call :fetch_opt "%BASE_URL%/decoder.onnx.data" models\tableformer\decoder.onnx.data
call :fetch_opt "%BASE_URL%/decoder_kv.onnx" models\tableformer\decoder_kv.onnx
call :fetch_opt "%BASE_URL%/decoder_kv.onnx.data" models\tableformer\decoder_kv.onnx.data
call :fetch "%BASE_URL%/bbox.onnx" models\tableformer\bbox.onnx || goto fail
call :fetch_opt "%BASE_URL%/bbox.onnx.data" models\tableformer\bbox.onnx.data

:ocr_v3
echo fetching PP-OCRv3 recognition pairs
call :fetch "%BASE_URL%/ocr_rec.onnx" models\ocr_rec.onnx || goto fail
call :fetch "%BASE_URL%/ppocr_keys_v1.txt" models\ppocr_keys_v1.txt || goto fail
call :fetch "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx" models\ocr_rec_en.onnx || goto fail
call :fetch "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt" models\en_dict.txt || goto fail

if "%WITH_V5%"=="0" goto done
if exist models\ppocrv5_mobile_det.onnx if exist models\ppocrv5_mobile_rec.onnx if "%FORCE%"=="0" (
  echo   = PP-OCRv5 det+rec ^(already present^)
  goto done
)
where python >nul 2>nul
if errorlevel 1 (
  echo warning: python not found -- skipping PP-OCRv5 conversion ^(v3 fallback still works;
  echo          picture OCR on screenshots will be much weaker without v5^) 1>&2
  goto done
)
echo fetching + converting PP-OCRv5 mobile det/rec from %PADDLE_HF%
where paddle2onnx >nul 2>nul || python -m pip install --user --quiet paddle2onnx
set "TMPDIR_V5=%TEMP%\docmill-v5-%RANDOM%"
for %%m in (PP-OCRv5_mobile_det PP-OCRv5_mobile_rec) do (
  mkdir "%TMPDIR_V5%\%%m" 2>nul
  for %%f in (inference.json inference.pdiparams inference.yml) do (
    curl -fsSL -o "%TMPDIR_V5%\%%m\%%f" "%PADDLE_HF%/%%m/resolve/main/%%f" || goto v5fail
  )
)
python -m paddle2onnx --model_dir "%TMPDIR_V5%\PP-OCRv5_mobile_det" --model_filename inference.json --params_filename inference.pdiparams --save_file models\ppocrv5_mobile_det.onnx --opset_version 14 >nul 2>nul || goto v5fail
python -m paddle2onnx --model_dir "%TMPDIR_V5%\PP-OCRv5_mobile_rec" --model_filename inference.json --params_filename inference.pdiparams --save_file models\ppocrv5_mobile_rec.onnx --opset_version 14 >nul 2>nul || goto v5fail
call :fetch "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt" models\ppocrv5_dict.txt || goto v5fail
echo   ^> models\ppocrv5_mobile_det.onnx
echo   ^> models\ppocrv5_mobile_rec.onnx
rmdir /s /q "%TMPDIR_V5%" 2>nul
goto done

:v5fail
del models\ppocrv5_mobile_det.onnx models\ppocrv5_mobile_rec.onnx 2>nul
rmdir /s /q "%TMPDIR_V5%" 2>nul
echo warning: PP-OCRv5 fetch/conversion failed -- the v3 fallback still works 1>&2
goto done

:fetch
if exist %2 if "%FORCE%"=="0" (
  echo   = %2 ^(already present^)
  exit /b 0
)
echo   ^> %2
curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 2 -o "%~2.download" %1 || exit /b 1
move /y "%~2.download" %2 >nul
exit /b 0

:fetch_opt
if exist %2 exit /b 0
curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 2 -o "%~2.download" %1 >nul 2>nul
if errorlevel 1 (
  del "%~2.download" 2>nul
  exit /b 0
)
move /y "%~2.download" %2 >nul
echo   ^> %2
exit /b 0

:fail
echo error: download failed 1>&2
exit /b 1

:done
echo done.
exit /b 0
