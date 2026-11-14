@echo off
REM correr_fp16.bat - Doble clic. Solo FP16 variante 4: HLS (sin cosim) + bitstream + ZIP. ~30 min.
cd /d "%~dp0"
echo ===== Carpeta: %cd%
if not exist run_hls.tcl ( echo ERROR: este .bat debe estar dentro de la carpeta FPGA & pause & exit /b 1 )
call C:\AMDDesignTools\2026.1\Vitis\settings64.bat
if exist C:\AMDDesignTools\2026.1\Vivado\settings64.bat call C:\AMDDesignTools\2026.1\Vivado\settings64.bat

echo ===== [1/3] HLS FP16 v4 ... %time%
set PRECS=fp16
set VARIANTES=4
set COSIM=0
call vitis-run --mode hls --tcl run_hls.tcl > log_hls_fp16_v4.txt 2>&1
findstr /C:"Estimated Fmax" /C:"AVISO" log_hls_fp16_v4.txt

echo ===== [2/3] Bitstream FP16 (~25 min) ... %time%
if exist proy_fp16_v4\sol\impl\ip (
    call vivado -mode batch -nojournal -source build_bitstream.tcl -tclargs fp16 > log_vivado_fp16.txt 2>&1
) else ( echo No se exporto el IP FP16 v4 )

echo ===== [3/3] Empaquetando ... %time%
if not exist overlay mkdir overlay
powershell -NoProfile -Command "Compress-Archive -Force -Path reportes,overlay,log_hls_fp16_v4.txt,log_vivado_fp16.txt -DestinationPath fp16_para_ronald.zip" 2>nul
if not exist fp16_para_ronald.zip powershell -NoProfile -Command "Compress-Archive -Force -Path reportes,log_hls_fp16_v4.txt -DestinationPath fp16_para_ronald.zip"
echo.
echo ===== LISTO: enviar a Ronald %cd%\fp16_para_ronald.zip
pause
