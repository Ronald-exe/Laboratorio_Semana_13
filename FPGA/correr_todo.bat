@echo off
REM correr_todo.bat - Doble clic y listo. Corre todo lo que falta del flujo FPGA:
REM   1) HLS FP32 (variantes 2, 3 y 4)      2) bitstream FP32
REM   3) HLS FP16 (variantes 2, 3 y 4)      4) bitstream FP16 (solo si el IP se exporto)
REM   5) empaqueta reportes, overlay y logs en resultados_para_ronald.zip
REM Debe estar dentro de la carpeta FPGA (junto a run_hls.tcl). Tarda ~1 hora; no cerrar la ventana.

cd /d "%~dp0"
echo ===== Carpeta: %cd%
if not exist run_hls.tcl ( echo ERROR: este .bat debe estar dentro de la carpeta FPGA & pause & exit /b 1 )

call C:\AMDDesignTools\2026.1\Vitis\settings64.bat
if exist C:\AMDDesignTools\2026.1\Vivado\settings64.bat call C:\AMDDesignTools\2026.1\Vivado\settings64.bat

echo ===== [1/5] HLS FP32 (v2, v3, v4) ... %time%
set PRECS=fp32
set VARIANTES=2 3 4
call vitis-run --mode hls --tcl run_hls.tcl > log_hls_fp32.txt 2>&1
findstr /C:"Estimated Fmax" /C:"AVISO" log_hls_fp32.txt

echo ===== [2/5] Bitstream FP32 (~25 min) ... %time%
call vivado -mode batch -nojournal -source build_bitstream.tcl -tclargs fp32 > log_vivado_fp32.txt 2>&1
if exist overlay\softmax_fp32.bit ( echo OK: overlay\softmax_fp32.bit ) else ( echo FALLO el bitstream FP32, ver log_vivado_fp32.txt )

echo ===== [3/5] HLS FP16 (v2, v3, v4) ... %time%
set PRECS=fp16
set VARIANTES=2 3 4
call vitis-run --mode hls --tcl run_hls.tcl > log_hls_fp16.txt 2>&1
findstr /C:"Estimated Fmax" /C:"AVISO" log_hls_fp16.txt

echo ===== [4/5] Bitstream FP16 ... %time%
if exist proy_fp16_v4\sol\impl\ip (
    call vivado -mode batch -nojournal -source build_bitstream.tcl -tclargs fp16 > log_vivado_fp16.txt 2>&1
) else (
    echo No se exporto el IP FP16 v4, se salta el bitstream FP16
)

echo ===== [5/5] Empaquetando ... %time%
powershell -NoProfile -Command "Compress-Archive -Force -Path reportes,overlay,log_*.txt -DestinationPath resultados_para_ronald.zip"
echo.
echo ===== LISTO: enviar a Ronald el archivo %cd%\resultados_para_ronald.zip
pause
