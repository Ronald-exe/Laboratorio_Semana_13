# Instrucciones para correr el HLS del softmax (Windows, Vitis/Vivado 2026)

Son 2 pasos: **HLS** (síntesis + cosimulación) y **bitstream**. No hay que editar nada.

## 0. Preparar

1. Descomprimir el ZIP en **`C:\lab13`**, de modo que quede `C:\lab13\FPGA\softmax_hls.cpp`.
   *(Ruta corta a propósito: Vivado falla en Windows con rutas largas.)*

## 1. HLS: C-sim + síntesis + cosimulación (≈ 30–60 min)

1. Menú Inicio → abrir **"Vitis 2026.x Command Prompt"**.
2. Correr:

```bat
cd C:\lab13\FPGA
vitis-run --mode hls --tcl run_hls.tcl
```

- Corre 8 proyectos: FP32 y FP16, cada uno con 4 variantes.
- **Si va lento**, cancelar con `Ctrl+C` y correrlo sin cosimulación:

```bat
set COSIM=0
vitis-run --mode hls --tcl run_hls.tcl
```

- Al terminar debe existir la carpeta `C:\lab13\FPGA\reportes\` con 8 subcarpetas (`fp32_v0` … `fp16_v3`).

## 2. Bitstream para la KV260 (≈ 20–30 min cada uno)

1. Menú Inicio → abrir **"Vivado 2026.x Tcl Shell"**.
2. FP32:

```tcl
cd C:/lab13/FPGA
source build_bitstream.tcl
```

3. FP16: **cerrar esa consola**, abrir otra "Vivado 2026.x Tcl Shell" y correr:

```tcl
cd C:/lab13/FPGA
set argv {fp16}
source build_bitstream.tcl
```

- Al terminar cada uno sale `LISTO: overlay/softmax_<prec>.bit ...`.
- Debe existir `C:\lab13\FPGA\overlay\` con `softmax_fp32.bit`, `softmax_fp32.hwh`, `softmax_fp16.bit`, `softmax_fp16.hwh` y los `.rpt`.

## 3. Devolver

Comprimir en un ZIP y enviar **estas dos carpetas**:

- `C:\lab13\FPGA\reportes\`
- `C:\lab13\FPGA\overlay\`

## Si no aparecen las consolas en el menú Inicio

Abrir un `cmd` normal y cargar el entorno primero (ajustar la versión/ruta a la instalada):

```bat
C:\AMDDesignTools\2026.1\Vitis\settings64.bat
```

o, en instalaciones viejas, `C:\Xilinx\Vitis\2026.1\settings64.bat`. Después ya funcionan `vitis-run` y `vivado`.
Para el bitstream desde ese `cmd`:

```bat
cd C:\lab13\FPGA
vivado -mode batch -source build_bitstream.tcl -tclargs fp32
vivado -mode batch -source build_bitstream.tcl -tclargs fp16
```

## Si algo falla

Mandar una captura del error (las últimas ~30 líneas de la consola) y el archivo `vitis_hls.log` o `vivado.log` que queda en `C:\lab13\FPGA\`.
