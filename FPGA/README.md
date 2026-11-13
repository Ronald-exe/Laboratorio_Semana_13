# FPGA: Softmax en HLS para la KV260

Softmax numéricamente estable (`y_i = exp(x_i - m) / Σ exp(x_j - m)`) en FP32 (`float`) y FP16 (`half`),
con la función superior que pide el enunciado:

```cpp
void softmax_hls(const data_t x[1000], data_t y[1000], int n);
```

## Archivos

| Archivo | Qué es |
|---|---|
| `softmax_hls.h` | `data_t` (FP32, o FP16 con `-DUSE_FP16`) y la variante (`-DVARIANTE=0..3`) |
| `softmax_hls.cpp` | Kernel: máximo → `hls::exp(x_i - m)` + suma → normalización |
| `softmax_tb.cpp` | Testbench (C-sim y cosim): misma entrada que CPU/GPU (splitmix64, semilla 12345), referencia FP64 |
| `run_hls.tcl` | C-sim + síntesis + cosim de las 4 variantes en FP32 y FP16; exporta el IP |
| `extraer_reportes.py` | Junta latencia, II, frecuencia y recursos en `resultados/tabla_hls.csv` |
| `build_bitstream.tcl` | Diseño de bloques en Vivado (PS + IP) → `overlay/softmax_<prec>.bit/.hwh` |
| `host_kv260.py` | Host PYNQ en la Kria: tiempo del IP, transferencias, total y energía (INA260) |

## Variantes (optimización incremental)

| Variante | Qué cambia |
|---|---|
| 0: secuencial | Tres bucles con `PIPELINE off` (Vitis hace pipeline automático; aquí lo apagamos para tener la línea base) |
| 1: + PIPELINE | `#pragma HLS PIPELINE II=1` en cada bucle. La suma queda limitada por la dependencia `s += e` |
| 2: + ARRAY_PARTITION | Copia local de `x` particionada `cyclic factor=8` + `UNROLL`: 8 exponenciales por iteración, árboles de máximo y suma, `1/s` una vez y luego multiplicación |
| 3: + DATAFLOW | La variante 2 dividida en procesos (`leer → maximo → exp_suma → normalizar → escribir`) con `#pragma HLS DATAFLOW` |

`x` y `y` viven en la DDR (interfaz `m_axi`), por eso no se particionan directamente: se particiona su copia local.

## 1. HLS (PC x86 con Vitis 2024.2 o posterior)

```bash
cd FPGA
vitis-run --mode hls --tcl run_hls.tcl        # Vitis HLS ≤ 2023.2:  vitis_hls -f run_hls.tcl
python3 extraer_reportes.py                   # -> resultados/tabla_hls.csv
```

Opciones: `COSIM=0` salta la cosim; `PRECS="fp32"` y `VARIANTES="2"` limitan qué se corre.
Ejemplo: `PRECS=fp32 VARIANTES="0 1" vitis-run --mode hls --tcl run_hls.tcl`.

Los reportes quedan en `reportes/<prec>_v<k>/` (`csynth.rpt`, `csynth.xml`, `cosim/`, `validacion_hls.csv`).

## 2. Bitstream (misma PC, Vivado)

```bash
vivado -mode batch -source build_bitstream.tcl -tclargs fp32
vivado -mode batch -source build_bitstream.tcl -tclargs fp16
```

Genera `overlay/softmax_fp32.bit` + `.hwh` (y lo mismo para fp16), más los reportes de utilización,
timing y potencia estimada de Vivado. Usa la variante 2 por defecto (`-tclargs fp32 3` usa la 3).

## 3. Medición en la KV260

```bash
# desde la PC
rsync -avz -e "ssh -J ecaslab" FPGA/host_kv260.py FPGA/overlay el5859@192.168.0.83:~/Ronald/FPGA/
# en la Kria
cd ~/Ronald/FPGA
sudo bash -c 'source /etc/profile.d/pynq_venv.sh && python3 host_kv260.py --bit overlay/softmax_fp32.bit --precision fp32'
sudo bash -c 'source /etc/profile.d/pynq_venv.sh && python3 host_kv260.py --bit overlay/softmax_fp16.bit --precision fp16'
```

Salida: `resultados/kv260_<prec>.csv` con 5 corridas por N (calentamiento descartado).

## Protocolo de medición

- **Entrada:** la misma de CPU y GPU (splitmix64, semilla 12345, `x ∈ [-4, 4]`). Referencia: FP64 con `std::exp` / `np.exp`.
- **Tiempos** desde el host con `perf_counter_ns`:
  - `t_transfer`: copia al buffer + `sync_to_device` + `sync_from_device` + lectura.
  - `t_ip`: `ap_start` → `ap_done`, incluye el sondeo AXI-Lite.
  - `t_total`: la suma de los dos anteriores.
  - La latencia pura del IP sale de la cosim (ciclos × 10 ns).
- **Energía:** sensor INA260 (U14) de la placa, leído por `/sys/class/hwmon/*/power1_input`, muestreado cada ~10 ms.
  - Mide **todo el SOM** (PS + PL + DDR), no solo el acelerador.
  - `E_llamada = P_prom · t_corrida / reps` y `E_elemento = E_llamada / N`.
  - Se reporta también la potencia en reposo y la energía dinámica (`P - P_reposo`).
- **Condiciones:** reloj del PL de 100 MHz, mismo bitstream por precisión, warm-up descartado, 5 corridas por N.
