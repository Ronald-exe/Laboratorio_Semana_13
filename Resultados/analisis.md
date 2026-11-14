# Análisis de resultados: Softmax en CPU, GPU y FPGA

## Condiciones de medición

| Plataforma | Equipo | Configuración | Medición de energía |
|---|---|---|---|
| CPU | Jetson Nano 2GB (Cortex-A57), `jetson-181` | Modo 5W (modo 1), 2 núcleos activos (0-1), governor `schedutil` a 921.6 MHz, g++ 7.5 `-O3 -march=native` | **N/D**: la Nano 2GB no trae sensor INA3221, `tegrastats` no reporta `POM_5V_IN` |
| GPU | Jetson Nano 2GB (Maxwell 128 núcleos) | Mismo modo 5W, CUDA 10.2, frecuencia de GPU dinámica (máx. 614.4 MHz), kernel de 1 bloque × 256 hilos con reducciones en memoria compartida, 1000 mediciones por N, `nvprof` (con `sudo`) | **N/D**: misma razón |
| KV260 | Kria KV260 `kria-85` (Zynq UltraScale+ xck26) | IP variante 4 a 100 MHz (WNS = +3.78 ns en FP32 y +3.65 ns en FP16), PYNQ 3.0.1, Ubuntu 22.04, host en Python, 5 corridas × 2000 llamadas por N, temperatura 34–35 °C | INA260 (U14) por `/sys/class/hwmon`, potencia de **todo el SOM** (PS + PL + DDR), ~96 muestras/s; reposo **3.13–3.15 W** |

En todas las plataformas se usó la misma entrada: generador `splitmix64`, semilla 12345, `x ∈ [-4, 4]`, y la referencia FP64 con `std::exp`. Se hizo una corrida de calentamiento, al menos 5 mediciones por configuración, y se reporta mediana, mínimo y máximo.

## Resultados

Las tablas completas están en `tablas_comparativas.md` (generadas con `python3 Resultados/consolidar.py`), el desglose de la GPU en `perfil_gpu_nvprof.md`, la síntesis HLS en `FPGA/resultados/tabla_hls.csv` y las gráficas en `graficas/`.

En la CPU el tiempo de la función crece de forma lineal con N, aproximadamente **0.049 µs por elemento** en FP16 y **0.044 µs por elemento** en FP32, pasando de **0.55 µs** con N=10 a **48.70 µs** con N=1000 en FP16, y de **0.50 µs** a **43.92 µs** en FP32. Notamos que FP16 es alrededor de un **10 % más lento** que FP32, esto se debe a que la Cortex-A57 no tiene aritmética FP16, solo instrucciones de conversión, por lo cual cada elemento se convierte a FP32 para la exponencial y la suma y luego se redondea de vuelta a 16 bits, agregando trabajo sin ahorrar cómputo.

En la GPU el comportamiento es el opuesto, ya que el tiempo casi no cambia con N. Según `nvprof` el kernel tarda **65.7 µs** con N=10 y **103.5 µs** con N=1000 en FP16, mientras que el tiempo end-to-end se mantiene cerca de **415–440 µs** para todos los tamaños. Notamos que el kernel medido con `cudaEvent` (**149 µs** con N=10) es más del doble que el reportado por el perfilador, esto se debe a que el evento incluye el costo de lanzar el kernel, el cual según `nvprof` es de **~95 µs** por `cudaLaunchKernel`. Las copias de memoria son pequeñas (**2–9 µs**), por lo cual el costo de la GPU está dominado por lanzamiento, sincronización y llamadas a la API, y no por el cómputo ni por los datos.

En la KV260 (FP32, variante 4) el tiempo total observado desde el host va de **241 µs** con N=10 a **289 µs** con N=1000, del cual el tiempo del IP medido desde Python es **25.1 µs** y **54.8 µs** respectivamente, y el resto (**216–235 µs**) corresponde a las transferencias: copiar el vector al buffer DMA y sincronizar la caché (`sync_to_device`/`sync_from_device`). Según la cosimulación, la latencia pura del IP es de **141 ciclos (1.4 µs)** con N=10 y **2865 ciclos (28.7 µs)** con N=1000 a 100 MHz, por lo cual unos **24–26 µs** del tiempo del IP son costo de acceder a los registros AXI-Lite y sondear `ap_done` desde Python. La placa consume **3.47 W** durante las llamadas contra **3.15 W** en reposo, lo cual da **861 µJ** por llamada con N=10 y **1014 µJ** con N=1000 (de ellos solo **~75–79 µJ** son energía dinámica por encima del reposo). En FP16 el comportamiento es casi igual: **240 µs** con N=10 y **270 µs** con N=1000 de tiempo total, con el IP en **24.9–48.6 µs**, y **846–941 µJ** por llamada.

En cuanto a la exactitud, en FP32 todas las plataformas tienen errores absolutos del orden de **1e-8 a 1e-9**, mientras que en FP16 están entre **3e-4 y 8e-6**. El error absoluto baja con N solamente porque cada salida vale aproximadamente 1/N, por eso es más representativo el error relativo, el cual en FP16 sube de **1.7e-3** (N=10) a **1.2e-2** (N=1000).

### Síntesis HLS (FP32, reloj objetivo 10 ns)

| Variante | Latencia máx. (ciclos, N=1000) | Fmax estimada | Tiempo a Fmax (N=1000) | LUT | FF | DSP | BRAM 18K | Cosim |
|---|---|---|---|---|---|---|---|---|
| 0: secuencial | 34023 | 137.0 MHz | 248 µs | 3663 | 2829 | 9 | 4 | PASS |
| 1: + PIPELINE II=1 | 3058 | 38.9 MHz | 78.7 µs | 4046 | 3219 | 11 | 4 | PASS |
| 2: + ARRAY_PARTITION + UNROLL | 2464 | 38.9 MHz | 63.4 µs | 18321 | 14141 | 112 | 34 | PASS |
| 3: + DATAFLOW | 2468 | 38.9 MHz | 63.5 µs | 17482 | 14457 | 112 | 34 | PASS |
| **4: propuesta (II=4 en la suma)** | **2853** | **117.1 MHz** | **24.4 µs** | **9771** | **9271** | **46** | **34** | **PASS** |

En FP16 (todo en `half`), la variante 0 da 18023 ciclos a 137 MHz con 6 DSP y la variante 1 da 3044 ciclos a 81.7 MHz con 6 DSP, o sea, la mitad de DSP que en FP32, ya que los núcleos de 16 bits son más pequeños. La variante 4 en FP16 da **2855 ciclos a 117.1 MHz** con 10010 LUT y 46 DSP, prácticamente lo mismo que en FP32, ya que internamente calcula en FP32 y solo la entrada y la salida son `half`.

## Preguntas

**1. ¿Qué etapa domina el tiempo en cada plataforma y cómo cambia con N?**

En la CPU domina el cómputo de la función (exponencial y suma), creciendo lineal con N. En la GPU domina el costo fijo de la API: `cudaLaunchKernel` (~95 µs), `cudaMemcpy` (~100 µs) y la sincronización (~110 µs), contra un kernel de 66–103 µs y copias de 2–9 µs, por lo cual el end-to-end casi no cambia con N. En la KV260 dominan las transferencias desde el host (**81–90 %** del total, 216–235 µs), mientras que el IP crece de 1.4 a 28.7 µs con N y es la única parte que escala con el tamaño.

**2. ¿Qué diferencia de error aparece entre FP16 y FP32? ¿Se debe al formato, la exponencial o la reducción?**

FP16 tiene un error relativo alrededor de 10⁴ veces mayor que FP32. En la CPU y la GPU se debe al **formato**, ya que solo la entrada y la salida se redondean a 16 bits, y con N=1000 varias salidas son menores a 6.1e-5 (el menor normal de FP16) y pierden bits de mantisa, por eso el error relativo crece con N. En el kernel HLS con todo en `half` también aporta la **reducción**: con la suma secuencial la salida suma **1.0055** con N=1000 (contra 0.99999 en la CPU), y además las salidas muy pequeñas se vuelven 0 (error relativo de 1.0). Esto último se mantiene incluso en la variante 4 medida en la KV260, donde el cómputo es en FP32: con N ≥ 100 el error relativo máximo es **1.0** y la suma cae a **0.9929** con N=1000, y el error absoluto máximo (**6.08e-5**) queda justo debajo de **6.1e-5**, el menor normal de FP16. Esto se debe a que el núcleo de conversión a `half` de AMD **aplana los subnormales a cero** (*flush-to-zero*), mientras que el `__fp16` de la CPU sí los conserva (error de **7.7e-6**), por lo cual en la FPGA la diferencia se debe al **formato** y no a la exponencial.

**3. ¿Por qué la GPU puede ser peor que la CPU para vectores pequeños?**

Porque con N=10 la CPU termina en **0.55 µs** y la GPU tarda **416 µs** end-to-end (unas 750 veces más), ya que paga un costo fijo de lanzamiento, copias y sincronización que no depende de N, y con 10 elementos casi todos los 256 hilos del bloque quedan ociosos. Incluso con N=1000 la CPU (**48.7 µs**) sigue siendo más rápida que el kernel solo (**103.5 µs**), ya que se usa un único bloque y la GPU de 5W trabaja a baja frecuencia.

**4. ¿Qué diferencia hay entre latencia HLS y tiempo total observado desde el host?**

La latencia HLS son solo los ciclos del IP (**1.4 µs** con N=10 y **28.7 µs** con N=1000 a 100 MHz), mientras que el host observa **241–289 µs**, ya que además incluye copiar el vector al buffer, sincronizar la caché (**216–235 µs**) y acceder a los registros AXI-Lite y sondear `ap_done` desde Python (**~25 µs**).

**5. ¿Qué optimización HLS mejoró más la latencia y cuánto aumentó el uso de recursos?**

En ciclos, la que más mejoró fue **PIPELINE**: bajó de 34023 a 3058 ciclos (**11 veces menos**) aumentando solo un 10 % los LUT (3663 → 4046) y de 9 a 11 DSP, mientras que ARRAY_PARTITION + UNROLL solo bajó a 2464 ciclos (1.24 veces más) a cambio de **4.5 veces** más LUT, **10 veces** más DSP (112) y 8.5 veces más BRAM, y DATAFLOW no cambió la latencia, ya que el máximo tiene que terminar antes de empezar la exponencial. Sin embargo, forzar II=1 bajó la Fmax de **137 a 38.9 MHz**, por lo cual en tiempo real la mejora fue de 248 a 78.7 µs (3.2 veces) y no de 11 veces, lo cual motivó la propuesta.

### Propuesta de optimización (variante 4)

- **Hipótesis:** la reducción flotante con II=1 es la que limita la frecuencia. La suma `s += …` depende de la iteración anterior y Vitis encadenó el sumador en un solo ciclo (13.08 ns contra 7.3 ns de presupuesto a 100 MHz). El sumador flotante de la KV260 tiene latencia de **4 ciclos** a 100 MHz (núcleo `faddfsub_32ns_32ns_32_4_full_dsp`), por lo cual la suma necesita **II=4**; como cada iteración procesa 8 elementos, se mantiene un throughput de 2 elementos por ciclo.
- **Verificación intermedia:** un primer intento con **II=2** solo subió la Fmax de **38.9 a 46.5 MHz** (combinacional de 10.97 ns), lo cual confirmó que el límite es la latencia del sumador.
- **Cambio adicional en FP16:** la entrada y la salida siguen en `half`, pero máximo, exponencial, suma y normalización se hacen en FP32, igual que en la CPU. Esto quitó el error de la reducción (la suma secuencial en `half` daba 1.0055), pero en hardware las salidas menores a 6.1e-5 siguen saliendo en 0 por el *flush-to-zero* de la conversión a `half`.
- **Antes / después (variante 3 → 4, FP32):** la Fmax sube de **38.9 a 117.1 MHz** (3 veces), la latencia pasa de 2468 a 2853 ciclos (+16 %), pero el tiempo a la Fmax baja de **63.5 a 24.4 µs (2.6 veces más rápido)**, y además usa **menos** recursos: LUT de 17482 a 9771 (−44 %), FF de 14457 a 9271 (−36 %) y DSP de 112 a 46 (−59 %), ya que con II=4 Vitis puede compartir unidades de cómputo entre ciclos. La cosim pasa y el bitstream cierra timing a 100 MHz con **WNS = +3.78 ns**, por lo cual **el cambio sí mejoró el rendimiento** y además fue el que permitió medir en hardware.

**6. ¿Cuál plataforma minimiza tiempo y cuál energía por elemento? ¿Cambia la respuesta entre FP16 y FP32?**

En tiempo la **CPU** gana para todos los N (**0.50–43.9 µs** en FP32), seguida por la KV260 (**241–289 µs** total host) y por último la GPU (**353–437 µs** end-to-end). Si se compara solo el cómputo, el IP de la KV260 (**28.7 µs** por cosim con N=1000) ya es más rápido que la CPU (**43.9 µs**), pero el costo de las transferencias desde Python lo opaca. En energía solo se pudo medir la KV260 (**1.01 µJ por elemento** con N=1000 y **86 µJ** con N=10 en FP32, porque el costo fijo por llamada se reparte entre más elementos), ya que la Nano 2GB no tiene sensor. La respuesta no cambia entre FP16 y FP32: la CPU sigue siendo la más rápida, aunque en la KV260 FP16 es algo mejor que FP32 con N=1000 (**270 contra 289 µs** y **0.94 contra 1.01 µJ por elemento**), ya que mueve la mitad de bytes por la interfaz.

## Limitaciones

- La Jetson Nano 2GB no tiene sensor de potencia; no se reporta energía de CPU ni de GPU en vez de estimarla.
- La potencia de la KV260 es la del SOM completo; se reporta además la energía dinámica (`P - P_reposo`), que es de solo ~75–79 µJ por llamada.
- Los núcleos `half` de AMD aplanan los subnormales a cero, por lo cual en la KV260 las salidas FP16 menores a 6.1e-5 salen en 0.
- La cosimulación de los diseños con puertos `half` falla en Vitis 2026.1 sobre Windows al generar el testbench RTL (`extra characters after close-quote`), por lo cual FP16 se validó con C-sim y síntesis.
- El tiempo del IP medido desde Python incluye ~25 µs de acceso a registros; la latencia pura del IP se toma de la cosimulación.
- El `entorno.txt` de la CPU no registró `nvpmodel` porque `sudo` no tenía terminal; el modo (5W, modo 1) se verificó aparte con `nvpmodel -q`.

## Conclusiones

- En la CPU de la Nano, FP16 no acelera softmax, porque no hay unidades FP16 y la conversión agrega costo.
- Para N ≤ 1000 la GPU de la Nano es más lenta que su CPU, porque el costo fijo de la API de CUDA (~400 µs end-to-end) domina sobre el cómputo.
- El tiempo del kernel medido con `cudaEvent` sobrestima al del perfilador, porque incluye el lanzamiento.
- En la KV260 el IP ya supera a la CPU en cómputo para N=1000, pero las transferencias desde el host dominan el tiempo total.
- Forzar `PIPELINE II=1` en una reducción flotante destruye la frecuencia; ajustar el II a la latencia del sumador (II=4) dio 2.6 veces más velocidad con 59 % menos DSP.
- El error de FP16 crece con N por las salidas subnormales; en la CPU se conservan, pero en la FPGA el *flush-to-zero* de la conversión a `half` las vuelve 0.
- En la KV260, FP16 ahorra un 7 % de tiempo y energía por elemento con N=1000 frente a FP32, ya que la interfaz mueve la mitad de bytes.
