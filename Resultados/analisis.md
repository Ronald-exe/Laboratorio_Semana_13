# Análisis de resultados: Softmax en CPU, GPU y FPGA

> Borrador de la sección RO. Las partes marcadas **[PENDIENTE]** se completan cuando lleguen los
> datos de la GPU (FA) y de la KV260 (síntesis + mediciones físicas). No se llenan con estimaciones.

## Condiciones de medición

| Plataforma | Equipo | Configuración | Medición de energía |
|---|---|---|---|
| CPU | Jetson Nano 2GB (Cortex-A57), `jetson-181` | Modo 5W (modo 1), 2 núcleos activos (0-1), governor `schedutil` a 921.6 MHz, g++ 7.5 `-O3 -march=native` | **N/D**: la Nano 2GB no trae sensor INA3221, `tegrastats` no reporta `POM_5V_IN` |
| GPU | Jetson Nano 2GB (Maxwell 128 núcleos) | Mismo modo 5W, CUDA 10.2, frecuencia de GPU dinámica (máx. 614.4 MHz), kernel de 1 bloque × 256 hilos con reducciones en memoria compartida, 1000 mediciones por N, `nvprof` (con `sudo`) | **N/D**: misma razón |
| KV260 | Kria KV260 `kria-85` (Zynq UltraScale+ xck26) | PL a 100 MHz, PYNQ 3.0.1, Ubuntu 22.04 | INA260 (U14) por `/sys/class/hwmon`, potencia de **todo el SOM** (PS + PL + DDR), muestreo ~10 ms |

En todas las plataformas se usó la misma entrada: generador `splitmix64`, semilla 12345, `x ∈ [-4, 4]`, y la referencia FP64 con `std::exp`. Se hizo una corrida de calentamiento, 5 mediciones por configuración, y se reporta mediana, mínimo y máximo.

## Resultados

Las tablas completas están en `tablas_comparativas.md` (generadas con `python3 Resultados/consolidar.py`) y las gráficas en `graficas/`.

En la CPU el tiempo de la función crece de forma lineal con N, aproximadamente **0.049 µs por elemento** en FP16 y **0.044 µs por elemento** en FP32, pasando de **0.55 µs** con N=10 a **48.70 µs** con N=1000 en FP16, y de **0.50 µs** a **43.92 µs** en FP32. Notamos que FP16 es alrededor de un **10 % más lento** que FP32, esto se debe a que la Cortex-A57 no tiene aritmética FP16, solo instrucciones de conversión, por lo cual cada elemento se convierte a FP32 para la exponencial y la suma y luego se redondea de vuelta a 16 bits, agregando trabajo sin ahorrar cómputo.

En la GPU el comportamiento es el opuesto, ya que el tiempo casi no cambia con N. Según `nvprof` el kernel tarda **65.7 µs** con N=10 y **103.5 µs** con N=1000 en FP16, mientras que el tiempo end-to-end se mantiene cerca de **415–440 µs** para todos los tamaños. Notamos que el kernel medido con `cudaEvent` (**149 µs** con N=10) es más del doble que el reportado por el perfilador, esto se debe a que el evento incluye el costo de lanzar el kernel, el cual según `nvprof` es de **~95 µs** por `cudaLaunchKernel`. Las copias de memoria son pequeñas (**2–9 µs**), por lo cual el costo de la GPU está dominado por lanzamiento, sincronización y llamadas a la API, y no por el cómputo ni por los datos (ver `perfil_gpu_nvprof.md`).

En cuanto a la exactitud, la Tabla de FP32 muestra errores absolutos del orden de **1e-8**, mientras que en FP16 están entre **3e-4 y 8e-6**. El error absoluto baja con N solamente porque cada salida vale aproximadamente 1/N, por eso es más representativo el error relativo, el cual en FP16 sube de **1.7e-3** (N=10) a **1.2e-2** (N=1000).

## Preguntas

**1. ¿Qué etapa domina el tiempo en cada plataforma y cómo cambia con N?**

En la CPU domina el cómputo de la función (exponencial y suma), creciendo lineal con N. En la GPU domina el costo fijo de la API: `cudaLaunchKernel` (~95 µs), `cudaMemcpy` (~100 µs) y la sincronización (~110 µs), contra un kernel de 66–103 µs y copias de 2–9 µs, por lo cual el end-to-end casi no cambia con N y solo el kernel crece un poco al pasar de 500 a 1000. En la KV260 domina el tiempo del host (sincronización de caché y accesos AXI-Lite desde Python) sobre la latencia del IP **[PENDIENTE: t_ip vs t_transfer vs latencia de cosim]**.

**2. ¿Qué diferencia de error aparece entre FP16 y FP32? ¿Se debe al formato, la exponencial o la reducción?**

FP16 tiene un error relativo alrededor de 10⁴ veces mayor que FP32. En la CPU se debe al **formato**, ya que la exponencial y la suma se hacen en FP32 y solo la entrada y la salida se redondean a 16 bits. Además, con N=1000 varias salidas son menores a 6.1e-5 (el menor normal de FP16) y quedan como subnormales con menos bits de mantisa, por eso el error relativo crece con N. En el kernel HLS, donde todo se calcula en `half`, la **reducción** también aporta error: la suma secuencial acumula redondeo y el árbol de sumas de la variante 2 lo reduce **[PENDIENTE: confirmar con la C-sim de Vitis]**.

**3. ¿Por qué la GPU puede ser peor que la CPU para vectores pequeños?**

Porque con N=10 la CPU termina en **0.55 µs** y la GPU tarda **416 µs** end-to-end (unas 750 veces más), ya que paga un costo fijo de lanzamiento, copias y sincronización que no depende de N, y con 10 elementos casi todos los 256 hilos del bloque quedan ociosos. Incluso con N=1000 la CPU (**48.7 µs**) sigue siendo más rápida que el kernel solo (**103.5 µs**), ya que se usa un único bloque y la GPU de 5W trabaja a baja frecuencia.

**4. ¿Qué diferencia hay entre latencia HLS y tiempo total observado desde el host?**

La latencia HLS son solo los ciclos del IP desde `ap_start` hasta `ap_done` (ciclos × 10 ns). El tiempo del host además incluye copiar la entrada al buffer, la sincronización de caché (`sync_to_device`/`sync_from_device`), las escrituras y lecturas de registros AXI-Lite y el sondeo de `ap_done` desde Python **[PENDIENTE: cuantificar con la KV260]**.

**5. ¿Qué optimización HLS mejoró más la latencia y cuánto aumentó el uso de recursos?**

**[PENDIENTE: latencias y recursos de tabla_hls.csv]**. La variante 2 (ARRAY_PARTITION cyclic 8 + UNROLL) es la que más debería reducir la latencia en ciclos, porque procesa 8 elementos por iteración, a cambio de replicar 8 unidades `hls::exp` (más DSP y LUT). Sin embargo, notamos en la síntesis que forzar `PIPELINE II=1` en FP32 bajó la frecuencia estimada de **137 MHz** (variante 0) a **38.9 MHz** (variantes 1–3), esto se debe a que la suma `s += e` depende de la iteración anterior y Vitis encadenó el sumador flotante en un solo ciclo (13.08 ns contra 7.3 ns de presupuesto a 100 MHz). Por eso la ganancia en ciclos no se traduce en tiempo real, lo cual motivó la propuesta de la variante 4.

### Propuesta de optimización (variante 4)

- **Hipótesis:** la reducción flotante con II=1 es la que limita la frecuencia. El sumador flotante de la KV260 tiene latencia de **4 ciclos** a 100 MHz (núcleo `faddfsub_32ns_32ns_32_4_full_dsp`), por lo cual la suma `s += …` necesita **II=4**. Como cada iteración procesa 8 elementos, se mantiene un throughput de 2 elementos por ciclo. Un primer intento con **II=2** solo subió la Fmax estimada de **38.9 a 46.5 MHz** (combinacional de 10.97 ns), lo cual confirmó la hipótesis de que el límite es la latencia del sumador.
- **Cambio adicional en FP16:** la entrada y la salida siguen en `half`, pero máximo, exponencial, suma y normalización se hacen en FP32. En la variante 2 con todo en `half`, las salidas muy pequeñas se volvían 0 (error relativo de 1.0), y además, en Vitis 2026.1 sobre Windows, la generación de la cosim falla para todos los diseños con puertos `half` (`extra characters after close-quote`), mientras que la C-sim y la síntesis sí pasan.
- **Antes / después:** **[PENDIENTE: Fmax, latencia y recursos de v3 contra v4]**. Con g++ la variante 4 en FP16 da exactamente el mismo error que la CPU (error absoluto máximo de **7.75e-6** con N=1000), ya que ambas calculan internamente en FP32.

**6. ¿Cuál plataforma minimiza tiempo y cuál energía por elemento? ¿Cambia la respuesta entre FP16 y FP32?**

Entre CPU y GPU, la **CPU** minimiza el tiempo para todos los N y en ambas precisiones (por ejemplo **48.7 µs** contra **441 µs** end-to-end con N=1000 en FP16), y la respuesta no cambia entre FP16 y FP32 **[PENDIENTE: comparar con el total host de la KV260]**. En la Jetson Nano 2GB la energía no se puede medir con `tegrastats` por falta de sensor, por lo cual la comparación de energía queda limitada a la KV260 o a un medidor externo.

## Limitaciones

- La Jetson Nano 2GB no tiene sensor de potencia; no se reporta energía de CPU ni de GPU en vez de estimarla.
- La potencia de la KV260 es la del SOM completo (en reposo marca **~3.1 W**, medido con el INA260 a ~96 muestras/s); se reporta además la energía dinámica (`P - P_reposo`).
- El `entorno.txt` de la CPU no registró `nvpmodel` porque `sudo` no tenía terminal; el modo (5W, modo 1) se verificó aparte con `nvpmodel -q`.

## Conclusiones

- **[PENDIENTE: se completan con todas las plataformas]**
- En la CPU de la Nano, FP16 no acelera softmax, porque no hay unidades FP16 y la conversión agrega costo.
- Para N ≤ 1000 la GPU de la Nano es más lenta que su CPU, porque el costo fijo de la API de CUDA (~400 µs end-to-end) domina sobre el cómputo.
- El tiempo del kernel medido con `cudaEvent` sobrestima al del perfilador, porque incluye el lanzamiento.
- El error de FP16 crece con N por las salidas subnormales, no solo por la mantisa de 10 bits.
