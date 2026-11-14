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

En la Tabla 1 se presenta el desglose del tiempo por llamada (µs, FP32) para el vector más pequeño y el más grande.

| Plataforma | Etapa | N=10 | N=1000 |
|---|---|---|---|
| CPU | Función completa | 0.50 | 43.93 |
| GPU | Kernel (`nvprof`) | 64.99 | 101.46 |
| GPU | Copias H→D + D→H (`nvprof`) | 6.89 | 17.86 |
| GPU | End-to-end | 352.59 | 436.78 |
| KV260 | IP (cosim, 100 MHz) | 1.41 | 28.65 |
| KV260 | Transferencias desde el host | 215.94 | 234.51 |
| KV260 | Total host | 240.95 | 289.41 |

*Tabla 1. Desglose del tiempo por llamada en FP32.*

En la CPU domina el cómputo, el cual crece lineal con N, mientras que en la GPU y en la KV260 domina el costo fijo de comunicarse con el acelerador (la API de CUDA y las transferencias desde Python), por lo cual su tiempo total casi no cambia con N y solo el cómputo del acelerador crece con el tamaño.

**2. ¿Qué diferencia de error aparece entre FP16 y FP32? ¿Se debe al formato, la exponencial o la reducción?**

En la Tabla 2 se presenta el error relativo máximo contra FP64.

| Plataforma | FP32, N=10 | FP32, N=1000 | FP16, N=10 | FP16, N=1000 |
|---|---|---|---|---|
| CPU | 2.04e-7 | 1.14e-6 | 1.69e-3 | 1.16e-2 |
| GPU | 3.15e-7 | 4.59e-7 | 1.69e-3 | 1.16e-2 |
| KV260 | 2.14e-7 | 4.47e-7 | 1.69e-3 | 1.00 |

*Tabla 2. Error relativo máximo contra FP64.*

Notamos que FP16 tiene un error unas 10⁴ veces mayor que FP32 y que la CPU y la GPU dan exactamente el mismo error en FP16, esto se debe a que el error viene del **formato**, ya que ambas redondean a 16 bits solo la entrada y la salida, y crece con N porque las salidas menores a 6.1e-5 quedan como subnormales con menos bits de mantisa. En la KV260 el error relativo llega a 1.0 porque el núcleo `half` de AMD aplana los subnormales a cero, o sea, sigue siendo el formato, mientras que la **reducción** solo aportó error cuando la suma se hacía en `half` (la variante 1 sumaba **1.0055** con N=1000), lo cual se eliminó calculando en FP32 en la variante 4.

**3. ¿Por qué la GPU puede ser peor que la CPU para vectores pequeños?**

En la Tabla 3 se compara la CPU con la GPU en FP16.

| N | CPU (función) | GPU (kernel, `nvprof`) | GPU (end-to-end) |
|---|---|---|---|
| 10 | 0.55 µs | 65.74 µs | 416.16 µs |
| 1000 | 48.70 µs | 103.45 µs | 441.37 µs |

*Tabla 3. CPU contra GPU en FP16.*

La GPU es peor porque paga un costo fijo de lanzamiento, copias y sincronización de unos 400 µs que no depende de N, y con 10 elementos casi todos los 256 hilos quedan ociosos, por lo cual con N=10 es unas 750 veces más lenta que la CPU, e incluso con N=1000 su kernel solo ya tarda más que la CPU completa.

**4. ¿Qué diferencia hay entre latencia HLS y tiempo total observado desde el host?**

En la Tabla 4 se presentan los tiempos de la KV260 en FP32.

| N | Latencia HLS (cosim) | IP medido desde Python | Total observado desde el host |
|---|---|---|---|
| 10 | 141 ciclos = 1.41 µs | 25.05 µs | 240.95 µs |
| 1000 | 2865 ciclos = 28.65 µs | 54.83 µs | 289.41 µs |

*Tabla 4. Latencia HLS contra tiempo observado desde el host (FP32).*

La latencia HLS son solo los ciclos del IP, mientras que el host además ve unos 24–26 µs de acceso a registros AXI-Lite y sondeo de `ap_done` desde Python y unos 216–235 µs de copia y sincronización de caché, por lo cual el IP representa menos del 10 % del tiempo total.

**5. ¿Qué optimización HLS mejoró más la latencia y cuánto aumentó el uso de recursos?**

En la Tabla 5 se presentan las variantes en FP32 con N=1000.

| Variante | Ciclos | Fmax | Tiempo a Fmax | LUT | DSP | BRAM 18K |
|---|---|---|---|---|---|---|
| 0: secuencial | 34023 | 137.0 MHz | 248 µs | 3663 | 9 | 4 |
| 1: + PIPELINE | 3058 | 38.9 MHz | 78.7 µs | 4046 | 11 | 4 |
| 2: + ARRAY_PARTITION | 2464 | 38.9 MHz | 63.4 µs | 18321 | 112 | 34 |
| 3: + DATAFLOW | 2468 | 38.9 MHz | 63.5 µs | 17482 | 112 | 34 |
| 4: propuesta | 2853 | 117.1 MHz | 24.4 µs | 9771 | 46 | 34 |

*Tabla 5. Latencia y recursos de las variantes HLS (FP32).*

La que más mejoró fue **PIPELINE**, que bajó los ciclos 11 veces con solo un 10 % más de LUT, mientras que ARRAY_PARTITION solo ganó 1.24 veces más a cambio de 4.5 veces más LUT y 10 veces más DSP, y DATAFLOW no cambió la latencia. Sin embargo, forzar II=1 bajó la Fmax de 137 a 38.9 MHz, por lo cual en tiempo real la mejora fue solo de 3.2 veces, lo cual motivó la propuesta de la variante 4.

**6. ¿Cuál plataforma minimiza tiempo y cuál energía por elemento? ¿Cambia la respuesta entre FP16 y FP32?**

En la Tabla 6 se presenta el tiempo total por llamada y la energía por elemento.

| Plataforma | Tiempo FP32 (N=10 / N=1000) | Tiempo FP16 (N=10 / N=1000) | Energía por elemento FP32 (N=10 / N=1000) | Energía por elemento FP16 (N=10 / N=1000) |
|---|---|---|---|---|
| CPU | 0.50 / 43.93 µs | 0.55 / 48.70 µs | N/D | N/D |
| GPU (end-to-end) | 352.59 / 436.78 µs | 416.16 / 441.37 µs | N/D | N/D |
| KV260 (total host) | 240.95 / 289.41 µs | 239.67 / 270.45 µs | 86.10 / 1.01 µJ | 84.58 / 0.94 µJ |

*Tabla 6. Tiempo total y energía por elemento (la Nano 2GB no tiene sensor de potencia).*

La **CPU** minimiza el tiempo para todos los N en ambas precisiones, y la única plataforma con energía medida es la **KV260**, cuya energía por elemento baja de unos 85 µJ con N=10 a menos de 1 µJ con N=1000, ya que el costo fijo por llamada se reparte entre más elementos. La respuesta no cambia entre FP16 y FP32, aunque en la KV260 FP16 es un 7 % mejor con N=1000, debido a que mueve la mitad de bytes por la interfaz.

## Propuesta de optimización (variante 4)

En la Tabla 7 se presenta la comparación antes y después de la propuesta.

| | Variante 3 (antes) | Variante 4 (después) | Cambio |
|---|---|---|---|
| II del bucle de la suma | 1 | 4 | — |
| Fmax estimada | 38.9 MHz | 117.1 MHz | 3.0 veces |
| Latencia (N=1000) | 2468 ciclos | 2853 ciclos | +16 % |
| Tiempo a Fmax (N=1000) | 63.5 µs | 24.4 µs | 2.6 veces más rápido |
| LUT / FF | 17482 / 14457 | 9771 / 9271 | −44 % / −36 % |
| DSP | 112 | 46 | −59 % |
| Timing a 100 MHz en Vivado | no cerraría | WNS = +3.78 ns | cumple |

*Tabla 7. Antes y después de la propuesta (FP32).*

La hipótesis fue que la suma `s += e`, al depender de la iteración anterior, obligaba a Vitis a encadenar el sumador flotante en un solo ciclo (13.08 ns contra 7.3 ns de presupuesto), y como el sumador de la KV260 tiene una latencia de 4 ciclos a 100 MHz, se usó **II=4** en ese bucle, manteniendo 2 elementos por ciclo porque cada iteración procesa 8. Un primer intento con II=2 solo llegó a 46.5 MHz, lo cual confirmó la hipótesis. Además, en FP16 se dejó solo la entrada y la salida en `half` y se calculó en FP32, eliminando el error de la reducción. Con esto concluimos que **el cambio sí mejoró el rendimiento**, ya que fue 2.6 veces más rápido con menos de la mitad de los DSP y permitió medir en hardware.

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
