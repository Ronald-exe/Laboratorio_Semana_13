# Desglose de la GPU según nvprof (promedio, µs por llamada)

Actividades en la GPU (kernel, copias) y costo de las llamadas a la API vistas desde la CPU.

| Precisión | N | Kernel (GPU) | memcpy H→D | memcpy D→H | cudaLaunchKernel (API) | cudaMemcpy (API) | cudaDeviceSynchronize (API) |
|---|---|---|---|---|---|---|---|
| fp16 | 10 | 65.74 | 2.29 | 4.53 | 94.66 | 101.41 | 114.33 |
| fp16 | 100 | 66.08 | 2.58 | 4.90 | 94.90 | 101.58 | 108.04 |
| fp16 | 500 | 79.47 | 3.90 | 5.65 | 95.23 | 104.05 | 108.98 |
| fp16 | 1000 | 103.45 | 5.51 | 6.68 | 97.50 | 106.44 | 132.35 |
| fp32 | 10 | 64.99 | 2.32 | 4.57 | 90.69 | 102.97 | 109.43 |
| fp32 | 100 | 65.44 | 2.98 | 5.22 | 97.38 | 103.93 | 109.92 |
| fp32 | 500 | 78.47 | 5.51 | 6.68 | 95.42 | 103.89 | 109.86 |
| fp32 | 1000 | 101.46 | 9.30 | 8.56 | 94.04 | 105.54 | 137.05 |
