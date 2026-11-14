# Tablas comparativas

Tiempo: mediana [mín–máx] de las corridas, en µs por llamada. Energía: µJ por llamada; J/elem en µJ por elemento. **pendiente** = sin datos todavía; **N/D** = la plataforma no tiene cómo medirlo (la Jetson Nano 2GB no trae sensor de potencia).


## FP16

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | µJ/elem. | Error máx. |
|---|---|---|---|---|---|---|
| CPU | 10 | función: 0.55 [0.54–0.55] | N/D | N/D | N/D | 3.05e-04 |
| CPU | 100 | función: 4.93 [4.93–4.96] | N/D | N/D | N/D | 5.77e-05 |
| CPU | 500 | función: 24.36 [24.35–24.37] | N/D | N/D | N/D | 1.23e-05 |
| CPU | 1000 | función: 48.70 [48.64–49.31] | N/D | N/D | N/D | 7.75e-06 |
| GPU | 10 | kernel (nvprof, prom.): 65.74 [65.48–72.88]<br>kernel: 149.38 [142.97–203.38]<br>end-to-end: 416.16 [364.59–651.11] | N/D | N/D | N/D | 3.05e-04 |
| GPU | 100 | kernel (nvprof, prom.): 66.08 [65.79–75.64]<br>kernel: 149.38 [144.32–229.32]<br>end-to-end: 413.92 [360.95–515.17] | N/D | N/D | N/D | 5.77e-05 |
| GPU | 500 | kernel (nvprof, prom.): 79.47 [79.18–89.39]<br>kernel: 163.12 [154.95–447.92]<br>end-to-end: 423.29 [376.94–524.70] | N/D | N/D | N/D | 1.23e-05 |
| GPU | 1000 | kernel (nvprof, prom.): 103.45 [103.04–112.62]<br>kernel: 184.74 [162.81–1454.01]<br>end-to-end: 441.37 [342.72–916.27] | N/D | N/D | N/D | 7.75e-06 |
| KV260 | 10 | pendiente | pendiente | pendiente | pendiente | pendiente |
| KV260 | 100 | pendiente | pendiente | pendiente | pendiente | pendiente |
| KV260 | 500 | pendiente | pendiente | pendiente | pendiente | pendiente |
| KV260 | 1000 | pendiente | pendiente | pendiente | pendiente | pendiente |

## FP32

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | µJ/elem. | Error máx. |
|---|---|---|---|---|---|---|
| CPU | 10 | función: 0.50 [0.49–0.54] | N/D | N/D | N/D | 1.13e-08 |
| CPU | 100 | función: 4.41 [4.41–4.42] | N/D | N/D | N/D | 1.63e-08 |
| CPU | 500 | función: 21.94 [21.94–21.97] | N/D | N/D | N/D | 9.97e-09 |
| CPU | 1000 | función: 43.93 [43.89–47.38] | N/D | N/D | N/D | 6.37e-09 |
| GPU | 10 | kernel (nvprof, prom.): 64.99 [64.64–73.97]<br>kernel: 124.43 [106.88–182.81]<br>end-to-end: 352.59 [289.44–1393.63] | N/D | N/D | N/D | 7.09e-08 |
| GPU | 100 | kernel (nvprof, prom.): 65.44 [65.17–74.02]<br>kernel: 144.48 [134.11–236.09]<br>end-to-end: 400.06 [348.29–597.93] | N/D | N/D | N/D | 1.08e-08 |
| GPU | 500 | kernel (nvprof, prom.): 78.47 [78.19–88.08]<br>kernel: 136.67 [120.83–273.12]<br>end-to-end: 367.87 [311.67–611.06] | N/D | N/D | N/D | 2.17e-09 |
| GPU | 1000 | kernel (nvprof, prom.): 101.46 [100.90–110.85]<br>kernel: 181.07 [141.46–236.77]<br>end-to-end: 436.78 [353.29–681.11] | N/D | N/D | N/D | 1.71e-09 |
| KV260 | 10 | pendiente | pendiente | pendiente | pendiente | pendiente |
| KV260 | 100 | pendiente | pendiente | pendiente | pendiente | pendiente |
| KV260 | 500 | pendiente | pendiente | pendiente | pendiente | pendiente |
| KV260 | 1000 | pendiente | pendiente | pendiente | pendiente | pendiente |
