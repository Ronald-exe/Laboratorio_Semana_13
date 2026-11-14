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
| KV260 | 10 | total host: 239.67 [238.26–242.13]<br>IP: 24.85 [24.82–25.46]<br>transferencias: 214.82 [213.41–217.20] | 3.435 | 845.757 | 84.57571 | 3.05e-04 |
| KV260 | 100 | total host: 240.46 [239.65–243.03]<br>IP: 24.85 [24.80–24.96]<br>transferencias: 215.61 [214.85–218.06] | 3.430 | 848.281 | 8.48281 | 5.77e-05 |
| KV260 | 500 | total host: 264.93 [262.96–266.69]<br>IP: 40.95 [40.90–41.05]<br>transferencias: 223.97 [222.06–225.66] | 3.435 | 934.816 | 1.86963 | 6.02e-05 |
| KV260 | 1000 | total host: 270.45 [268.60–271.79]<br>IP: 48.59 [48.56–48.64]<br>transferencias: 221.89 [219.99–223.19] | 3.384 | 940.733 | 0.94073 | 6.08e-05 |

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
| KV260 | 10 | total host: 240.95 [240.35–243.41]<br>IP: 25.05 [25.01–25.57]<br>transferencias: 215.93 [215.27–217.83] | 3.470 | 861.019 | 86.10191 | 1.13e-08 |
| KV260 | 100 | total host: 243.02 [242.86–243.55]<br>IP: 25.06 [24.98–25.11]<br>transferencias: 218.04 [217.75–218.49] | 3.470 | 867.793 | 8.67793 | 1.17e-08 |
| KV260 | 500 | total host: 262.30 [260.99–262.96]<br>IP: 41.05 [40.94–41.15]<br>transferencias: 221.16 [220.05–221.87] | 3.427 | 924.411 | 1.84882 | 2.52e-09 |
| KV260 | 1000 | total host: 289.41 [288.31–289.56]<br>IP: 54.83 [54.66–55.05]<br>transferencias: 234.51 [233.65–234.62] | 3.415 | 1013.609 | 1.01361 | 1.31e-09 |
