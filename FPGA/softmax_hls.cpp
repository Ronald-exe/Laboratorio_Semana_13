// softmax_hls.cpp - Softmax numericamente estable en HLS para la KV260
//
//   m   = max_i x_i
//   y_i = exp(x_i - m) / sum_j exp(x_j - m)
//
// Restamos el maximo para que exp() nunca reciba algo positivo: asi exp(x_i - m) <= 1
// y no hay desbordamiento (en FP16 el maximo representable es 65504, exp(11) ya se pasa).
//
// Las 4 variantes (VARIANTE = 0..3) hacen EXACTAMENTE la misma cuenta; solo cambian las
// directivas, para poder medir que aporta cada optimizacion por separado.

#include "softmax_hls.h"

#if __has_include("hls_math.h")
  // En el kernel sintetizable NO se usa std::exp: hls::exp tiene una implementacion en hardware
  #include "hls_math.h"
  template <typename T> static inline T exp_t(T v) { return hls::exp(v); }
  #define EXP(v) exp_t(v)
#else
  // Solo para compilar el testbench con g++ fuera de Vitis
  #include <cmath>
  template <typename T> static inline T exp_t(T v) { return (T)std::exp((float)v); }
  #define EXP(v) exp_t(v)
#endif

#if VARIANTE <= 1
// =====================================================================================
// VARIANTES 0 y 1: tres bucles simples, leyendo y escribiendo directo en memoria (m_axi)
// =====================================================================================
static void softmax_simple(const data_t* x, data_t* y, int n) {
    data_t t[MAX_N];   // guardamos exp(x_i - m) para no calcularla dos veces (queda en BRAM)

    // Etapa 1: reduccion del maximo
    data_t m = x[0];
MAXIMO:
    for (int i = 1; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 9 max = 999
#if VARIANTE == 0
#pragma HLS PIPELINE off   // Vitis hace pipeline automatico; en la linea base lo apagamos
#else
#pragma HLS PIPELINE II = 1
#endif
        if (x[i] > m) m = x[i];
    }

    // Etapa 2: exponencial y reduccion de la suma
    // Con PIPELINE el II real de este bucle queda limitado por la suma: s depende del valor
    // de s de la iteracion anterior, y un sumador flotante tarda varios ciclos (dependencia
    // entre iteraciones). Lo vemos en el reporte y lo atacamos en la variante 2.
    data_t s = 0;
EXP_SUMA:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 1000
#if VARIANTE == 0
#pragma HLS PIPELINE off
#else
#pragma HLS PIPELINE II = 1
#endif
        data_t e = EXP(x[i] - m);
        t[i] = e;
        s += e;
    }

    // Etapa 3: normalizacion (una division por elemento)
NORMALIZA:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 1000
#if VARIANTE == 0
#pragma HLS PIPELINE off
#else
#pragma HLS PIPELINE II = 1
#endif
        y[i] = t[i] / s;
    }
}

#else
// =====================================================================================
// VARIANTES 2, 3 y 4: copiamos x a memoria local particionada y procesamos FACTOR_PART
// elementos por iteracion. La 3 agrega DATAFLOW. La 4 es la propuesta de optimizacion:
//   - II=4 en EXP_SUMA: con II=1 la suma flotante (s += ...) depende de la iteracion anterior y
//     Vitis la encadeno en un solo ciclo (13 ns > 7.3 ns de presupuesto a 100 MHz), bajando la
//     Fmax estimada a ~39 MHz (con II=2 solo subio a ~46 MHz). El sumador flotante de este chip
//     tiene latencia de 4 ciclos a 100 MHz (nucleo faddfsub_..._4_full_dsp), por eso II=4: cada
//     iteracion procesa FACTOR_PART = 8 elementos, o sea 2 elementos por ciclo.
//   - comp_t = float: en FP16 solo la entrada y la salida son half; max, exp, suma y division se
//     hacen en FP32 (igual que la CPU). Evita que los resultados chicos se vuelvan 0 en half y
//     evita los nucleos half que fallaron en cosim/export.
// =====================================================================================
#if VARIANTE >= 4
typedef float comp_t;
#else
typedef data_t comp_t;
#endif

// Leemos x de la DDR una sola vez (rafaga, 1 dato por ciclo) y dejamos dos copias:
// una para el maximo y otra para la exponencial. DATAFLOW exige que cada arreglo tenga
// un solo productor y un solo consumidor, por eso son dos copias.
static void leer(const data_t* x, comp_t xa[MAX_N], comp_t xb[MAX_N], int n) {
LEER:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 1000
#pragma HLS PIPELINE II = 1
        comp_t v = (comp_t)x[i];
        xa[i] = v;
        xb[i] = v;
    }
}

// Maximo: en cada iteracion leemos FACTOR_PART elementos a la vez (uno de cada banco de la
// particion cyclic) y los reducimos con un arbol 8->4->2->1. Asi la dependencia entre
// iteraciones es UNA sola comparacion (con m), no ocho.
static void maximo(const comp_t xa[MAX_N], int n, comp_t& m_out) {
    comp_t m = xa[0];
MAXIMO:
    for (int i = 0; i < n; i += FACTOR_PART) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 125
#pragma HLS PIPELINE II = 1
        comp_t v[FACTOR_PART];
#pragma HLS ARRAY_PARTITION variable = v complete
        for (int k = 0; k < FACTOR_PART; k++) {
#pragma HLS UNROLL
            int idx = i + k;
            v[k] = (idx < n) ? xa[idx] : xa[0];   // relleno neutro: xa[0] es un dato valido
        }
        for (int w = FACTOR_PART / 2; w > 0; w /= 2) {
#pragma HLS UNROLL
            for (int k = 0; k < w; k++) {
#pragma HLS UNROLL
                if (v[k + w] > v[k]) v[k] = v[k + w];
            }
        }
        if (v[0] > m) m = v[0];
    }
    m_out = m;
}

// Exponencial y suma: FACTOR_PART unidades hls::exp en paralelo + arbol de sumas. La dependencia
// entre iteraciones queda en una sola suma (s += parcial) en vez de una por elemento.
static void exp_suma(const comp_t xb[MAX_N], comp_t m, int n, comp_t t[MAX_N], comp_t& s_out) {
    comp_t s = 0;
EXP_SUMA:
    for (int i = 0; i < n; i += FACTOR_PART) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 125
#if VARIANTE >= 4
#pragma HLS PIPELINE II = 4
#else
#pragma HLS PIPELINE II = 1
#endif
        comp_t v[FACTOR_PART];
#pragma HLS ARRAY_PARTITION variable = v complete
        for (int k = 0; k < FACTOR_PART; k++) {
#pragma HLS UNROLL
            int idx = i + k;
            // Las posiciones que sobran del ultimo bloque (idx >= n) usan m: exp(0) = 1 y luego se
            // descartan. Asi nunca se calcula exp() sobre memoria sin inicializar.
            comp_t xv = (idx < n) ? xb[idx] : m;
            comp_t e = EXP((comp_t)(xv - m));
            if (idx < n) t[idx] = e;
            v[k] = (idx < n) ? e : (comp_t)0;   // relleno neutro de la suma: 0
        }
        for (int w = FACTOR_PART / 2; w > 0; w /= 2) {
#pragma HLS UNROLL
            for (int k = 0; k < w; k++) {
#pragma HLS UNROLL
                v[k] = v[k] + v[k + w];
            }
        }
        s += v[0];
    }
    s_out = s;
}

// Normalizacion: calculamos 1/s UNA vez y multiplicamos (un multiplicador es mucho mas
// barato y rapido que un divisor flotante por elemento).
static void normalizar(const comp_t t[MAX_N], comp_t s, int n, data_t yb[MAX_N]) {
    comp_t inv = (comp_t)1 / s;
NORMALIZA:
    for (int i = 0; i < n; i += FACTOR_PART) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 125
#pragma HLS PIPELINE II = 1
        for (int k = 0; k < FACTOR_PART; k++) {
#pragma HLS UNROLL
            int idx = i + k;
            if (idx < n) yb[idx] = (data_t)(t[idx] * inv);
        }
    }
}

// Escribimos el resultado a la DDR (rafaga, 1 dato por ciclo)
static void escribir(const data_t yb[MAX_N], data_t* y, int n) {
ESCRIBIR:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 1000
#pragma HLS PIPELINE II = 1
        y[i] = yb[i];
    }
}

static void softmax_etapas(const data_t* x, data_t* y, int n) {
#if VARIANTE >= 3
    // DATAFLOW convierte cada funcion en un proceso con su propio control. Entre procesos
    // los arreglos se vuelven buffers ping-pong: mientras una etapa trabaja sobre la
    // llamada k, la anterior ya puede empezar la llamada k+1 (mejora el intervalo).
#pragma HLS DATAFLOW
#endif
    comp_t xa[MAX_N], xb[MAX_N], t[MAX_N];
    data_t yb[MAX_N];
    // La particion cyclic reparte el arreglo en FACTOR_PART bancos: el elemento i va al banco i % FACTOR_PART.
    // Asi los FACTOR_PART elementos consecutivos de cada iteracion estan en bancos distintos y se
    // leen en el mismo ciclo (una BRAM sola solo tiene 2 puertos).
    // x y y no se particionan porque viven en la DDR (m_axi); particionamos sus copias locales.
#pragma HLS ARRAY_PARTITION variable = xa cyclic factor = FACTOR_PART
#pragma HLS ARRAY_PARTITION variable = xb cyclic factor = FACTOR_PART
#pragma HLS ARRAY_PARTITION variable = t cyclic factor = FACTOR_PART
#pragma HLS ARRAY_PARTITION variable = yb cyclic factor = FACTOR_PART
    comp_t m, s;
    leer(x, xa, xb, n);
    maximo(xa, n, m);
    exp_suma(xb, m, n, t, s);
    normalizar(t, s, n, yb);
    escribir(yb, y, n);
}
#endif

// =====================================================================================
// Funcion superior
// =====================================================================================
void softmax_hls(const data_t x[MAX_N], data_t y[MAX_N], int n) {
    // x, y: maestro AXI (m_axi) -> el IP lee/escribe directo en la DDR que reserva el host.
    //       Usamos dos bundles para que lectura y escritura tengan puertos separados.
    // n y el control (start/done): registros AXI-Lite (s_axilite) que el host escribe.
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 depth = 1000
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem1 depth = 1000
#pragma HLS INTERFACE s_axilite port = n
#pragma HLS INTERFACE s_axilite port = return

    if (n < 1) return;
    if (n > MAX_N) n = MAX_N;

#if VARIANTE <= 1
    softmax_simple(x, y, n);
#else
    softmax_etapas(x, y, n);
#endif
}
