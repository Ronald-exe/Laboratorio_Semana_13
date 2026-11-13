// softmax_hls.h - Definiciones comunes del kernel HLS (KV260)
//
// Elegimos dos cosas al compilar, con banderas -D:
//   -DUSE_FP16      -> data_t = half (FP16: signo, 5 bits de exponente, 10 de mantisa)
//                      sin esta bandera data_t = float (FP32)
//   -DVARIANTE=k    -> que version del kernel se sintetiza:
//                      0 = secuencial (sin pipeline, es la linea base)
//                      1 = + PIPELINE II=1 en cada bucle
//                      2 = + ARRAY_PARTITION cyclic + UNROLL (procesa F elementos por iteracion)
//                      3 = + DATAFLOW (las etapas se vuelven procesos que se solapan)
#ifndef SOFTMAX_HLS_H
#define SOFTMAX_HLS_H

#define MAX_N 1000   // tamano maximo del vector (lo pide el enunciado)
#define F 8          // factor de particion/unroll de las variantes 2 y 3 (MAX_N es multiplo de F)

#ifndef VARIANTE
#define VARIANTE 0
#endif

#ifdef USE_FP16
  #if __has_include("hls_half.h")
    // Dentro de Vitis HLS usamos el tipo half de AMD (FP16 IEEE real, NO punto fijo)
    #include "hls_half.h"
    typedef half data_t;
  #else
    // Fuera de Vitis (probar el testbench con g++) usamos el FP16 del compilador
    typedef _Float16 data_t;
  #endif
#else
  typedef float data_t;
#endif

// Funcion superior (top) que pide el enunciado
void softmax_hls(const data_t x[MAX_N], data_t y[MAX_N], int n);

#endif
