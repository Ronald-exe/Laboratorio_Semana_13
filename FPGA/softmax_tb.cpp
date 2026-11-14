// softmax_tb.cpp - Testbench de C-sim y cosimulacion
//
// Usamos la MISMA entrada que CPU (KD) y GPU (FA): generador splitmix64, semilla 12345,
// x = -4 + 8u con u uniforme de 53 bits. Para cada N la secuencia arranca desde la semilla.
// La referencia de exactitud es softmax en double con std::exp (FP64).
//
// Revisamos lo mismo que en CPU: suma ~ 1, todos los y_i en [0,1], posicion del maximo
// igual a la de FP64, y error absoluto/relativo contra FP64.
// Devuelve 0 si todo pasa (Vitis marca la C-sim/cosim como PASS) y 1 si algo falla.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "softmax_hls.h"

static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void softmax_fp64(const double* x, double* y, int n) {
    double m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { y[i] = std::exp(x[i] - m); s += y[i]; }
    for (int i = 0; i < n; i++) y[i] /= s;
}

int main() {
#ifdef USE_FP16
    const char* prec = "fp16";
    const double tol_suma = 2e-2;   // FP16 tiene ~3 cifras decimales
#else
    const char* prec = "fp32";
    const double tol_suma = 1e-4;
#endif
    const int tamanos[] = {10, 100, 500, 1000};
    const uint64_t SEED = 12345;
    int fallas = 0;

    FILE* csv = std::fopen("validacion_hls.csv", "w");
    if (csv) std::fprintf(csv, "plataforma,precision,variante,n,seed,suma,en_rango_0_1,argmax_fp64,argmax_hls,"
                               "argmax_coincide,err_abs_max,err_abs_medio,err_rel_max,err_rel_medio\n");
    std::printf("precision=%s variante=%d\n", prec, VARIANTE);
    std::printf("   N        suma  rango  argmax  err_abs_max  err_rel_max\n");

    for (int n : tamanos) {
        // Entrada en double y luego convertida al formato del kernel
        std::vector<double> xd(n), yref(n);
        uint64_t s = SEED;
        for (int i = 0; i < n; i++) {
            double u = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
            xd[i] = -4.0 + 8.0 * u;
        }
        softmax_fp64(xd.data(), yref.data(), n);

        static data_t x[MAX_N], y[MAX_N];
        for (int i = 0; i < MAX_N; i++) { x[i] = (data_t)0; y[i] = (data_t)0; }
        for (int i = 0; i < n; i++) x[i] = (data_t)xd[i];

        softmax_hls(x, y, n);   // llamada al kernel (en cosim, esto corre el RTL)

        double suma = 0, ea_max = 0, ea_sum = 0, er_max = 0, er_sum = 0;
        bool rango = true;
        int am_ref = 0, am_hls = 0;
        for (int i = 0; i < n; i++) {
            double v = (double)(float)y[i];
            suma += v;
            if (!(v >= 0.0 && v <= 1.0)) rango = false;   // tambien atrapa NaN
            if (v > (double)(float)y[am_hls]) am_hls = i;
            if (yref[i] > yref[am_ref]) am_ref = i;
            double ea = std::fabs(v - yref[i]);
            double er = ea / yref[i];
            if (ea > ea_max) ea_max = ea;
            if (er > er_max) er_max = er;
            ea_sum += ea; er_sum += er;
        }
        // En FP16 dos salidas pueden redondear al mismo valor: si y[am_ref] empata con el
        // maximo de la salida, la posicion del maximo se considera correcta.
        bool am_ok = (am_ref == am_hls) || ((float)y[am_ref] == (float)y[am_hls]);
        bool ok = rango && am_ok && std::fabs(suma - 1.0) < tol_suma;
        if (!ok) fallas++;

        std::printf("%4d  %10.7f  %5s  %3d/%-3d  %11.3e  %11.3e  %s\n", n, suma, rango ? "si" : "NO",
                    am_ref, am_hls, ea_max, er_max, ok ? "OK" : "FALLA");
        if (csv) std::fprintf(csv, "KV260,%s,%d,%d,%llu,%.10f,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e\n", prec,
                              VARIANTE, n, (unsigned long long)SEED, suma, rango ? 1 : 0, am_ref, am_hls,
                              am_ok ? 1 : 0, ea_max, ea_sum / n, er_max, er_sum / n);
    }
    if (csv) std::fclose(csv);
    std::printf(fallas ? "*** %d prueba(s) fallaron ***\n" : "*** TODAS LAS PRUEBAS PASAN ***\n", fallas);
    return fallas ? 1 : 0;
}
