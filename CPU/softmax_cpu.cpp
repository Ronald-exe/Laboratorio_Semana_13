// softmax_cpu.cpp - Softmax FP16 / FP32 en CPU (Jetson Nano), referencia FP64.
// Compilar: g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm
//
// Modos (--mode):
//   validar  : una fila CSV con suma, rango, argmax y errores vs FP64
//   medir    : filas CSV con tiempo de la FUNCION (us por llamada), --runs mediciones de --reps llamadas
//   potencia : repite la funcion --segundos s para que tegrastats tenga una ventana larga
//
// Entrada comun (acordar con FA/GPU): x_i en [-4,4], generador splitmix64 con --seed,
// x = -4 + 8*u, u uniforme [0,1) de 53 bits. Se genera en double y se convierte a cada formato.
// La referencia FP64 se calcula con la entrada double ORIGINAL (el error incluye la cuantizacion de la entrada).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// GCC 7 (JetPack 4.x) no tiene _Float16; en aarch64 se usa __fp16 (IEEE binary16, solo almacenamiento).
#if defined(__FLT16_MAX__) && !defined(__aarch64__)
typedef _Float16 half_t;
static const char* HALF_NAME = "_Float16";
#elif defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
typedef __fp16 half_t;
static const char* HALF_NAME = "__fp16";
#else
#error "Este compilador no soporta FP16 (_Float16 / __fp16)"
#endif

static const int MAXN = 4096;

static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void gen_input(std::vector<double>& x, uint64_t seed) {
    uint64_t s = seed;
    for (auto& v : x) {
        double u = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
        v = -4.0 + 8.0 * u;
    }
}

// ---------- Softmax numericamente estable ----------
__attribute__((noinline)) static void softmax_fp64(const double* x, double* y, int n) {
    double m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { y[i] = std::exp(x[i] - m); s += y[i]; }
    for (int i = 0; i < n; i++) y[i] /= s;
}

__attribute__((noinline)) static void softmax_fp32(const float* x, float* y, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { y[i] = std::exp(x[i] - m); s += y[i]; }
    for (int i = 0; i < n; i++) y[i] /= s;
}

// FP16: entrada y salida en half; maximo comparado en half (exacto); exp y suma en FP32
// (la CPU Cortex-A57 no tiene aritmetica FP16 nativa, solo conversion).
__attribute__((noinline)) static void softmax_fp16(const half_t* x, half_t* y, int n) {
    static float tmp[MAXN];
    half_t mh = x[0];
    for (int i = 1; i < n; i++) if ((float)x[i] > (float)mh) mh = x[i];
    const float m = (float)mh;
    float s = 0.0f;
    for (int i = 0; i < n; i++) { tmp[i] = std::exp((float)x[i] - m); s += tmp[i]; }
    for (int i = 0; i < n; i++) y[i] = (half_t)(tmp[i] / s);
}

// ---------- Utilidades ----------
struct Args {
    std::string prec = "fp32", mode = "validar";
    int n = 100, reps = 10000, runs = 5, run_id = 0;
    uint64_t seed = 12345;
    double seg = 10.0;
    bool header = false;
};

static double epoch_now() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

template <typename T>
static void validar(const std::vector<double>& x, int n, const char* prec,
                    void (*fn)(const T*, T*, int), const Args& a) {
    std::vector<double> yref(n);
    softmax_fp64(x.data(), yref.data(), n);
    std::vector<T> xi(n), y(n);
    for (int i = 0; i < n; i++) xi[i] = (T)x[i];
    fn(xi.data(), y.data(), n);

    double suma = 0, ymin = 1e300, ymax = -1e300, ea_max = 0, ea_sum = 0, er_max = 0, er_sum = 0;
    int am_ref = 0, am_cpu = 0;
    bool rango = true;
    for (int i = 0; i < n; i++) {
        double v = (double)y[i];
        suma += v;
        ymin = std::min(ymin, v);
        ymax = std::max(ymax, v);
        if (!(v >= 0.0 && v <= 1.0)) rango = false;  // tambien atrapa NaN
        if (v > (double)y[am_cpu]) am_cpu = i;
        if (yref[i] > yref[am_ref]) am_ref = i;
        double ea = std::fabs(v - yref[i]);
        double er = ea / yref[i];
        ea_max = std::max(ea_max, ea); ea_sum += ea;
        er_max = std::max(er_max, er); er_sum += er;
    }
    int empates = 0;
    for (int i = 0; i < n; i++) if ((double)y[i] == ymax) empates++;

    if (a.header)
        std::printf("precision,n,seed,suma,error_suma,y_min,y_max,en_rango_0_1,argmax_fp64,argmax_cpu,"
                    "argmax_coincide,empates_en_max,err_abs_max,err_abs_medio,err_rel_max,err_rel_medio\n");
    std::printf("%s,%d,%llu,%.10f,%.3e,%.6e,%.6e,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e\n",
                prec, n, (unsigned long long)a.seed, suma, std::fabs(suma - 1.0), ymin, ymax,
                rango ? 1 : 0, am_ref, am_cpu, am_ref == am_cpu ? 1 : 0, empates,
                ea_max, ea_sum / n, er_max, er_sum / n);
}

#define BARRIER(p) asm volatile("" : : "r"(p) : "memory")

template <typename T>
static void medir(const std::vector<double>& x, int n, const char* prec,
                  void (*fn)(const T*, T*, int), const Args& a) {
    std::vector<T> xi(n), y(n);
    for (int i = 0; i < n; i++) xi[i] = (T)x[i];
    for (int i = 0; i < std::max(200, a.reps / 10); i++) { fn(xi.data(), y.data(), n); BARRIER(y.data()); }
    if (a.header) std::printf("precision,n,reps,run,t_funcion_us_por_llamada,epoch_inicio,epoch_fin\n");
    for (int r = 0; r < a.runs; r++) {
        double e0 = epoch_now();
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < a.reps; k++) { fn(xi.data(), y.data(), n); BARRIER(y.data()); }
        auto t1 = std::chrono::steady_clock::now();
        double e1 = epoch_now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / a.reps;
        std::printf("%s,%d,%d,%d,%.6f,%.6f,%.6f\n", prec, n, a.reps, a.run_id > 0 ? a.run_id : r + 1, us, e0, e1);
    }
    std::fprintf(stderr, "# sink=%g\n", (double)y[0]);
}

template <typename T>
static void potencia(const std::vector<double>& x, int n, const char* prec,
                     void (*fn)(const T*, T*, int), const Args& a) {
    std::vector<T> xi(n), y(n);
    for (int i = 0; i < n; i++) xi[i] = (T)x[i];
    for (int i = 0; i < 1000; i++) { fn(xi.data(), y.data(), n); BARRIER(y.data()); }
    if (a.header) std::printf("precision,n,llamadas,segundos,t_llamada_us,epoch_inicio,epoch_fin\n");
    double e0 = epoch_now();
    auto t0 = std::chrono::steady_clock::now();
    long long calls = 0;
    double el = 0;
    do {
        for (int k = 0; k < 1000; k++) { fn(xi.data(), y.data(), n); BARRIER(y.data()); }
        calls += 1000;
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } while (el < a.seg);
    double e1 = epoch_now();
    std::printf("%s,%d,%lld,%.4f,%.6f,%.6f,%.6f\n", prec, n, calls, el, el * 1e6 / calls, e0, e1);
    std::fprintf(stderr, "# sink=%g\n", (double)y[0]);
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "falta valor para %s\n", k.c_str()); std::exit(1); } return argv[++i]; };
        if (k == "--precision") a.prec = next();
        else if (k == "--n") a.n = std::atoi(next());
        else if (k == "--reps") a.reps = std::atoi(next());
        else if (k == "--runs") a.runs = std::atoi(next());
        else if (k == "--run-id") a.run_id = std::atoi(next());
        else if (k == "--seed") a.seed = std::strtoull(next(), nullptr, 10);
        else if (k == "--segundos") a.seg = std::atof(next());
        else if (k == "--mode") a.mode = next();
        else if (k == "--header") a.header = true;
        else { std::fprintf(stderr, "argumento desconocido: %s\n", k.c_str()); return 1; }
    }
    if (a.n < 1 || a.n > MAXN) { std::fprintf(stderr, "n debe estar en [1,%d]\n", MAXN); return 1; }
    if (a.prec != "fp16" && a.prec != "fp32") { std::fprintf(stderr, "--precision fp16|fp32\n"); return 1; }

    std::vector<double> x(a.n);
    gen_input(x, a.seed);

    if (a.mode == "validar") {
        if (a.prec == "fp16") validar<half_t>(x, a.n, "fp16", softmax_fp16, a);
        else validar<float>(x, a.n, "fp32", softmax_fp32, a);
    } else if (a.mode == "medir") {
        if (a.prec == "fp16") medir<half_t>(x, a.n, "fp16", softmax_fp16, a);
        else medir<float>(x, a.n, "fp32", softmax_fp32, a);
    } else if (a.mode == "potencia") {
        if (a.prec == "fp16") potencia<half_t>(x, a.n, "fp16", softmax_fp16, a);
        else potencia<float>(x, a.n, "fp32", softmax_fp32, a);
    } else { std::fprintf(stderr, "--mode validar|medir|potencia\n"); return 1; }
    (void)HALF_NAME;
    return 0;
}
