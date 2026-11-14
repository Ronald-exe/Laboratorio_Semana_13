#!/usr/bin/env python3
"""consolidar.py - Junta los resultados de CPU, GPU y KV260 en las tablas y graficas del informe.

Lee (desde la raiz del repositorio) lo que exista:
  CPU/resultados/mediciones_<prec>.csv, validacion_<prec>.csv, tegrastats.log   (KD)
  GPU/resultados/*.csv  (formato de softmax_cuda: tipo,n,precision,...,kernel_ms,end_to_end_ms,...)  (FA)
  FPGA/resultados/kv260_<prec>.csv  (host_kv260.py)                               (KA/RO)

Escribe en Resultados/:
  tablas_comparativas.csv   formato largo: una fila por plataforma/precision/N/medida
  tablas_comparativas.md    la tabla del enunciado, una por precision
  graficas/*.png            tiempo vs N, error vs FP64, energia por llamada y por elemento

No se estima nada: si una plataforma o una medida no tiene datos, la celda queda como "pendiente"
o "N/D" (por ejemplo la potencia de la Jetson Nano 2GB, que no tiene sensor INA3221).

Uso:  python3 Resultados/consolidar.py
"""
import csv
import glob
import math
import os
import re
import statistics as st

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "Resultados")
NS = [10, 100, 500, 1000]
PRECS = ["fp16", "fp32"]


def leer_csv(ruta):
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def resumen(valores):
    """Mediana, minimo y maximo (lo que pide el protocolo)."""
    v = [x for x in valores if x is not None and not math.isnan(x)]
    if not v:
        return None
    return {"mediana": st.median(v), "min": min(v), "max": max(v), "muestras": len(v)}


def potencia_tegrastats(ruta, t0, t1):
    """Potencia media (W) de la placa entre dos tiempos epoch, si tegrastats reporta POM_5V_IN.
    En la Jetson Nano 2GB no hay sensor, entonces devuelve None."""
    if not os.path.exists(ruta):
        return None
    mw = []
    for linea in open(ruta, errors="ignore"):
        partes = linea.split(" ", 1)
        try:
            t = float(partes[0])
        except ValueError:
            continue
        m = re.search(r"POM_5V_IN (\d+)/", linea)
        if m and t0 <= t <= t1:
            mw.append(int(m.group(1)))
    return st.mean(mw) / 1000.0 if mw else None


# ------------------------------------------------------------------------------- lectores
def datos_cpu():
    filas = []
    for p in PRECS:
        med = os.path.join(RAIZ, "CPU", "resultados", f"mediciones_{p}.csv")
        val = os.path.join(RAIZ, "CPU", "resultados", f"validacion_{p}.csv")
        teg = os.path.join(RAIZ, "CPU", "resultados", "tegrastats.log")
        if not os.path.exists(med):
            continue
        errores = {int(r["n"]): r for r in leer_csv(val)} if os.path.exists(val) else {}
        m = leer_csv(med)
        for n in NS:
            rs = [r for r in m if int(r["n"]) == n]
            if not rs:
                continue
            pot = [potencia_tegrastats(teg, float(r["epoch_inicio"]), float(r["epoch_fin"])) for r in rs]
            e = errores.get(n, {})
            comunes = {"err_abs_max": e.get("err_abs_max"), "err_rel_max": e.get("err_rel_max"),
                       "potencia": resumen([x for x in pot if x is not None])}
            filas.append({"plataforma": "CPU", "precision": p, "n": n, "medida": "funcion",
                          "t": resumen([float(r["t_funcion_us_por_llamada"]) for r in rs]), **comunes})
            filas.append({"plataforma": "CPU", "precision": p, "n": n, "medida": "programa_completo",
                          "t": resumen([float(r["t_programa_completo_ms"]) * 1000 / int(r["reps"]) for r in rs]),
                          **comunes})
    return filas


def datos_gpu():
    filas = []
    todos = []
    for ruta in glob.glob(os.path.join(RAIZ, "GPU", "resultados", "*.csv")):
        try:
            todos += [r for r in leer_csv(ruta) if r.get("tipo") == "benchmark"]
        except (KeyError, csv.Error):
            pass
    for p in PRECS:
        for n in NS:
            rs = [r for r in todos if r["precision"] == p and int(r["n"]) == n]
            if not rs:
                continue
            comunes = {"err_abs_max": max(float(r["error_abs_max"]) for r in rs),
                       "err_rel_max": max(float(r["error_rel_max"]) for r in rs), "potencia": None}
            filas.append({"plataforma": "GPU", "precision": p, "n": n, "medida": "kernel",
                          "t": resumen([float(r["kernel_ms"]) * 1000 for r in rs]), **comunes})
            filas.append({"plataforma": "GPU", "precision": p, "n": n, "medida": "end_to_end",
                          "t": resumen([float(r["end_to_end_ms"]) * 1000 for r in rs]), **comunes})
            # Tiempo del kernel segun el perfilador (nvprof da promedio/min/max, no mediana)
            prof = nvprof(os.path.join(RAIZ, "GPU", "perfilador", f"nvprof_{p}_n{n}.txt"))
            k = next((v for nombre, v in prof.items() if "softmax_kernel" in nombre), None)
            if k:
                filas.append({"plataforma": "GPU", "precision": p, "n": n, "medida": "kernel_nvprof",
                              "t": {"mediana": k[0], "min": k[1], "max": k[2], "muestras": k[3]}, **comunes})
    return filas


def a_us(texto):
    """'66.085us' / '2.5ms' / '937ns' -> microsegundos."""
    m = re.fullmatch(r"([\d.]+)(ns|us|ms|s)", texto)
    if not m:
        return None
    return float(m.group(1)) * {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[m.group(2)]


def nvprof(ruta):
    """Lee el resumen de nvprof: {nombre: (prom_us, min_us, max_us, llamadas)} para las
    actividades de GPU (kernel, memcpy) y las llamadas a la API de CUDA."""
    salida = {}
    if not os.path.exists(ruta):
        return salida
    for linea in open(ruta, errors="ignore"):
        linea = re.sub(r"^\s*(GPU activities|API calls):", "", linea)
        c = linea.split()
        if len(c) >= 7 and c[0].endswith("%") and c[2].isdigit():
            prom, mn, mx = a_us(c[3]), a_us(c[4]), a_us(c[5])
            if prom is not None:
                salida[" ".join(c[6:])] = (prom, mn, mx, int(c[2]))
    return salida


def perfil_gpu():
    """Tabla del desglose de la GPU segun nvprof (para la pregunta 1)."""
    filas = []
    for p in PRECS:
        for n in NS:
            prof = nvprof(os.path.join(RAIZ, "GPU", "perfilador", f"nvprof_{p}_n{n}.txt"))
            if not prof:
                continue
            def buscar(clave):
                v = next((v for k, v in prof.items() if clave in k), None)
                return f"{v[0]:.2f}" if v else "N/D"
            filas.append([p, n, buscar("softmax_kernel"), buscar("memcpy HtoD"), buscar("memcpy DtoH"),
                          buscar("cudaLaunchKernel"), buscar("cudaMemcpy"), buscar("cudaDeviceSynchronize")])
    if not filas:
        return None
    ruta = os.path.join(SALIDA, "perfil_gpu_nvprof.md")
    lineas = ["# Desglose de la GPU según nvprof (promedio, µs por llamada)\n",
              "Actividades en la GPU (kernel, copias) y costo de las llamadas a la API vistas desde la CPU.\n",
              "| Precisión | N | Kernel (GPU) | memcpy H→D | memcpy D→H | cudaLaunchKernel (API) | cudaMemcpy (API) | cudaDeviceSynchronize (API) |",
              "|---|---|---|---|---|---|---|---|"]
    lineas += ["| " + " | ".join(str(x) for x in f) + " |" for f in filas]
    open(ruta, "w").write("\n".join(lineas) + "\n")
    return ruta


def datos_kv260():
    filas = []
    for p in PRECS:
        ruta = os.path.join(RAIZ, "FPGA", "resultados", f"kv260_{p}.csv")
        if not os.path.exists(ruta):
            continue
        m = leer_csv(ruta)
        for n in NS:
            rs = [r for r in m if int(r["n"]) == n]
            if not rs:
                continue
            comunes = {"err_abs_max": rs[0]["err_abs_max"], "err_rel_max": rs[0]["err_rel_max"],
                       "potencia": resumen([float(r["p_prom_W"]) for r in rs]),
                       "energia": resumen([float(r["E_llamada_uJ"]) for r in rs])}
            for medida, col in (("ip", "t_ip_us"), ("transferencias", "t_transfer_us"), ("total_host", "t_total_us")):
                filas.append({"plataforma": "KV260", "precision": p, "n": n, "medida": medida,
                              "t": resumen([float(r[col]) for r in rs]), **comunes})
    return filas


# ------------------------------------------------------------------------------- salidas
def energia_de(f):
    """Energia por llamada (uJ): la medida directa (KV260) o P*t con la mediana del tiempo."""
    if f.get("energia"):
        return f["energia"]["mediana"]
    if f.get("potencia") and f.get("t"):
        return f["potencia"]["mediana"] * f["t"]["mediana"]   # W * us = uJ
    return None


def escribir_csv(filas):
    ruta = os.path.join(SALIDA, "tablas_comparativas.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plataforma", "precision", "n", "medida", "t_mediana_us", "t_min_us", "t_max_us",
                    "muestras", "potencia_W", "energia_llamada_uJ", "energia_elemento_uJ",
                    "err_abs_max", "err_rel_max"])
        for x in filas:
            e = energia_de(x)
            w.writerow([x["plataforma"], x["precision"], x["n"], x["medida"],
                        f"{x['t']['mediana']:.4f}", f"{x['t']['min']:.4f}", f"{x['t']['max']:.4f}",
                        x["t"]["muestras"],
                        f"{x['potencia']['mediana']:.3f}" if x.get("potencia") else "N/D",
                        f"{e:.4f}" if e is not None else "N/D",
                        f"{e / x['n']:.6f}" if e is not None else "N/D",
                        x["err_abs_max"], x["err_rel_max"]])
    return ruta


def fmt_t(f):
    t = f["t"]
    return f"{t['mediana']:.2f} [{t['min']:.2f}–{t['max']:.2f}]"


def fmt_err(v):
    try:
        return f"{float(v):.2e}"
    except (TypeError, ValueError):
        return "N/D"


def escribir_md(filas):
    nombres = {"funcion": "función", "programa_completo": "programa completo", "kernel": "kernel",
               "end_to_end": "end-to-end", "kernel_nvprof": "kernel (nvprof, prom.)", "ip": "IP", "transferencias": "transferencias",
               "total_host": "total host"}
    lineas = ["# Tablas comparativas\n",
              "Tiempo: mediana [mín–máx] de las corridas, en µs por llamada. "
              "Energía: µJ por llamada; J/elem en µJ por elemento. "
              "**pendiente** = sin datos todavía; **N/D** = la plataforma no tiene cómo medirlo "
              "(la Jetson Nano 2GB no trae sensor de potencia).\n"]
    for p in PRECS:
        lineas += [f"\n## {p.upper()}\n",
                   "| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | µJ/elem. | Error máx. |",
                   "|---|---|---|---|---|---|---|"]
        for plat, medidas in (("CPU", ["funcion"]), ("GPU", ["kernel_nvprof", "kernel", "end_to_end"]),
                              ("KV260", ["total_host", "ip", "transferencias"])):
            for n in NS:
                fs = {f["medida"]: f for f in filas if f["plataforma"] == plat and f["precision"] == p and f["n"] == n}
                if not fs:
                    lineas.append(f"| {plat} | {n} | pendiente | pendiente | pendiente | pendiente | pendiente |")
                    continue
                tiempo = "<br>".join(f"{nombres[m]}: {fmt_t(fs[m])}" for m in medidas if m in fs)
                ref = fs.get(medidas[0]) or next(iter(fs.values()))
                e = energia_de(ref)
                pot = f"{ref['potencia']['mediana']:.3f}" if ref.get("potencia") else "N/D"
                lineas.append(f"| {plat} | {n} | {tiempo} | {pot} | "
                              f"{f'{e:.3f}' if e is not None else 'N/D'} | "
                              f"{f'{e / n:.5f}' if e is not None else 'N/D'} | {fmt_err(ref['err_abs_max'])} |")
    ruta = os.path.join(SALIDA, "tablas_comparativas.md")
    open(ruta, "w").write("\n".join(lineas) + "\n")
    return ruta


def graficas(filas):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: sin matplotlib, no se generan graficas")
        return []
    os.makedirs(os.path.join(SALIDA, "graficas"), exist_ok=True)
    # Un color fijo por plataforma (la identidad no cambia entre graficas); la medida
    # secundaria de cada plataforma va con la misma tinta y linea punteada.
    COLOR = {"CPU": "#2a78d6", "GPU": "#eb6834", "KV260": "#1baf7a"}
    SERIES = [("CPU", "funcion", "-", "CPU (función)"),
              ("GPU", "kernel", "-", "GPU (kernel)"),
              ("GPU", "kernel_nvprof", ":", "GPU (kernel, nvprof)"),
              ("GPU", "end_to_end", "--", "GPU (end-to-end)"),
              ("KV260", "total_host", "-", "KV260 (total host)"),
              ("KV260", "ip", "--", "KV260 (IP)")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#5c5c58", "ytick.color": "#5c5c58"})
    hechas = []

    def figura(nombre, titulo, eje_y, valor, dispersion=True):
        fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        hay = False
        for ax, p in zip(ejes, PRECS):
            for plat, med, estilo, etiqueta in SERIES:
                pts = sorted((f["n"], f) for f in filas if f["plataforma"] == plat and f["medida"] == med
                             and f["precision"] == p)
                pts = [(n, f, valor(f)) for n, f in pts if valor(f) is not None]
                if not pts:
                    continue
                hay = True
                xs = [n for n, _, _ in pts]
                ys = [v for _, _, v in pts]
                ax.plot(xs, ys, estilo, color=COLOR[plat], lw=2, marker="o", ms=6, label=etiqueta,
                        markeredgecolor="white", markeredgewidth=1.5)
                if dispersion:
                    lo = [v - f["t"]["min"] for _, f, v in pts]
                    hi = [f["t"]["max"] - v for _, f, v in pts]
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=COLOR[plat], elinewidth=1, capsize=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS])
            ax.set_title(p.upper(), color="#2b2b29")
            ax.set_xlabel("N (elementos)")
            ax.grid(True, which="major", color="#e4e3df", lw=0.8)
        if not hay:
            plt.close(fig)
            return
        ejes[0].set_ylabel(eje_y)
        # Leyenda abajo, fuera de los paneles, para no tapar datos
        manejadores, etiquetas = {}, {}
        for ax in ejes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                manejadores.setdefault(l, h)
        fig.legend(list(manejadores.values()), list(manejadores.keys()), loc="lower center",
                   ncol=min(len(manejadores), 5), frameon=False, fontsize=9)
        fig.suptitle(titulo, color="#2b2b29")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        ruta = os.path.join(SALIDA, "graficas", nombre)
        fig.savefig(ruta, dpi=160, facecolor="white")
        plt.close(fig)
        hechas.append(ruta)

    figura("tiempo_vs_N.png", "Tiempo por llamada contra N (mediana; barras = mín–máx)",
           "Tiempo por llamada (µs)", lambda f: f["t"]["mediana"])
    figura("error_vs_N.png", "Error absoluto máximo contra FP64", "Error absoluto máximo",
           lambda f: float(f["err_abs_max"]) if f.get("err_abs_max") not in (None, "") else None,
           dispersion=False)
    # El error absoluto baja con N solo porque cada y_i ~ 1/N se achica; el relativo muestra
    # el deterioro real (sobre todo en FP16, donde la reduccion acumula redondeo).
    figura("error_relativo_vs_N.png", "Error relativo máximo contra FP64", "Error relativo máximo",
           lambda f: float(f["err_rel_max"]) if f.get("err_rel_max") not in (None, "") else None,
           dispersion=False)
    figura("energia_llamada_vs_N.png", "Energía por llamada contra N", "Energía por llamada (µJ)",
           energia_de, dispersion=False)
    figura("energia_elemento_vs_N.png", "Energía por elemento contra N", "Energía por elemento (µJ)",
           lambda f: energia_de(f) / f["n"] if energia_de(f) is not None else None, dispersion=False)
    return hechas


def main():
    filas = datos_cpu() + datos_gpu() + datos_kv260()
    os.makedirs(SALIDA, exist_ok=True)
    for plat in ("CPU", "GPU", "KV260"):
        n = len([f for f in filas if f["plataforma"] == plat])
        print(f"{plat:6s}: {n} filas" + ("" if n else "  (sin datos todavia)"))
    if not filas:
        return
    print("->", escribir_csv(filas))
    print("->", escribir_md(filas))
    pg = perfil_gpu()
    if pg:
        print("->", pg)
    for g in graficas(filas):
        print("->", g)


if __name__ == "__main__":
    main()
