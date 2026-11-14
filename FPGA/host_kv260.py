#!/usr/bin/env python3
"""host_kv260.py - Programa host (PYNQ) que corre el IP softmax_hls en la KV260 y mide
tiempo y energia.

Por cada llamada separamos tres tiempos (medidos desde el host con perf_counter_ns):
  t_transfer : copiar x al buffer DMA + sync_to_device  +  sync_from_device + leer y
  t_ip       : escribir ap_start y esperar ap_done (incluye el costo de los accesos AXI-Lite)
  t_total    : todo lo anterior = lo que observa el host

Energia: leemos el sensor INA260 (U14) de la placa por /sys/class/hwmon/*/power1_input
(microwatts). Es la potencia de TODO el SOM (PS + PL + DDR), no solo del acelerador.
Un hilo muestrea la potencia mientras corren las repeticiones; E = P_promedio * t_corrida.
Tambien medimos la potencia en reposo (sin llamar al IP) para poder restar la base.

Uso en la Kria (PYNQ necesita root):
  sudo bash -c 'source /etc/profile.d/pynq_venv.sh && \
      python3 host_kv260.py --bit overlay/softmax_fp32.bit --precision fp32'
"""
import argparse
import csv
import glob
import os
import threading
import time

import numpy as np
from pynq import Overlay, allocate

MASK64 = (1 << 64) - 1


def entrada_comun(n, seed=12345):
    """Misma entrada que CPU/GPU/testbench: splitmix64, x = -4 + 8u, u de 53 bits."""
    s, x = seed, np.empty(n)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
        x[i] = -4.0 + 8.0 * ((z >> 11) * (1.0 / 9007199254740992.0))
    return x


def softmax_fp64(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ------------------------------------------------------------------ sensor de potencia
def buscar_ina260():
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(os.path.join(h, "name")).read().strip().startswith("ina260"):
                return os.path.join(h, "power1_input")
        except OSError:
            pass
    return None


def leer_temp():
    """Temperatura del PS por el AMS (si esta disponible), en grados C."""
    for f in glob.glob("/sys/class/hwmon/hwmon*/temp1_input"):
        try:
            return int(open(f).read()) / 1000.0
        except (OSError, ValueError):
            pass
    return float("nan")


class Muestreador(threading.Thread):
    """Lee la potencia cada 'periodo' segundos hasta que se llame detener()."""

    def __init__(self, ruta, periodo=0.01):
        super().__init__(daemon=True)
        self.ruta, self.periodo, self.muestras = ruta, periodo, []
        self._fin = threading.Event()

    def run(self):
        while not self._fin.is_set():
            try:
                self.muestras.append(int(open(self.ruta).read()) / 1e6)   # uW -> W
            except (OSError, ValueError):
                pass
            time.sleep(self.periodo)

    def detener(self):
        self._fin.set()
        self.join()
        return float(np.mean(self.muestras)) if self.muestras else float("nan")


# ------------------------------------------------------------------ registros del IP
def offsets(ol, nombre_ip):
    """Busca en el .hwh los offsets AXI-Lite de x, y, n. Si no los encuentra usa los
    valores tipicos de Vitis HLS (0x10, 0x1c, 0x28)."""
    regs = ol.ip_dict[nombre_ip].get("registers", {})
    tipicos = {"x": 0x10, "y": 0x1C, "n": 0x28}
    salida = {}
    for arg, defecto in tipicos.items():
        for clave in (arg + "_1", arg, arg.upper()):
            if clave in regs:
                salida[arg] = regs[clave]["address_offset"]
                break
        else:
            print(f"AVISO: registro '{arg}' no esta en el hwh, uso 0x{defecto:x}")
            salida[arg] = defecto
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[10, 100, 500, 1000])
    ap.add_argument("--reps", type=int, default=2000, help="llamadas por corrida")
    ap.add_argument("--runs", type=int, default=5, help="corridas medidas por N")
    ap.add_argument("--reposo", type=float, default=3.0, help="segundos para medir potencia en reposo")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dtype = np.float16 if a.precision == "fp16" else np.float32
    out = a.out or f"resultados/kv260_{a.precision}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ol = Overlay(a.bit)
    nombre_ip = [k for k in ol.ip_dict if "softmax_hls" in k][0]
    ip = getattr(ol, nombre_ip)
    off = offsets(ol, nombre_ip)

    xbuf = allocate(shape=(1000,), dtype=dtype)
    ybuf = allocate(shape=(1000,), dtype=dtype)
    # Las direcciones fisicas de los buffers se escriben una sola vez (64 bits: baja y alta)
    for arg, buf in (("x", xbuf), ("y", ybuf)):
        ip.write(off[arg], buf.device_address & 0xFFFFFFFF)
        ip.write(off[arg] + 4, buf.device_address >> 32)

    sensor = buscar_ina260()
    if sensor is None:
        print("AVISO: no se encontro el INA260; no habra potencia/energia")

    # Potencia en reposo (placa encendida, overlay cargado, sin llamar al IP)
    p_reposo = float("nan")
    if sensor:
        m = Muestreador(sensor)
        m.start()
        time.sleep(a.reposo)
        p_reposo = m.detener()
        print(f"Potencia en reposo: {p_reposo:.3f} W")

    nuevo = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if nuevo:
        w.writerow(["plataforma", "precision", "n", "reps", "run", "t_ip_us", "t_transfer_us", "t_total_us",
                    "p_prom_W", "p_reposo_W", "E_llamada_uJ", "E_elemento_uJ", "E_dinamica_llamada_uJ",
                    "temp_C", "err_abs_max", "err_rel_max", "suma", "argmax_ok", "epoch_inicio", "epoch_fin"])

    for n in a.n:
        x64 = entrada_comun(n)
        xin = x64.astype(dtype)
        ref = softmax_fp64(x64)
        ip.write(off["n"], n)

        def llamada():
            t0 = time.perf_counter_ns()
            xbuf[:n] = xin
            xbuf.sync_to_device()          # vaciamos la cache para que el IP vea x en la DDR
            t1 = time.perf_counter_ns()
            ip.write(0x00, 1)              # ap_start
            while (ip.read(0x00) & 0x2) == 0:   # esperamos ap_done
                pass
            t2 = time.perf_counter_ns()
            ybuf.sync_from_device()        # invalidamos la cache para leer y desde la DDR
            y = np.array(ybuf[:n])
            t3 = time.perf_counter_ns()
            return y, (t1 - t0) + (t3 - t2), t2 - t1, t3 - t0

        # Validacion contra FP64 (una llamada)
        y, *_ = llamada()
        y64 = y.astype(np.float64)
        ea = np.abs(y64 - ref)
        err_abs, err_rel = float(ea.max()), float((ea / ref).max())
        argmax_ok = int(np.argmax(y64) == np.argmax(ref) or y64[np.argmax(ref)] == y64.max())
        print(f"N={n}: suma={y64.sum():.6f} err_abs_max={err_abs:.3e} err_rel_max={err_rel:.3e} argmax_ok={argmax_ok}")

        # Calentamiento (no se guarda)
        for _ in range(max(50, a.reps // 10)):
            llamada()

        for r in range(1, a.runs + 1):
            m = Muestreador(sensor) if sensor else None
            if m:
                m.start()
            tr = ti = tt = 0
            e0 = time.time()
            c0 = time.perf_counter_ns()
            for _ in range(a.reps):
                _, dtr, dti, dtt = llamada()
                tr += dtr; ti += dti; tt += dtt
            dur = (time.perf_counter_ns() - c0) / 1e9
            e1 = time.time()
            p = m.detener() if m else float("nan")
            # E_ejecucion = (E_fin - E_inicio) / repeticiones = P_prom * duracion / reps
            e_llamada = p * dur / a.reps * 1e6                    # uJ
            e_din = (p - p_reposo) * dur / a.reps * 1e6           # uJ por encima del reposo
            fila = ["KV260", a.precision, n, a.reps, r, ti / a.reps / 1e3, tr / a.reps / 1e3,
                    tt / a.reps / 1e3, p, p_reposo, e_llamada, e_llamada / n, e_din, leer_temp(),
                    err_abs, err_rel, float(y64.sum()), argmax_ok, f"{e0:.6f}", f"{e1:.6f}"]
            w.writerow(fila)
            f.flush()
            print(f"  run {r}: t_ip={fila[5]:.2f}us t_transf={fila[6]:.2f}us t_total={fila[7]:.2f}us "
                  f"P={p:.3f}W E={e_llamada:.2f}uJ/llamada")
    f.close()
    print("CSV:", out)


if __name__ == "__main__":
    main()
