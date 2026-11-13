#!/usr/bin/env python3
"""extraer_reportes.py - Junta los reportes de HLS en una tabla CSV.

Lee reportes/<prec>_v<k>/csynth.xml (sintesis) y reportes/<prec>_v<k>/cosim/*cosim.rpt
(cosimulacion) y escribe resultados/tabla_hls.csv con, por precision y variante:
latencia (ciclos), intervalo, II de cada bucle, frecuencia estimada y recursos.

Uso (desde FPGA/):  python3 extraer_reportes.py
"""
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET

AQUI = os.path.dirname(os.path.abspath(__file__))
NOMBRES = {"0": "secuencial", "1": "pipeline", "2": "pipeline+partition", "3": "pipeline+partition+dataflow"}


def texto(raiz, etiqueta):
    """Primer valor de <etiqueta> en el XML (o '' si no esta)."""
    nodo = raiz.find(".//" + etiqueta)
    return nodo.text.strip() if nodo is not None and nodo.text else ""


def ii_de_bucles(rpt):
    """Saca 'bucle:II' de la tabla de bucles del csynth.rpt (si existe)."""
    if not os.path.exists(rpt):
        return ""
    salida = []
    for linea in open(rpt, errors="ignore"):
        # filas tipo: |- EXP_SUMA  |  ...  |  ...  |  II  | ...  |  yes  |
        m = re.match(r"\s*\|\s*[-o+ ]*([A-Z_]+[A-Za-z0-9_]*)\s*\|(.*)", linea)
        if m and m.group(1) in ("MAXIMO", "EXP_SUMA", "NORMALIZA", "LEER", "ESCRIBIR"):
            celdas = [c.strip() for c in m.group(2).split("|")]
            # buscamos la celda de II: primer numero despues de latencia/iteracion
            numeros = [c for c in celdas if re.fullmatch(r"\d+|-", c)]
            salida.append(f"{m.group(1)}:{'/'.join(numeros[:6])}")
    return "; ".join(dict.fromkeys(salida))


def cosim(carpeta):
    """Latencia min/prom/max (ciclos) de la cosim en Verilog, si existe."""
    for rpt in glob.glob(os.path.join(carpeta, "cosim", "*cosim.rpt")):
        for linea in open(rpt, errors="ignore"):
            if "Verilog" in linea:
                c = [x.strip() for x in linea.strip().strip("|").split("|")]
                # | Verilog | Pass | lat_min | lat_avg | lat_max | int_min | int_avg | int_max | total |
                if len(c) >= 5:
                    return c[1], c[2], c[3], c[4]
    return "", "", "", ""


def main():
    filas = []
    for carpeta in sorted(glob.glob(os.path.join(AQUI, "reportes", "*_v*"))):
        prec, var = os.path.basename(carpeta).split("_v")
        xml = os.path.join(carpeta, "csynth.xml")
        if not os.path.exists(xml):
            continue
        r = ET.parse(xml).getroot()
        clk_obj = texto(r, "TargetClockPeriod")
        clk_est = texto(r, "EstimatedClockPeriod")
        fmax = f"{1000.0 / float(clk_est):.1f}" if clk_est else ""
        estado, lmin, lavg, lmax = cosim(carpeta)
        filas.append({
            "precision": prec, "variante": var, "nombre": NOMBRES.get(var, ""),
            "clk_objetivo_ns": clk_obj, "clk_estimado_ns": clk_est, "fmax_estimada_MHz": fmax,
            "lat_min_ciclos": texto(r, "Best-caseLatency"), "lat_max_ciclos": texto(r, "Worst-caseLatency"),
            "intervalo_min": texto(r, "Interval-min"), "intervalo_max": texto(r, "Interval-max"),
            "LUT": texto(r, "LUT"), "FF": texto(r, "FF"), "BRAM_18K": texto(r, "BRAM_18K"),
            "URAM": texto(r, "URAM"), "DSP": texto(r, "DSP"),
            "cosim_estado": estado, "cosim_lat_min": lmin, "cosim_lat_prom": lavg, "cosim_lat_max": lmax,
            "bucles": ii_de_bucles(os.path.join(carpeta, "csynth.rpt")),
        })
    if not filas:
        print("No hay reportes en reportes/*_v*/csynth.xml (corra run_hls.tcl primero)")
        return
    os.makedirs(os.path.join(AQUI, "resultados"), exist_ok=True)
    salida = os.path.join(AQUI, "resultados", "tabla_hls.csv")
    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    for x in filas:
        print(f"{x['precision']} v{x['variante']:<2} lat={x['lat_min_ciclos']}-{x['lat_max_ciclos']} "
              f"clk={x['clk_estimado_ns']}ns LUT={x['LUT']} FF={x['FF']} DSP={x['DSP']} BRAM={x['BRAM_18K']} "
              f"cosim={x['cosim_estado']}")
    print("Tabla escrita en", salida)


if __name__ == "__main__":
    main()
