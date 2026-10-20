"""Estudio E: efecto de las banderas de compilación (mismo código, distintas opciones de g++).
Uso: python estudio_compilacion.py <proy_ruido.cpp> <carpeta_datos> <salida.csv> [iteraciones]"""
import sys, os, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from generar_datos import generar
import numpy as np
fuente, datos, salida = sys.argv[1], sys.argv[2], sys.argv[3]
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5
N = 40000
os.makedirs(datos, exist_ok=True)
ent = os.path.join(datos, f"{N}_data.csv")
if not os.path.exists(ent): np.savetxt(ent, generar(N), delimiter=",", fmt="%.3f")
variantes = {"-O0": ["-O0"], "-O2": ["-O2"], "-O3": ["-O3"], "-O3 -march=native": ["-O3", "-march=native", "-ffp-contract=off"]}
configs = [("serial", 1, 1), ("p1", 1, 2), ("p2", 4, 2), ("p2", 16, 2), ("p2", 64, 2)]
fh = open(salida, "w", newline=""); w = csv.writer(fh)
w.writerow(["bandera", "version", "k", "n_puntos", "hilos", "iteracion", "tiempo_segundos"])
for nombre, flags in variantes.items():
    exe = os.path.join(datos, "dbscan_" + nombre.replace(" ", "").replace("=", "_"))
    subprocess.run(["g++", *flags, "-fopenmp", "-o", exe, fuente], check=True)
    for v, k, h in configs:
        for it in range(1, iters + 1):
            r = subprocess.run([exe, ent, os.path.join(datos, "out_comp.csv"), "0.03", "10", v, str(k), str(h)], capture_output=True, text=True, check=True)
            w.writerow([nombre, v, k, N, h, it, r.stdout.strip()]); fh.flush()
    print(nombre, flush=True)
fh.close()
