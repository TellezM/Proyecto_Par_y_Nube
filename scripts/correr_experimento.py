"""Corre el experimento completo de la rúbrica y guarda todos los tiempos en un CSV.

  k       : 2, 4, 8, 16, 32, 64                          (solo para P2)
  puntos  : 20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000
  hilos   : 1, cores/2, cores, cores*2
  10 iteraciones por configuración.

Uso:
  python correr_experimento.py <ejecutable> <carpeta_datos> <resultados.csv> --cores N
Ejemplo (computadora con 2 núcleos):
  python correr_experimento.py ./dbscan datos resultados_tiempos.csv --cores 2

- Si el CSV ya existe, continúa donde se quedó (no repite lo que ya está).
- Las corridas se hacen en "rondas": en cada ronda se corre UNA iteración de todas las
  configuraciones. Así cualquier variación del sistema se reparte parejo entre versiones.
- En cada corrida se compara la salida contra la de la versión serial (columna salida_ok).
"""
import argparse
import csv
import hashlib
import os
import subprocess
import time
import numpy as np
from generar_datos import generar

# ---------------- Argumentos ----------------
ap = argparse.ArgumentParser()
ap.add_argument("ejecutable")
ap.add_argument("carpeta_datos")
ap.add_argument("salida_csv")
ap.add_argument("--cores", type=int, default=os.cpu_count())
ap.add_argument("--iters", type=int, default=10)
args = ap.parse_args()

EPS, MIN_SAMPLES = "0.03", "10"
TAMANOS = [20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000]
KS = [2, 4, 8, 16, 32, 64]
HILOS = sorted({1, max(1, args.cores // 2), args.cores, args.cores * 2})

os.makedirs(args.carpeta_datos, exist_ok=True)
salida_tmp = os.path.join(args.carpeta_datos, "salida_tmp.csv")

# ---------------- Configuraciones a probar: (versión, k, hilos) ----------------
# k = 0 cuando no aplica (serial y p1)
configuraciones = [("serial", 0, 1)]
configuraciones += [("p1", 0, h) for h in HILOS]
configuraciones += [("p2", k, h) for k in KS for h in HILOS]

# ---------------- Generar los archivos de entrada que falten ----------------
for n in TAMANOS:
    ruta = os.path.join(args.carpeta_datos, f"{n}_data.csv")
    if not os.path.exists(ruta):
        np.savetxt(ruta, generar(n), delimiter=",", fmt="%.3f")

# ---------------- Retomar si ya había resultados ----------------
ya_hechas = set()
if os.path.exists(args.salida_csv):
    with open(args.salida_csv) as f:
        for fila in csv.DictReader(f):
            ya_hechas.add((fila["version"], int(fila["k"]), int(fila["n_puntos"]),
                           int(fila["hilos"]), int(fila["iteracion"])))

archivo_nuevo = not os.path.exists(args.salida_csv)
salida = open(args.salida_csv, "a", newline="")
escritor = csv.writer(salida)
if archivo_nuevo:
    escritor.writerow(["version", "k", "n_puntos", "hilos", "iteracion", "tiempo_segundos", "salida_ok"])


def huella(ruta):
    """md5 del archivo de salida: sirve para saber si dos salidas son idénticas."""
    return hashlib.md5(open(ruta, "rb").read()).hexdigest()


# ---------------- Experimento ----------------
inicio = time.time()
huella_serial = {}   # huella de la salida serial de cada tamaño
hechas = 0
total = len(configuraciones) * args.iters * len(TAMANOS)

for iteracion in range(1, args.iters + 1):
    for n in TAMANOS:
        entrada = os.path.join(args.carpeta_datos, f"{n}_data.csv")
        for version, k, hilos in configuraciones:
            hechas += 1
            if (version, k, n, hilos, iteracion) in ya_hechas:
                continue

            resultado = subprocess.run(
                [args.ejecutable, entrada, salida_tmp, EPS, MIN_SAMPLES, version, str(max(k, 1)), str(hilos)],
                capture_output=True, text=True, check=True)
            tiempo = resultado.stdout.strip()

            # La primera corrida de cada tamaño es la serial: su salida es la referencia
            h = huella(salida_tmp)
            if version == "serial" and n not in huella_serial:
                huella_serial[n] = h
            salida_ok = "" if n not in huella_serial else int(h == huella_serial[n])

            escritor.writerow([version, k, n, hilos, iteracion, tiempo, salida_ok])
            salida.flush()
    print(f"ronda {iteracion}/{args.iters}  [{hechas}/{total}]  {time.time() - inicio:.0f}s", flush=True)

salida.close()
print("LISTO", f"{time.time() - inicio:.0f}s")
