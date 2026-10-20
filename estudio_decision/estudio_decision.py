"""Estudio previo para decidir CÓMO paralelizar (se usa proy_ruido_estudio.cpp compilado con -DESTUDIO_SCHEDULE,
que deja la planificación de los ciclos en schedule(runtime) y se elige con OMP_SCHEDULE).

Uso: python estudio_decision.py <binario_estudio> <carpeta_datos> <salida.csv> [iteraciones]
Bloques:
  A  P1 con el ciclo exterior en paralelo  vs  ciclo interior con reduction
  B  P1: planificación de iteraciones (static, dynamic con varios chunk, guided)
  C  P2: planificación de las regiones (static, dynamic, guided) para varios k
  D  P2: paralelismo por región  vs  paralelismo por punto (ambas con listas de índices)
  F  P2: puntos de cada región como listas de índices (p2idx)  vs  bloques contiguos reordenados (p2)
"""
import sys, os, csv, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from generar_datos import generar
import numpy as np

exe, datos, salida = sys.argv[1], sys.argv[2], sys.argv[3]
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5
EPS, MS = "0.03", "10"
os.makedirs(datos, exist_ok=True)

def archivo(n):
    p = os.path.join(datos, f"{n}_data.csv")
    if not os.path.exists(p):
        np.savetxt(p, generar(n), delimiter=",", fmt="%.3f")
    return p

fh = open(salida, "w", newline="")
w = csv.writer(fh)
w.writerow(["bloque", "version", "schedule", "k", "n_puntos", "hilos", "iteracion", "tiempo_segundos"])

def correr(bloque, version, sched, k, n, hilos):
    env = dict(os.environ, OMP_SCHEDULE=sched)
    for it in range(1, iters + 1):
        r = subprocess.run([exe, archivo(n), os.path.join(datos, "out_estudio.csv"), EPS, MS, version, str(k), str(hilos)],
                           capture_output=True, text=True, env=env, check=True)
        w.writerow([bloque, version, sched, k, n, hilos, it, r.stdout.strip()]); fh.flush()
    print(bloque, version, sched, "k=", k, "n=", n, "hilos=", hilos, flush=True)

# Línea base serial
for n in (20000, 40000, 80000, 120000):
    correr("base", "serial", "static", 1, n, 1)

# A: ciclo exterior vs interior (P1), N=20000
for h in (2, 4):
    correr("A", "p1", "dynamic,16", 1, 20000, h)
    correr("A", "p1i", "static", 1, 20000, h)

# B: planificación en P1, N=40000
for h in (2, 4):
    for s in ("static", "static,16", "dynamic,1", "dynamic,16", "dynamic,64", "guided"):
        correr("B", "p1", s, 1, 40000, h)

# C: planificación de regiones en P2, N=80000
for k in (4, 16, 64):
    for h in (2, 4):
        for s in ("static", "dynamic,1", "guided"):
            correr("C", "p2", s, k, 80000, h)

# D: región vs punto
for k in (4, 16, 64):
    for h in (2, 4):
        correr("D", "p2idx", "dynamic,1", k, 80000, h)
        correr("D", "p2pts", "dynamic,16", k, 80000, h)
# F: layout de memoria de P2 (índices vs reordenado por región), 2 hilos, varios N
for k in (4, 16, 64):
    for n in (20000, 40000, 80000, 120000):
        correr("F", "p2idx", "dynamic,1", k, n, 2)
        correr("F", "p2", "dynamic,1", k, n, 2)
fh.close()
print("LISTO")
