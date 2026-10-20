"""Prueba de correctitud: compara el resultado del programa contra sklearn.DBSCAN y
compara p1 y p2 (con varios k y varios hilos) contra la versión serial.

Uso:  python prueba_correctitud.py <ejecutable>
Ejemplo:  python prueba_correctitud.py ./dbscan
"""
import subprocess
import sys
import tempfile
import os
import numpy as np
from sklearn.cluster import DBSCAN
from generar_datos import generar

ejecutable = sys.argv[1]
carpeta_tmp = tempfile.mkdtemp()


def correr(entrada, version, eps, min_samples, k, hilos):
    """Corre el programa en C++ y regresa las etiquetas que produjo."""
    salida = os.path.join(carpeta_tmp, "salida.csv")
    subprocess.run([ejecutable, entrada, salida, str(eps), str(min_samples), version, str(k), str(hilos)],
                   check=True, capture_output=True)
    return np.loadtxt(salida, delimiter=",")[:, 2].astype(int)


# Casos de prueba: (nombre, puntos, eps, min_samples)
rng = np.random.default_rng(0)
casos = [
    ("blobs_4000",       generar(4000),  0.03, 10),
    ("blobs_20000",      generar(20000), 0.03, 10),
    ("eps_grande",       generar(4000),  0.30, 10),                                     # eps mayor que una celda (halo > 1)
    ("uniforme",         np.round(rng.random((3000, 2)), 3), 0.02, 5),                  # puntos sin grupos
    ("muchos_repetidos", np.round(rng.random((2000, 2)) * 0.2, 2), 0.01, 4),           # puntos duplicados
]

total = 0
fallos = 0
for nombre, puntos, eps, min_samples in casos:
    entrada = os.path.join(carpeta_tmp, nombre + ".csv")
    np.savetxt(entrada, puntos, delimiter=",", fmt="%.3f")

    # 1) serial contra sklearn (ruido y cores)
    sk = DBSCAN(eps=eps, min_samples=min_samples).fit(puntos)
    serial = correr(entrada, "serial", eps, min_samples, 1, 1)

    total += 1
    if not np.array_equal(serial == 0, sk.labels_ == -1):
        fallos += 1
        print("FALLA (ruido serial vs sklearn):", nombre)

    cores_sk = np.zeros(len(puntos), dtype=bool)
    cores_sk[sk.core_sample_indices_] = True
    total += 1
    if not np.array_equal(serial == 1, cores_sk):
        fallos += 1
        print("FALLA (cores serial vs sklearn):", nombre)

    # 2) p1 y p2 contra serial, con distintos k y número de hilos
    for version in ["p1", "p2"]:
        for k in [1, 2, 4, 8, 16, 32, 64, 7]:           # 7 no es cuadrado: prueba una cuadrícula 1x7
            for hilos in [1, 2, 4, 8]:
                if version == "p1" and k != 2:           # p1 no usa k, con un valor basta
                    continue
                total += 1
                if not np.array_equal(correr(entrada, version, eps, min_samples, k, hilos), serial):
                    fallos += 1
                    print("FALLA", nombre, version, "k =", k, "hilos =", hilos)

print(f"{total - fallos}/{total} comparaciones correctas")
sys.exit(1 if fallos else 0)
