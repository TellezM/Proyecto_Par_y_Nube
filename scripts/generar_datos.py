"""Genera archivos de puntos (x,y) con la MISMA receta de la libreta DBSCAN_noise.ipynb:
4 grupos de puntos (make_blobs), valores positivos y 3 decimales.

Uso:  python generar_datos.py <carpeta_salida> <n1> <n2> ...
Ejemplo:  python generar_datos.py datos 4000 20000
Crea datos/4000_data.csv y datos/20000_data.csv
"""
import sys
import os
import numpy as np
from sklearn.datasets import make_blobs


def generar(n_puntos):
    """Regresa un arreglo de n_puntos filas con columnas x, y."""
    puntos, _ = make_blobs(n_samples=n_puntos, centers=4, cluster_std=0.06,
                           random_state=11, center_box=(0, 1.0))
    # Se invierten las columnas, se toma valor absoluto y se redondea a 3 decimales (como en la libreta)
    return np.round(np.abs(puntos[:, ::-1]), 3)


if __name__ == "__main__":
    carpeta = sys.argv[1]
    os.makedirs(carpeta, exist_ok=True)
    for n in map(int, sys.argv[2:]):
        ruta = os.path.join(carpeta, f"{n}_data.csv")
        np.savetxt(ruta, generar(n), delimiter=",", fmt="%.3f")
        print("generado", ruta)
