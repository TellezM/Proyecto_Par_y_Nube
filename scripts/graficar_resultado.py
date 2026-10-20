"""Gráfica de los puntos de entrada y del resultado (equivale a las celdas de DBSCAN_noise.ipynb).
Uso: python visualizar_resultado.py <entrada.csv> <salida.csv> <figura.png>"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.cluster import DBSCAN
d0 = os.path.expanduser("~/.fonts")
if os.path.isdir(d0):
    for f in os.listdir(d0):
        if f.lower().startswith("lexend"): fm.fontManager.addfont(os.path.join(d0, f))
plt.rcParams.update({"font.family": "Lexend", "font.size": 9})
pts = np.loadtxt(sys.argv[1], delimiter=","); res = np.loadtxt(sys.argv[2], delimiter=",")
sk = DBSCAN(eps=0.03, min_samples=10).fit_predict(pts)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
ax[0].scatter(pts[:, 0], pts[:, 1], c=np.where(sk == -1, 0, 1), s=6, cmap="viridis")
ax[0].set_title("Referencia: sklearn.DBSCAN\n(morado = ruido)", fontsize=9, loc="left", color="#016853")
nombres = {0: ("Ruido", "#eb6834"), 1: ("Core de primer orden", "#016853"), 2: ("Core de segundo orden", "#2a78d6")}
for e, (n, c) in nombres.items():
    m = res[:, 2] == e
    ax[1].scatter(res[m, 0], res[m, 1], s=6 if e == 1 else 12, c=c, label=f"{n} ({m.sum()})")
ax[1].set_title("Resultado del programa en C++", fontsize=9, loc="left", color="#016853")
ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, markerscale=2)
for a_ in ax: a_.set_xticks([]); a_.set_yticks([]); [s.set_visible(False) for s in a_.spines.values()]
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight", facecolor="white")
