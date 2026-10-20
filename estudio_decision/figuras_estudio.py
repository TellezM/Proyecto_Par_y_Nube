"""Gráficas del estudio previo de decisión (estudio_decision.csv [+ estudio_compilacion.csv]).
Uso: python figuras_estudio.py <estudio_decision.csv> <carpeta_figuras>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
d0 = os.path.expanduser("~/.fonts")
if os.path.isdir(d0):
    for f in os.listdir(d0):
        if f.lower().startswith("lexend"): fm.fontManager.addfont(os.path.join(d0, f))
plt.rcParams.update({"font.family": "Lexend", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df",
                     "axes.axisbelow": True, "savefig.dpi": 200, "legend.frameon": False})
V = "#016853"
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
g = df.groupby(["bloque", "version", "schedule", "k", "n_puntos", "hilos"]).tiempo_segundos.agg(["mean", "std"]).reset_index()
base = g[g.bloque == "base"].set_index("n_puntos")["mean"]
def save(fig, n): fig.savefig(os.path.join(out, n), bbox_inches="tight", facecolor="white"); plt.close(fig)

# E1: A (exterior vs interior) y B (schedule en P1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [1, 2.1]})
A = g[g.bloque == "A"]; hs = sorted(A.hilos.unique())
w = 0.36
for i, (ver, etq, col) in enumerate([("p1", "Ciclo exterior", "#2a78d6"), ("p1i", "Ciclo interior", "#eb6834")]):
    d = A[A.version == ver].set_index("hilos").reindex(hs)
    a1.bar(np.arange(len(hs)) + (i - .5) * (w + .02), base[20000] / d["mean"], w, yerr=base[20000] / d["mean"] * d["std"] / d["mean"], color=col, label=etq, capsize=2)
a1.axhline(1, color="#8a8a85", lw=.8); a1.set_xticks(range(len(hs))); a1.set_xticklabels([f"{h} hilos" for h in hs])
a1.set_ylabel("Speedup vs. serial"); a1.set_title("A. ¿Qué ciclo paralelizar?\n(N = 20 mil)", loc="left", fontsize=9, color=V); a1.legend(fontsize=7.5, loc="upper center", ncol=2); a1.set_ylim(0, 2.6)
B = g[g.bloque == "B"]; ss = ["static", "static,16", "dynamic,1", "dynamic,16", "dynamic,64", "guided"]
cols = {2: "#2a78d6", 4: "#173f73"}
for i, h in enumerate(sorted(B.hilos.unique())):
    d = B[B.hilos == h].set_index("schedule").reindex(ss)
    a2.bar(np.arange(len(ss)) + (i - .5) * (w + .02), base[40000] / d["mean"], w, yerr=base[40000] / d["mean"] * d["std"] / d["mean"], color=cols[h], label=f"{h} hilos", capsize=2)
a2.axhline(1, color="#8a8a85", lw=.8); a2.set_xticks(range(len(ss))); a2.set_xticklabels(ss, rotation=25, ha="right", fontsize=8)
a2.set_ylabel("Speedup vs. serial"); a2.set_title("B. Planificación de iteraciones en P1\n(N = 40 mil)", loc="left", fontsize=9, color=V); a2.legend(fontsize=7.5, loc="upper center", ncol=2); a2.set_ylim(0, 2.7)
fig.tight_layout(); save(fig, "e1_ciclo_y_schedule_p1.png")

# E2: C (schedule de regiones en P2)
C = g[g.bloque == "C"]; ks = sorted(C.k.unique()); sc = ["static", "dynamic,1", "guided"]
colr = {"static": "#9bbfe8", "dynamic,1": "#1baf7a", "guided": "#eb6834"}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
for ax, h in zip(axs, sorted(C.hilos.unique())):
    for i, s in enumerate(sc):
        d = C[(C.hilos == h) & (C.schedule == s)].set_index("k").reindex(ks)
        ax.bar(np.arange(len(ks)) + (i - 1) * 0.27, base[80000] / d["mean"], 0.25, yerr=base[80000] / d["mean"] * d["std"] / d["mean"], color=colr[s], label=s, capsize=2)
    ax.axhline(1, color="#8a8a85", lw=.8); ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_title(f"{h} hilos", loc="left", fontsize=9, color=V)
axs[0].set_ylabel("Speedup vs. serial"); axs[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("C. Planificación de las regiones en P2 (N = 80 mil)", x=0.01, ha="left", fontsize=9.5, color=V)
fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "e2_schedule_p2.png")

# E3: D (región vs punto)
D = g[g.bloque == "D"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
for ax, h in zip(axs, sorted(D.hilos.unique())):
    for i, (ver, etq, col) in enumerate([("p2idx", "Por región", "#1baf7a"), ("p2pts", "Por punto", "#6250d6")]):
        d = D[(D.hilos == h) & (D.version == ver)].set_index("k").reindex(sorted(D.k.unique()))
        ax.bar(np.arange(len(d)) + (i - .5) * 0.38, base[80000] / d["mean"], 0.36, yerr=base[80000] / d["mean"] * d["std"] / d["mean"], color=col, label=etq, capsize=2)
    ax.axhline(1, color="#8a8a85", lw=.8); ax.set_xticks(range(len(d))); ax.set_xticklabels([f"k={k}" for k in sorted(D.k.unique())])
    ax.set_title(f"{h} hilos", loc="left", fontsize=9, color=V)
axs[0].set_ylabel("Speedup vs. serial"); axs[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("D. Paralelismo por región o por punto, con listas de índices (N = 80 mil)", x=0.01, ha="left", fontsize=9.5, color=V)
fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "e3_region_vs_punto.png")

# E4: layout de memoria (bloque F): índices vs reordenado por región
F = g[g.bloque == "F"]; nsF = sorted(F.n_puntos.unique())
fig, ax = plt.subplots(figsize=(6.4, 3.4)); ax.grid(True, axis="both")
cK = {4: "#8cc8ae", 16: "#2f946e", 64: "#014d3d"}
for k in (4, 16, 64):
    for ver, ls, mk in (("p2idx", "--", "o"), ("p2", "-", "s")):
        d = F[(F.k == k) & (F.version == ver)].set_index("n_puntos").reindex(nsF)
        ax.plot(range(len(nsF)), [base[n] / d.loc[n, "mean"] for n in nsF], ls=ls, marker=mk, ms=4.5, lw=1.8, color=cK[k])
ax.set_xticks(range(len(nsF))); ax.set_xticklabels([f"{n//1000} mil" for n in nsF])
ax.set_xlabel("Número de puntos"); ax.set_ylabel("Speedup vs. serial (2 hilos)")
from matplotlib.lines import Line2D
h1 = [Line2D([0], [0], color=cK[k], lw=2, label=f"k={k}") for k in (4, 16, 64)]
h2 = [Line2D([0], [0], color="#52514e", ls="--", marker="o", ms=4, label="Listas de índices"), Line2D([0], [0], color="#52514e", ls="-", marker="s", ms=4, label="Reordenado por región")]
ax.legend(handles=h1 + h2, fontsize=7.5, loc="upper left", ncol=2)
ax.set_title("F. Cómo se guardan los puntos de cada región", loc="left", fontsize=9.5, color=V)
save(fig, "e4_layout_memoria.png")
print("ok")
