"""Procesa resultados_tiempos.csv: promedia las iteraciones, calcula speedups y genera las gráficas.
Uso: python analizar_resultados.py <resultados_tiempos.csv> <carpeta_figuras> [--cores N]
Salidas: resumen_experimentos.csv y figuras PNG.
Definiciones:
  tiempo(config)   = media de las iteraciones
  speedup          = tiempo_serial(N) / tiempo(config, N)            (contra la versión serial)
  gan. paralela    = tiempo(misma versión y k, 1 hilo) / tiempo(config)    (solo el efecto de los hilos)
  gan. algorítmica = tiempo_serial(N) / tiempo(misma versión y k, 1 hilo)  (efecto de dividir en regiones)
  eficiencia       = gan. paralela / hilos
"""
import sys, os, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("figs")
ap.add_argument("--cores", type=int, default=2)
a = ap.parse_args()
os.makedirs(a.figs, exist_ok=True)

# Fuente Lexend (si está instalada en ~/.fonts); si no, matplotlib usa la fuente por defecto
for f in os.listdir(os.path.expanduser("~/.fonts")) if os.path.isdir(os.path.expanduser("~/.fonts")) else []:
    if f.lower().startswith("lexend"):
        fm.fontManager.addfont(os.path.join(os.path.expanduser("~/.fonts"), f))
plt.rcParams.update({"font.family": "Lexend", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b2b", "text.color": "#2b2b2b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "figure.dpi": 100, "savefig.dpi": 200, "legend.frameon": False})
VERDE_ITAM = "#016853"
C_SERIAL, C_P1 = "#52514e", "#eb6834"
RAMPA_K = {2: "#b7ddcd", 4: "#8cc8ae", 8: "#5eb08d", 16: "#2f946e", 32: "#0f7a5c", 64: "#014d3d"}

# ---- Leer los tiempos crudos y promediar las iteraciones de cada configuración
df = pd.read_csv(a.csv)
g = df.groupby(["version", "k", "n_puntos", "hilos"]).agg(
    tiempo=("tiempo_segundos", "mean"), desv=("tiempo_segundos", "std"),
    n_iter=("tiempo_segundos", "count")).reset_index()
# ---- Speedups y eficiencia (definiciones en el encabezado de este archivo)
serial = g[g.version == "serial"].set_index("n_puntos")["tiempo"]
g["speedup"] = g.apply(lambda r: serial[r.n_puntos] / r.tiempo, axis=1)
uno = g[g.hilos == 1].set_index(["version", "k", "n_puntos"])["tiempo"]
g["gan_paralela"] = g.apply(lambda r: uno[(r.version, r.k, r.n_puntos)] / r.tiempo, axis=1)
g["gan_algoritmica"] = g.apply(lambda r: serial[r.n_puntos] / uno[(r.version, r.k, r.n_puntos)] if r.version != "serial" else 1.0, axis=1)
g["eficiencia"] = g["gan_paralela"] / g["hilos"]
g.round(4).to_csv(os.path.join(os.path.dirname(os.path.abspath(a.csv)), "resumen_experimentos.csv"), index=False)

# ---- Listas de valores que se usan en las gráficas, y funciones auxiliares
ns = sorted(g.n_puntos.unique()); nmax = ns[-1]
hs = sorted(g[g.version != "serial"].hilos.unique())
ks = sorted(g[g.version == "p2"].k.unique())
hcores = a.cores if a.cores in hs else hs[-1]
def sel(v, k=None, n=None, h=None):
    d = g[g.version == v]
    if k is not None: d = d[d.k == k]
    if n is not None: d = d[d.n_puntos == n]
    if h is not None: d = d[d.hilos == h]
    return d.sort_values(["hilos", "k", "n_puntos"])
def nfmt(n): return f"{n//1000} mil"
def save(fig, name):
    fig.savefig(os.path.join(a.figs, name), bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- F1: speedup vs hilos (N máximo)
fig, ax = plt.subplots(figsize=(6.3, 3.6))
xs = range(len(hs))
for k in ks:
    d = sel("p2", k, nmax).set_index("hilos").reindex(hs)
    ax.errorbar(xs, d.speedup, yerr=d.speedup * (d.desv / d.tiempo), color=RAMPA_K[k], marker="o", ms=5, lw=1.8, capsize=2, label=f"P2, k={k}")
d = sel("p1", n=nmax).set_index("hilos").reindex(hs)
ax.errorbar(xs, d.speedup, yerr=d.speedup * (d.desv / d.tiempo), color=C_P1, marker="s", ms=5, lw=2, capsize=2, label="P1")
ax.axhline(1.5, color="#8a8a85", ls=":", lw=1.2); ax.text(len(hs) - 1 + 0.05, 1.5, " meta 1.5", va="bottom", fontsize=8, color="#52514e")
ax.axhline(1.0, color="#8a8a85", lw=0.8)
ax.set_xticks(list(xs)); ax.set_xticklabels([f"{h}" + (" (= núcleos)" if h == a.cores else (" (2×)" if h == 2 * a.cores else "")) for h in hs])
ax.set_xlabel("Número de hilos"); ax.set_ylabel("Speedup vs. serial")
ax.set_title(f"Speedup según el número de hilos (N = {nmax:,})".replace(",", ","), loc="left", fontsize=10, color=VERDE_ITAM)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
save(fig, "f1_speedup_vs_hilos.png")

# ---- F2: speedup vs k (N máximo), una línea por número de hilos
fig, ax = plt.subplots(figsize=(6.3, 3.6))
colh = {1: "#9bbfe8", 2: "#2a78d6", 4: "#173f73"}
for i, h in enumerate(hs):
    d = sel("p2", n=nmax, h=h).set_index("k").reindex(ks)
    ax.errorbar(range(len(ks)), d.speedup, yerr=d.speedup * (d.desv / d.tiempo), marker="o", ms=5, lw=1.8, capsize=2,
                color=colh.get(h, "#2a78d6"), label=f"P2, {h} hilo{'s' if h > 1 else ''}")
    p1 = sel("p1", n=nmax, h=h)
    ax.axhline(p1.speedup.iloc[0], color=colh.get(h, "#2a78d6"), ls="--", lw=1, alpha=.8)
ax.plot([], [], color="#52514e", ls="--", lw=1, label="P1 (misma gama de hilos)")
ax.axhline(1.5, color="#8a8a85", ls=":", lw=1.2)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
ax.set_xlabel("k (número de regiones)"); ax.set_ylabel("Speedup vs. serial")
ax.set_title(f"Speedup de P2 según k (N = {nmax:,})", loc="left", fontsize=10, color=VERDE_ITAM)
ax.legend(loc="upper left", fontsize=8)
save(fig, "f2_speedup_vs_k.png")

# ---- F3: speedup vs N (hilos = núcleos)
fig, ax = plt.subplots(figsize=(6.3, 3.6))
d = sel("p1", h=hcores).set_index("n_puntos").reindex(ns)
ax.plot(range(len(ns)), d.speedup, color=C_P1, marker="s", ms=5, lw=2, label="P1")
for k in (4, 16, 64):
    if k in ks:
        d = sel("p2", k, h=hcores).set_index("n_puntos").reindex(ns)
        ax.plot(range(len(ns)), d.speedup, color=RAMPA_K[k], marker="o", ms=5, lw=1.8, label=f"P2, k={k}")
ax.axhline(1.5, color="#8a8a85", ls=":", lw=1.2)
ax.set_xticks(range(len(ns))); ax.set_xticklabels([nfmt(n) for n in ns])
ax.set_xlabel("Número de puntos"); ax.set_ylabel("Speedup vs. serial")
ax.set_title(f"Speedup según el tamaño de la entrada ({hcores} hilos)", loc="left", fontsize=10, color=VERDE_ITAM)
ax.legend(loc="upper left", fontsize=8, ncol=2)
save(fig, "f3_speedup_vs_n.png")

# ---- F4: tiempos absolutos (log-log)
fig, ax = plt.subplots(figsize=(6.3, 3.6))
d = sel("serial").set_index("n_puntos").reindex(ns)
ax.plot(ns, d.tiempo, color=C_SERIAL, marker="o", ms=5, lw=2, label="Serial")
d = sel("p1", h=hcores).set_index("n_puntos").reindex(ns)
ax.plot(ns, d.tiempo, color=C_P1, marker="s", ms=5, lw=2, label=f"P1, {hcores} hilos")
if 64 in ks:
    d = sel("p2", 64, h=hcores).set_index("n_puntos").reindex(ns)
    ax.plot(ns, d.tiempo, color=RAMPA_K[64], marker="o", ms=5, lw=2, label=f"P2, k=64, {hcores} hilos")
ref = serial.iloc[0] * (np.array(ns) / ns[0]) ** 2
ax.plot(ns, ref, color="#8a8a85", ls=":", lw=1.2, label="referencia $N^2$")
ax.set_xscale("log"); ax.set_yscale("log")
sub = [n for n in ns if n in (20000, 40000, 80000, 120000, 160000, 200000)] or ns
ax.set_xticks(sub); ax.set_xticklabels([nfmt(n) for n in sub]); ax.minorticks_off()
ax.set_xlabel("Número de puntos"); ax.set_ylabel("Tiempo (s)")
ax.set_title("Tiempo de ejecución (escala logarítmica)", loc="left", fontsize=10, color=VERDE_ITAM)
ax.legend(loc="upper left", fontsize=8)
save(fig, "f4_tiempos.png")

# ---- F5: mapa de calor de P2 (k x N)
fig, ax = plt.subplots(figsize=(6.3, 3.0))
M = np.array([[sel("p2", k, n, hcores).speedup.iloc[0] for n in ns] for k in ks])
im = ax.imshow(M, cmap="Greens", aspect="auto", vmin=0, vmax=max(M.max(), 2))
ax.grid(False)
ax.set_xticks(range(len(ns))); ax.set_xticklabels([nfmt(n) for n in ns]); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
for i in range(len(ks)):
    for j in range(len(ns)):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.max() * 0.6 else "#1f1f1f")
ax.set_xlabel("Número de puntos"); ax.set_ylabel("k")
ax.set_title(f"Speedup de P2 (k × N, {hcores} hilos)", loc="left", fontsize=10, color=VERDE_ITAM)
for s in ax.spines.values(): s.set_visible(False)
save(fig, "f5_mapa_calor_p2.png")

# ---- F6: ganancia algorítmica vs ganancia paralela (P2, N máximo)
fig, ax = plt.subplots(figsize=(6.3, 3.4))
w = 0.36; xk = np.arange(len(ks))
alg = [sel("p2", k, nmax, hcores).gan_algoritmica.iloc[0] for k in ks]
par = [sel("p2", k, nmax, hcores).gan_paralela.iloc[0] for k in ks]
ax.bar(xk - w / 2 - 0.01, alg, w, color="#1baf7a", label="Por dividir en regiones (P2 con 1 hilo vs. serial)")
ax.bar(xk + w / 2 + 0.01, par, w, color="#2a78d6", label=f"Por los hilos (P2 con {hcores} hilos vs. P2 con 1 hilo)")
ax.axhline(1, color="#8a8a85", lw=0.8)
ax.set_xticks(xk); ax.set_xticklabels(ks); ax.set_xlabel("k"); ax.set_ylabel("Factor de mejora")
ax.set_title(f"De dónde viene el speedup de P2 (N = {nmax:,})", loc="left", fontsize=10, color=VERDE_ITAM)
ax.legend(loc="upper left", fontsize=7.5)
save(fig, "f6_descomposicion_p2.png")

# ---- F7: eficiencia paralela
fig, ax = plt.subplots(figsize=(6.3, 3.2))
etq = ["P1"] + [f"P2 k={k}" for k in ks]
ef = [sel("p1", n=nmax, h=hcores).eficiencia.iloc[0]] + [sel("p2", k, nmax, hcores).eficiencia.iloc[0] for k in ks]
cols = [C_P1] + [RAMPA_K[k] for k in ks]
ax.bar(range(len(etq)), ef, 0.6, color=cols)
for i, v in enumerate(ef): ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(1, color="#8a8a85", lw=0.8)
ax.set_xticks(range(len(etq))); ax.set_xticklabels(etq, fontsize=8); ax.set_ylabel("Eficiencia"); ax.set_ylim(0, 1.15)
ax.set_title(f"Eficiencia paralela con {hcores} hilos (N = {nmax:,})", loc="left", fontsize=10, color=VERDE_ITAM)
save(fig, "f7_eficiencia.png")

# ---- Resumen en pantalla: mejor speedup y revisión de que todas las salidas coincidieron con la serial
mejor = g[(g.version != "serial")].sort_values("speedup", ascending=False).iloc[0]
print("N máx:", nmax, "| hilos:", hs, "| k:", ks)
print("Mejor speedup:", mejor.version, "k=", mejor.k, "N=", mejor.n_puntos, "hilos=", mejor.hilos, round(mejor.speedup, 2))
print("Iteraciones por config (mín/máx):", g.n_iter.min(), g.n_iter.max())
print("Salidas incorrectas:", int((df.salida_ok == 0).sum()) if "salida_ok" in df else "n/d")
