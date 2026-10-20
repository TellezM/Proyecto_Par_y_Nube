"""Balance de carga de las regiones de P2: cuántos puntos tiene cada región y cuánto trabajo (comparaciones de la fase 1)
le toca a cada una. Con ese costo se simula el reparto de `schedule(dynamic, 1)` entre H hilos y se obtiene la
eficiencia máxima que permite el reparto de regiones (sin contar nada más).
Uso: python balance_regiones.py <archivo_puntos.csv> <salida.csv> [hilos=2]"""
import sys, math, heapq
import numpy as np, pandas as pd
pts = np.loadtxt(sys.argv[1], delimiter=","); H = int(sys.argv[3]) if len(sys.argv) > 3 else 2
eps = 0.03
filas = []
for k in (2, 4, 8, 16, 32, 64):
    rows, cols = 1, k
    for r in range(int(math.sqrt(k)), 0, -1):
        if k % r == 0: rows, cols = r, k // r; break
    x, y = pts[:, 0], pts[:, 1]
    cw = (x.max() - x.min() + 0.0001) / cols; ch = (y.max() - y.min() + 0.0001) / rows
    hc, hr = math.ceil(eps / cw), math.ceil(eps / ch)
    c = np.minimum(((x - x.min()) / cw).astype(int), cols - 1); r = np.minimum(((y - y.min()) / ch).astype(int), rows - 1)
    cnt = np.bincount(r * cols + c, minlength=k).reshape(rows, cols)
    costo = []
    for rr in range(rows):
        for cc in range(cols):
            halo = cnt[max(0, rr - hr):rr + hr + 1, max(0, cc - hc):cc + hc + 1].sum()
            costo.append(cnt[rr, cc] * halo)
    # dynamic,1: cada hilo que se libera toma la siguiente región (en orden)
    libres = [0.0] * H; heapq.heapify(libres)
    for cst in costo:
        t = heapq.heappop(libres); heapq.heappush(libres, t + cst)
    makespan = max(libres); total = sum(costo)
    filas.append({"k": k, "regiones": f"{rows}x{cols}", "puntos_min": int(cnt.min()), "puntos_max": int(cnt.max()),
                  "puntos_medio": round(cnt.mean()), "costo_max_sobre_medio": round(max(costo) / (total / k), 2),
                  "eficiencia_maxima_modelo": round(total / (H * makespan), 3)})
pd.DataFrame(filas).to_csv(sys.argv[2], index=False); print(pd.DataFrame(filas).to_string(index=False))
