# DBSCAN paralelo para detectar ruido (OpenMP)

Proyecto de Cómputo Paralelo y en la Nube (ITAM). Autores: Marcio Tellez e Irene Escudero.

Detecta puntos de ruido (outliers) con la idea de densidad de DBSCAN. El programa está en `proy_ruido.cpp`
y tiene tres versiones:

| Versión | Qué hace |
|---|---|
| `serial` | Algoritmo ingenuo en dos fases, compara cada punto contra todos. |
| `p1` | Matriz completa; el ciclo exterior se reparte entre hilos con `omp parallel for`. |
| `p2` | Matriz dividida en `k` regiones (cuadrícula con halo); cada hilo toma regiones completas. |

## Qué hay en cada carpeta

| Carpeta o archivo | Contenido |
|---|---|
| `proy_ruido.cpp` | Código final del algoritmo (comentado en español). |
| `DBSCAN_noise.ipynb` | Libreta del curso (generar datos, graficar y calcular speedups). |
| `scripts/` | Programas de apoyo: generar datos, correr el experimento, analizar resultados, probar correctitud. |
| `resultados/` | Datos: tiempos de todas las corridas y su resumen, datos de los estudios, hardware. |
| `imagenes/` | Gráficas y capturas que aparecen en los documentos (`experimento/`, `estudio_decision/`, `capturas/`). |
| `estudio_decision/` | Código del estudio previo con el que se decidió cómo paralelizar. |

## Entregables

Los dos documentos de la entrega están en Google Docs (así se pueden corregir sin actualizar el repositorio):

- [Descripción del código y de las estrategias de paralelización](https://docs.google.com/document/d/1SjLe5U_sEavYV4bmmF4A8xFXu8swvr04/edit?usp=sharing)
- [Evaluación experimental de desempeño](https://docs.google.com/document/d/1V6SehjfWGXJ1DAQDF3ad0bw01f9F4jOW/edit?usp=sharing)

## Compilar y ejecutar

```bash
g++ -O2 -fopenmp proy_ruido.cpp -o dbscan          # en Windows (MSYS2/MinGW) crea dbscan.exe
python scripts/generar_datos.py datos 4000          # crea datos/4000_data.csv
./dbscan datos/4000_data.csv salida.csv 0.03 10 p2 16 4
#        entrada             salida     eps  min  ver k hilos
```

El programa imprime solo el tiempo del algoritmo en segundos. La salida tiene una línea por punto:
`x,y,etiqueta` (0 ruido, 1 core de primer orden, 2 core de segundo orden). Se grafica con la libreta.

## Reproducir los experimentos

```bash
pip install numpy pandas matplotlib scikit-learn
python scripts/prueba_correctitud.py ./dbscan
python scripts/correr_experimento.py ./dbscan datos resultados/resultados_tiempos.csv --cores <núcleos>
python scripts/analizar_resultados.py resultados/resultados_tiempos.csv imagenes/experimento --cores <núcleos>
```

`--cores` es el número de núcleos de la computadora (los hilos del experimento son 1, cores/2, cores y cores×2).
En Windows: `Get-CimInstance Win32_Processor | Select NumberOfCores,NumberOfLogicalProcessors`.
El experimento completo tarda varias horas; si se interrumpe, al volver a correr el mismo comando continúa donde se quedó.

El estudio previo de decisión está en `estudio_decision/` (ver los comentarios al inicio de cada archivo).
