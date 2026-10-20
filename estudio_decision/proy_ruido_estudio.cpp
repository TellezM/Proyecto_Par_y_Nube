// COPIA DE ESTUDIO (no es el código final de la entrega; el código final es ../proy_ruido.cpp).
// Es el mismo programa pero con las variantes que se compararon para decidir cómo paralelizar:
//   p1i   = P1 con el ciclo interior en paralelo (reduction)
//   p2idx = P2 con listas de índices por región (primera versión de P2)
//   p2pts = P2 con el reparto por punto en lugar de por región
// Se compila con:  g++ -O2 -fopenmp -DESTUDIO_SCHEDULE proy_ruido_estudio.cpp -o dbscan_estudio
// DBSCAN paralelo para la detección de ruido (outliers) con OpenMP
// Cómputo Paralelo y en la Nube - ITAM
//
// Uso:
//   ./dbscan <in.csv> <out.csv> <eps> <min_samples> <version> <k> <hilos>
//   version: serial | p1 | p2   (p1i, p2idx y p2pts son variantes que solo se usaron en el estudio de decisión)
//
// Etiquetas de salida:  0 = ruido,  1 = core de primer orden,  2 = core de segundo orden

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <string>
#include <chrono>
#include <algorithm>

using namespace std;

// Planificación de las iteraciones de los ciclos paralelos.
// En el estudio de decisión se compila con -DESTUDIO_SCHEDULE y se elige con OMP_SCHEDULE.
#ifdef ESTUDIO_SCHEDULE
    #define SCHED_P1 schedule(runtime)
    #define SCHED_P2 schedule(runtime)
#else
    #define SCHED_P1 schedule(static)
    #define SCHED_P2 schedule(dynamic, 1)
#endif

// Estructura simple para evitar pasar muchos parámetros
struct Config {
    string inputFile;
    string outputFile;
    double epsilon;
    int min_samples;
    string version;
    int k;
    int num_threads;
};

// Lee el CSV asumiendo formato "x,y" sin encabezados
void leerCSV(const string& filename, vector<double>& x, vector<double>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo de entrada: " << filename << endl;
        exit(1);
    }
    string line, val;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // archivos de Windows
        if (line.empty()) continue;                                // líneas vacías
        stringstream ss(line);
        getline(ss, val, ',');
        x.push_back(stod(val));
        getline(ss, val, ',');
        y.push_back(stod(val));
    }
    file.close();
}

// Escribe el resultado en formato "x,y,etiqueta"
void escribirCSV(const string& filename, const vector<double>& x, const vector<double>& y, const vector<int>& labels) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo de salida: " << filename << endl;
        exit(1);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        file << x[i] << "," << y[i] << "," << labels[i] << "\n";
    }
    file.close();
}

// Distancia euclidiana al cuadrado (evita sqrt para optimizar)
inline double dist2(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

// ---------------------------------------------------------------------------
// Las tres versiones usan el mismo esquema en dos fases:
//   Fase 1: un punto es core de primer orden si tiene >= min_samples puntos a distancia <= eps
//           (se cuenta a sí mismo, igual que sklearn). Resultado en el arreglo 'core'.
//   Fase 2: un punto que no es core es de segundo orden si tiene algún core a distancia <= eps;
//           si no, es ruido. Solo LEE 'core' y escribe en su propia posición de 'labels'.
// Separar 'core' de 'labels' evita que un hilo lea una posición que otro hilo está escribiendo,
// así que el resultado no depende del orden de ejecución de los hilos.
// ---------------------------------------------------------------------------

// --- 1. VERSIÓN SERIAL ---
void dbscan_serial(const vector<double>& x, const vector<double>& y, vector<int>& labels, double eps, int min_samples) {
    int n = x.size();
    double eps2 = eps * eps;
    vector<char> core(n, 0);

    // Fase 1: cores de primer orden
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
        }
        core[i] = (vecinos >= min_samples);
    }

    // Fase 2: cores de segundo orden y ruido
    for (int i = 0; i < n; ++i) {
        if (core[i]) { labels[i] = 1; continue; }
        bool es_borde = false;
        for (int j = 0; j < n; ++j) {
            if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) { es_borde = true; break; }
        }
        labels[i] = es_borde ? 2 : 0;
    }
}

// --- 2. VERSIÓN PARALELA P1 (matriz completa, ciclo exterior en paralelo) ---
void dbscan_p1(const vector<double>& x, const vector<double>& y, vector<int>& labels, double eps, int min_samples) {
    int n = x.size();
    double eps2 = eps * eps;
    vector<char> core(n, 0);

    // Cada iteración i escribe solo core[i]: no hay condiciones de carrera.
    #pragma omp parallel for SCHED_P1
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
        }
        core[i] = (vecinos >= min_samples);
    }
    // Barrera implícita: cuando termina la fase 1, 'core' ya está completo.

    #pragma omp parallel for SCHED_P1
    for (int i = 0; i < n; ++i) {
        if (core[i]) { labels[i] = 1; continue; }
        bool es_borde = false;
        for (int j = 0; j < n; ++j) {
            if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) { es_borde = true; break; }
        }
        labels[i] = es_borde ? 2 : 0;
    }
}

// ---------------------------------------------------------------------------
// P2 (versión final): la matriz se DIVIDE de verdad en k regiones. Los puntos se reordenan por
// región (ordenamiento por conteo, O(n)) y cada región queda en un bloque contiguo de memoria:
// la región c ocupa las posiciones inicio[c] .. inicio[c+1]-1 de los arreglos xs, ys.
// Las celdas de una misma fila de la cuadrícula quedan una tras otra, así que el halo de una
// fila de celdas es un único rango contiguo.
// ---------------------------------------------------------------------------
struct CuadriculaOrd {
    int rows, cols;
    int halo_r, halo_c;
    vector<int> inicio;      // k+1 posiciones: la región c va de inicio[c] a inicio[c+1]
    vector<int> orden;       // orden[p] = índice original del punto que quedó en la posición p
    vector<double> xs, ys;   // coordenadas reordenadas por región
};

CuadriculaOrd construir_cuadricula_ord(const vector<double>& x, const vector<double>& y, double eps, int k) {
    int n = x.size();
    CuadriculaOrd g;
    g.rows = 1; g.cols = k;
    for (int r = (int)sqrt((double)k); r >= 1; --r) {
        if (k % r == 0) { g.rows = r; g.cols = k / r; break; }
    }
    double min_x = *min_element(x.begin(), x.end());
    double max_x = *max_element(x.begin(), x.end());
    double min_y = *min_element(y.begin(), y.end());
    double max_y = *max_element(y.begin(), y.end());
    double cell_w = (max_x - min_x + 0.0001) / g.cols;
    double cell_h = (max_y - min_y + 0.0001) / g.rows;
    g.halo_c = (int)ceil(eps / cell_w);
    g.halo_r = (int)ceil(eps / cell_h);

    // Región de cada punto y conteo de puntos por región
    vector<int> celda(n);
    g.inicio.assign(k + 1, 0);
    for (int i = 0; i < n; ++i) {
        int c = min((int)((x[i] - min_x) / cell_w), g.cols - 1);
        int r = min((int)((y[i] - min_y) / cell_h), g.rows - 1);
        celda[i] = r * g.cols + c;
        g.inicio[celda[i] + 1]++;
    }
    for (int c = 0; c < k; ++c) g.inicio[c + 1] += g.inicio[c];   // sumas acumuladas
    // Colocar cada punto en el bloque de su región
    vector<int> sig(g.inicio.begin(), g.inicio.end() - 1);
    g.orden.resize(n); g.xs.resize(n); g.ys.resize(n);
    for (int i = 0; i < n; ++i) {
        int p = sig[celda[i]]++;
        g.orden[p] = i; g.xs[p] = x[i]; g.ys[p] = y[i];
    }
    return g;
}

// Rangos de posiciones que hay que revisar para la región r_id: una por cada fila de celdas del halo
// (de la columna r_col - halo_c a la r_col + halo_c, que están contiguas en memoria).
void rangos_vecinos(const CuadriculaOrd& g, int r_id, vector<pair<int,int>>& out) {
    out.clear();
    int r_row = r_id / g.cols, r_col = r_id % g.cols;
    int c0 = max(0, r_col - g.halo_c), c1 = min(g.cols - 1, r_col + g.halo_c);
    for (int dr = -g.halo_r; dr <= g.halo_r; ++dr) {
        int nr = r_row + dr;
        if (nr < 0 || nr >= g.rows) continue;
        out.push_back({g.inicio[nr * g.cols + c0], g.inicio[nr * g.cols + c1 + 1]});
    }
}

// --- 3. VERSIÓN PARALELA P2 (matriz dividida en k regiones; cada hilo toma regiones completas) ---
void dbscan_p2(const vector<double>& x, const vector<double>& y, vector<int>& labels, double eps, int min_samples, int k) {
    int n = x.size();
    double eps2 = eps * eps;
    CuadriculaOrd g = construir_cuadricula_ord(x, y, eps, k);
    vector<char> core(n, 0);        // indexado por posición reordenada
    vector<int> etiq(n, 0);

    // Fase 1: cada región se procesa por separado. Cada hilo escribe solo los core[p] de su región.
    #pragma omp parallel for SCHED_P2
    for (int r_id = 0; r_id < k; ++r_id) {
        vector<pair<int,int>> vec;
        rangos_vecinos(g, r_id, vec);
        for (int p = g.inicio[r_id]; p < g.inicio[r_id + 1]; ++p) {
            int vecinos = 0;
            for (auto& rg : vec) {
                for (int q = rg.first; q < rg.second; ++q) {
                    if (dist2(g.xs[p], g.ys[p], g.xs[q], g.ys[q]) <= eps2) vecinos++;
                }
            }
            core[p] = (vecinos >= min_samples);
        }
    }
    // Barrera implícita: aquí se "unen" los resultados de las regiones. Los cores de una región
    // que se necesitan en la región de al lado ya están calculados antes de empezar la fase 2.

    // Fase 2: cores de segundo orden y ruido, con el mismo halo
    #pragma omp parallel for SCHED_P2
    for (int r_id = 0; r_id < k; ++r_id) {
        vector<pair<int,int>> vec;
        rangos_vecinos(g, r_id, vec);
        for (int p = g.inicio[r_id]; p < g.inicio[r_id + 1]; ++p) {
            if (core[p]) { etiq[p] = 1; continue; }
            bool es_borde = false;
            for (auto& rg : vec) {
                for (int q = rg.first; q < rg.second; ++q) {
                    if (core[q] && dist2(g.xs[p], g.ys[p], g.xs[q], g.ys[q]) <= eps2) { es_borde = true; break; }
                }
                if (es_borde) break;
            }
            etiq[p] = es_borde ? 2 : 0;
        }
    }

    // Regresar las etiquetas al orden original de los puntos
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n; ++p) labels[g.orden[p]] = etiq[p];
}

// --- Variante P1i (solo estudio): se paraleliza el ciclo INTERIOR con reduction ---
void dbscan_p1i(const vector<double>& x, const vector<double>& y, vector<int>& labels, double eps, int min_samples) {
    int n = x.size();
    double eps2 = eps * eps;
    vector<char> core(n, 0);

    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        #pragma omp parallel for reduction(+:vecinos)
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
        }
        core[i] = (vecinos >= min_samples);
    }
    for (int i = 0; i < n; ++i) {
        if (core[i]) { labels[i] = 1; continue; }
        int hay_core = 0;
        #pragma omp parallel for reduction(|:hay_core)
        for (int j = 0; j < n; ++j) {
            if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) hay_core = 1;
        }
        labels[i] = hay_core ? 2 : 0;
    }
}

// ---------------------------------------------------------------------------
// Cuadrícula con listas de índices (primera versión de P2, ahora variante de estudio P2idx).
// Cada región guarda los índices de sus puntos; las coordenadas siguen en el orden original.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// P2: la matriz de puntos se divide en k regiones (cuadrícula filas x columnas).
// Para decidir si un punto es core hay que mirar también las regiones vecinas (el "halo"):
// un vecino a distancia <= eps puede estar en una celda contigua. El halo se calcula en
// celdas: ceil(eps / tamaño_de_celda), así el resultado es correcto para cualquier k.
// ---------------------------------------------------------------------------
struct CuadriculaIdx {
    int rows, cols;         // regiones = rows * cols = k
    int halo_r, halo_c;     // cuántas celdas hacia cada lado hay que revisar
    vector<vector<int>> celdas;   // índices de los puntos de cada región
    vector<int> celda_de;         // región de cada punto
};

CuadriculaIdx construir_cuadricula_idx(const vector<double>& x, const vector<double>& y, double eps, int k) {
    int n = x.size();
    CuadriculaIdx g;

    // Filas y columnas lo más cuadradas posible (k = rows * cols)
    g.rows = 1; g.cols = k;
    for (int r = (int)sqrt((double)k); r >= 1; --r) {
        if (k % r == 0) { g.rows = r; g.cols = k / r; break; }
    }

    double min_x = *min_element(x.begin(), x.end());
    double max_x = *max_element(x.begin(), x.end());
    double min_y = *min_element(y.begin(), y.end());
    double max_y = *max_element(y.begin(), y.end());

    // Micro-margen para que el punto máximo no se salga de la cuadrícula
    double cell_w = (max_x - min_x + 0.0001) / g.cols;
    double cell_h = (max_y - min_y + 0.0001) / g.rows;
    g.halo_c = (int)ceil(eps / cell_w);
    g.halo_r = (int)ceil(eps / cell_h);

    g.celdas.assign(k, vector<int>());
    g.celda_de.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        int c = min((int)((x[i] - min_x) / cell_w), g.cols - 1);
        int r = min((int)((y[i] - min_y) / cell_h), g.rows - 1);
        g.celda_de[i] = r * g.cols + c;
        g.celdas[g.celda_de[i]].push_back(i);
    }
    return g;
}

// Regiones que hay que revisar para los puntos de la región r_id: ella misma y su halo
void regiones_vecinas_idx(const CuadriculaIdx& g, int r_id, vector<int>& out) {
    out.clear();
    int r_row = r_id / g.cols, r_col = r_id % g.cols;
    for (int dr = -g.halo_r; dr <= g.halo_r; ++dr) {
        for (int dc = -g.halo_c; dc <= g.halo_c; ++dc) {
            int nr = r_row + dr, nc = r_col + dc;
            if (nr >= 0 && nr < g.rows && nc >= 0 && nc < g.cols) out.push_back(nr * g.cols + nc);
        }
    }
}

// --- Variante P2idx (solo estudio): las regiones guardan LISTAS DE ÍNDICES de los puntos (primera versión de P2) ---
void dbscan_p2idx(const vector<double>& x, const vector<double>& y, vector<int>& labels, double eps, int min_samples, int k) {
    int n = x.size();
    double eps2 = eps * eps;
    CuadriculaIdx g = construir_cuadricula_idx(x, y, eps, k);
    vector<char> core(n, 0);

    // Fase 1: cada región se procesa por separado. Cada hilo escribe solo los core[i] de su región.
    #pragma omp parallel for SCHED_P2
    for (int r_id = 0; r_id < k; ++r_id) {
        vector<int> vec;
        regiones_vecinas_idx(g, r_id, vec);
        for (int i : g.celdas[r_id]) {
            int vecinos = 0;
            for (int v : vec) {
                for (int j : g.celdas[v]) {
                    if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
                }
            }
            core[i] = (vecinos >= min_samples);
        }
    }
    // Barrera implícita: aquí se "unen" los resultados de las regiones. Los cores de una región
    // que se necesitan en la región de al lado ya están calculados antes de empezar la fase 2.

    // Fase 2: cores de segundo orden y ruido, con el mismo halo
    #pragma omp parallel for SCHED_P2
    for (int r_id = 0; r_id < k; ++r_id) {
        vector<int> vec;
        regiones_vecinas_idx(g, r_id, vec);
        for (int i : g.celdas[r_id]) {
            if (core[i]) { labels[i] = 1; continue; }
            bool es_borde = false;
            for (int v : vec) {
                for (int j : g.celdas[v]) {
                    if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) { es_borde = true; break; }
                }
                if (es_borde) break;
            }
            labels[i] = es_borde ? 2 : 0;
        }
    }
}

// --- Variante P2pts (solo estudio): misma cuadrícula, pero el paralelismo es por PUNTO, no por región ---
void dbscan_p2pts(const vector<double>& x, const vector<double>& y, vector<int>& labels, double eps, int min_samples, int k) {
    int n = x.size();
    double eps2 = eps * eps;
    CuadriculaIdx g = construir_cuadricula_idx(x, y, eps, k);
    vector<char> core(n, 0);
    vector<vector<int>> vecinas(k);
    for (int r = 0; r < k; ++r) regiones_vecinas_idx(g, r, vecinas[r]);

    #pragma omp parallel for SCHED_P1
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int v : vecinas[g.celda_de[i]]) {
            for (int j : g.celdas[v]) {
                if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
            }
        }
        core[i] = (vecinos >= min_samples);
    }
    #pragma omp parallel for SCHED_P1
    for (int i = 0; i < n; ++i) {
        if (core[i]) { labels[i] = 1; continue; }
        bool es_borde = false;
        for (int v : vecinas[g.celda_de[i]]) {
            for (int j : g.celdas[v]) {
                if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) { es_borde = true; break; }
            }
            if (es_borde) break;
        }
        labels[i] = es_borde ? 2 : 0;
    }
}

// MAIN
int main(int argc, char* argv[]) {
    if (argc < 8) {
        cerr << "Uso: " << argv[0] << " <in.csv> <out.csv> <eps> <min_samples> <version: serial|p1|p2> <k> <hilos>" << endl;
        return 1;
    }

    Config cfg;
    cfg.inputFile = argv[1];
    cfg.outputFile = argv[2];
    cfg.epsilon = stod(argv[3]);
    cfg.min_samples = stoi(argv[4]);
    cfg.version = argv[5];
    cfg.k = stoi(argv[6]);
    cfg.num_threads = stoi(argv[7]);

    if (cfg.k < 1 || cfg.num_threads < 1) {
        cerr << "k y hilos deben ser >= 1" << endl;
        return 1;
    }
    omp_set_num_threads(cfg.num_threads);

    vector<double> x, y;
    leerCSV(cfg.inputFile, x, y);

    int n = x.size();
    vector<int> labels(n, 0);

    double t_start = omp_get_wtime();

    if (cfg.version == "serial") {
        dbscan_serial(x, y, labels, cfg.epsilon, cfg.min_samples);
    } else if (cfg.version == "p1") {
        dbscan_p1(x, y, labels, cfg.epsilon, cfg.min_samples);
    } else if (cfg.version == "p2") {
        dbscan_p2(x, y, labels, cfg.epsilon, cfg.min_samples, cfg.k);
    } else if (cfg.version == "p2idx") {
        dbscan_p2idx(x, y, labels, cfg.epsilon, cfg.min_samples, cfg.k);
    } else if (cfg.version == "p1i") {
        dbscan_p1i(x, y, labels, cfg.epsilon, cfg.min_samples);
    } else if (cfg.version == "p2pts") {
        dbscan_p2pts(x, y, labels, cfg.epsilon, cfg.min_samples, cfg.k);
    } else {
        cerr << "Version no reconocida. Use serial, p1 o p2." << endl;
        return 1;
    }

    double t_end = omp_get_wtime();

    // Imprime SOLO el tiempo (segundos): facilita que el script de experimentos lo recoja.
    cout << t_end - t_start << endl;

    escribirCSV(cfg.outputFile, x, y, labels);
    return 0;
}
