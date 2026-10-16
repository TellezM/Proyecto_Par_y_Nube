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

// Estructura simple para evitar pasar muchos parámetros
struct Config {
    string inputFile;
    string outputFile;
    float epsilon;
    int min_samples;
    string version;
    int k;
    int num_threads;
};




// Lee el CSV asumiendo formato "x,y" sin encabezados
void leerCSV(const string& filename, vector<float>& x, vector<float>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo de entrada: " << filename << endl;
        exit(1);
    }
    string line, val;
    while (getline(file, line)) {
        stringstream ss(line);
        getline(ss, val, ',');
        x.push_back(stof(val));
        getline(ss, val, ',');
        y.push_back(stof(val));
    }
    file.close();
}

// Escribe el resultado en formato "x,y,etiqueta"
void escribirCSV(const string& filename, const vector<float>& x, const vector<float>& y, const vector<int>& labels) {
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




// FUNCIÓN DE DISTANCIA


// Distancia euclidiana al cuadrado (evita sqrt para optimizar)
inline float dist2(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}



// ALGORITMOS DE DETECCIÓN DE RUIDO DBSCAN



// --- 1. VERSIÓN SERIAL ---
void dbscan_serial(const vector<float>& x, const vector<float>& y, vector<int>& labels, float eps, int min_samples) {
    int n = x.size();
    float eps2 = eps * eps;

    // Fase 1: Encontrar Cores de primer orden
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                vecinos++;
            }
        }
        if (vecinos >= min_samples) {
            labels[i] = 1; // 1 = Core
        }
    }

    // Fase 2: Encontrar Cores de segundo orden (Bordes)
    for (int i = 0; i < n; ++i) {
        if (labels[i] != 1) { // Si no es core de primer orden
            bool es_borde = false;
            for (int j = 0; j < n; ++j) {
                // Buscamos si algún vecino a distancia <= eps es un core (1)
                if (labels[j] == 1 && dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                    es_borde = true;
                    break;
                }
            }
            if (es_borde) {
                labels[i] = 2; // 2 = Borde (Core de segundo orden)
            } else {
                labels[i] = 0; // 0 = Ruido definitivo
            }
        }
    }
}

// --- 2. VERSIÓN PARALELA P1 (Paralelización ingenua directa) ---
void dbscan_p1(const vector<float>& x, const vector<float>& y, vector<int>& labels, float eps, int min_samples) {
    int n = x.size();
    float eps2 = eps * eps;

    // Fase 1: Paralelización del ciclo exterior.
    // Cada hilo escribe en su propio 'labels[i]', sin condiciones de carrera.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                vecinos++;
            }
        }
        if (vecinos >= min_samples) {
            labels[i] = 1;
        }
    }

    // Barrera implícita al final del pragma omp parallel for.

    // Fase 2: Identificar bordes. De nuevo, lecturas seguras y escrituras independientes.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        if (labels[i] != 1) {
            bool es_borde = false;
            for (int j = 0; j < n; ++j) {
                if (labels[j] == 1 && dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                    es_borde = true;
                    break;
                }
            }
            labels[i] = es_borde ? 2 : 0;
        }
    }
}

// --- 3. VERSIÓN PARALELA P2 (Grid Partitioning) ---
void dbscan_p2(const vector<float>& x, const vector<float>& y, vector<int>& labels, float eps, int min_samples, int k) {
    int n = x.size();
    float eps2 = eps * eps;

    // Calcular filas y columnas de la cuadrícula
    int rows = 1, cols = k;
    for (int r = sqrt(k); r >= 1; --r) {
        if (k % r == 0) {
            rows = r;
            cols = k / r;
            break;
        }
    }

    // Encontrar límites espaciales (bounding box)
    float min_x = *min_element(x.begin(), x.end());
    float max_x = *max_element(x.begin(), x.end());
    float min_y = *min_element(y.begin(), y.end());
    float max_y = *max_element(y.begin(), y.end());

    // Añadir un micro-margen para que el punto máximo no se salga del grid
    float cell_w = (max_x - min_x + 0.0001f) / cols;
    float cell_h = (max_y - min_y + 0.0001f) / rows;

    // Asignar puntos a celdas
    vector<vector<int>> grid(k);
    for (int i = 0; i < n; ++i) {
        int c = (x[i] - min_x) / cell_w;
        int r = (y[i] - min_y) / cell_h;
        int region_id = r * cols + c;
        grid[region_id].push_back(i);
    }

    // Fase 1: Procesar cada región en paralelo
    #pragma omp parallel for schedule(dynamic)
    for (int r_id = 0; r_id < k; ++r_id) {
        int r_row = r_id / cols;
        int r_col = r_id % cols;

        // Construir la vecindad de esta región (halo implícito al revisar celdas vecinas)
        vector<int> celdas_vecinas;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                int nr = r_row + dr;
                int nc = r_col + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    celdas_vecinas.push_back(nr * cols + nc);
                }
            }
        }

        // Evaluar puntos de la región actual
        for (int i : grid[r_id]) {
            int vecinos = 0;
            // Buscar vecinos en la región y su halo (regiones adyacentes)
            for (int vecino_id : celdas_vecinas) {
                for (int j : grid[vecino_id]) {
                    if (dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                        vecinos++;
                    }
                }
            }
            if (vecinos >= min_samples) {
                labels[i] = 1;
            }
        }
    }
    // Barrera de sincronización implícita de OpenMP

    // Fase 2: Ruido y bordes
    #pragma omp parallel for schedule(dynamic)
    for (int r_id = 0; r_id < k; ++r_id) {
        int r_row = r_id / cols;
        int r_col = r_id % cols;

        vector<int> celdas_vecinas;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                int nr = r_row + dr;
                int nc = r_col + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    celdas_vecinas.push_back(nr * cols + nc);
                }
            }
        }

        for (int i : grid[r_id]) {
            if (labels[i] != 1) {
                bool es_borde = false;
                for (int vecino_id : celdas_vecinas) {
                    for (int j : grid[vecino_id]) {
                        if (labels[j] == 1 && dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                            es_borde = true;
                            goto salir_busqueda; // Optimización de salida rápida
                        }
                    }
                }
                salir_busqueda:
                labels[i] = es_borde ? 2 : 0;
            }
        }
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
    cfg.epsilon = stof(argv[3]);
    cfg.min_samples = stoi(argv[4]);
    cfg.version = argv[5];
    cfg.k = stoi(argv[6]);
    cfg.num_threads = stoi(argv[7]);

    omp_set_num_threads(cfg.num_threads);

    vector<float> x, y;
    leerCSV(cfg.inputFile, x, y);
    
    int n = x.size();
    vector<int> labels(n, 0); // Inicializados en 0 por defecto

    double t_start = omp_get_wtime();

    if (cfg.version == "serial") {
        dbscan_serial(x, y, labels, cfg.epsilon, cfg.min_samples);
    } else if (cfg.version == "p1") {
        dbscan_p1(x, y, labels, cfg.epsilon, cfg.min_samples);
    } else if (cfg.version == "p2") {
        dbscan_p2(x, y, labels, cfg.epsilon, cfg.min_samples, cfg.k);
    } else {
        cerr << "Version no reconocida. Use serial, p1 o p2." << endl;
        return 1;
    }

    double t_end = omp_get_wtime();
    
    // Imprimir SOLO el tiempo, esto facilita que el script recolecte el output.
    cout << t_end - t_start << endl;

    escribirCSV(cfg.outputFile, x, y, labels);

    return 0;
}