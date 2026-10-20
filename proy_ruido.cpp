// =====================================================================================
// DBSCAN paralelo para detectar ruido (outliers) con OpenMP
// Cómputo Paralelo y en la Nube - ITAM
// Autores: Marcio Tellez e Irene Escudero
// =====================================================================================
//
// QUÉ HACE
//   Lee puntos 2D de un CSV (x,y) y le pone una etiqueta a cada punto:
//       0 = ruido
//       1 = core de primer orden   (tiene al menos min_samples puntos a distancia <= eps)
//       2 = core de segundo orden  (no es core, pero tiene un core a distancia <= eps)
//   El punto cuenta como vecino de sí mismo (igual que sklearn.DBSCAN).
//
// CÓMO SE COMPILA Y SE USA
//   g++ -O2 -fopenmp proy_ruido.cpp -o dbscan
//   ./dbscan <entrada.csv> <salida.csv> <eps> <min_samples> <version> <k> <hilos>
//
//   version : serial | p1 | p2
//   k       : número de regiones en que se divide la matriz (solo lo usa p2)
//   hilos   : número de hilos de OpenMP
//   Ejemplo : ./dbscan datos/4000_data.csv salida.csv 0.03 10 p2 16 2
//
//   El programa imprime solo el tiempo del algoritmo en segundos (sin leer ni escribir CSV).
//
// LAS TRES VERSIONES
//   serial : algoritmo ingenuo, compara cada punto contra todos los demás. Costo O(n^2).
//   p1     : lo mismo, pero el ciclo de afuera (el de cada punto i) se reparte entre hilos.
//   p2     : la matriz se divide en k regiones (una cuadrícula). Cada punto solo se compara
//            contra los puntos de su región y de las regiones vecinas (el "halo"). Cada hilo
//            se encarga de regiones completas.
//
// IDEA CLAVE PARA QUE LOS HILOS NO SE PISEN
//   Se trabaja en dos fases y se usa un arreglo 'core' separado de 'labels':
//     Fase 1: cada punto decide si es core y lo anota en core[i]. Solo escribe SU posición.
//     (Al terminar un "omp parallel for" hay una barrera: nadie pasa a la fase 2 hasta que
//      todos los hilos terminaron la fase 1.)
//     Fase 2: cada punto que no es core busca si tiene un core cerca. Solo LEE 'core'
//             y solo escribe SU etiqueta.
//   Así nunca hay dos hilos escribiendo la misma posición ni uno leyendo lo que otro escribe.
// =====================================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

// -------------------------------------------------------------------------------------
// Lectura y escritura de archivos
// -------------------------------------------------------------------------------------

// Lee un CSV con una línea "x,y" por punto (sin encabezado).
void leerCSV(const string& archivo, vector<double>& x, vector<double>& y) {
    ifstream f(archivo);
    if (!f.is_open()) {
        cerr << "No se pudo abrir el archivo de entrada: " << archivo << endl;
        exit(1);
    }
    string linea, valor;
    while (getline(f, linea)) {
        if (!linea.empty() && linea.back() == '\r') linea.pop_back();  // archivos de Windows
        if (linea.empty()) continue;                                    // ignorar líneas vacías
        stringstream ss(linea);
        getline(ss, valor, ',');
        x.push_back(stod(valor));
        getline(ss, valor, ',');
        y.push_back(stod(valor));
    }
}

// Escribe un CSV con una línea "x,y,etiqueta" por punto.
void escribirCSV(const string& archivo, const vector<double>& x, const vector<double>& y,
                 const vector<int>& labels) {
    ofstream f(archivo);
    if (!f.is_open()) {
        cerr << "No se pudo abrir el archivo de salida: " << archivo << endl;
        exit(1);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        f << x[i] << "," << y[i] << "," << labels[i] << "\n";
    }
}

// -------------------------------------------------------------------------------------
// Distancia entre dos puntos
// -------------------------------------------------------------------------------------

// Distancia euclidiana AL CUADRADO. Se compara contra eps*eps, así nos ahorramos la raíz cuadrada.
inline double dist2(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

// =====================================================================================
// VERSIÓN SERIAL
// =====================================================================================
void dbscan_serial(const vector<double>& x, const vector<double>& y, vector<int>& labels,
                   double eps, int min_samples) {
    int n = x.size();
    double eps2 = eps * eps;
    vector<char> core(n, 0);   // core[i] = 1 si el punto i es core de primer orden

    // Fase 1: ¿quién es core? Contamos cuántos puntos hay a distancia <= eps de cada punto.
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
        }
        core[i] = (vecinos >= min_samples);
    }

    // Fase 2: los que no son core son de segundo orden si tienen un core cerca; si no, son ruido.
    for (int i = 0; i < n; ++i) {
        if (core[i]) { labels[i] = 1; continue; }

        bool tiene_core_cerca = false;
        for (int j = 0; j < n; ++j) {
            if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                tiene_core_cerca = true;
                break;                      // con uno basta
            }
        }
        labels[i] = tiene_core_cerca ? 2 : 0;
    }
}

// =====================================================================================
// VERSIÓN P1: matriz completa, ciclo exterior en paralelo
// =====================================================================================
// Es igual que la serial, pero con "#pragma omp parallel for" en el ciclo de cada punto i.
// Cada iteración i solo escribe core[i] (o labels[i]), así que no hay condiciones de carrera.
// schedule(static): se reparten las iteraciones en bloques iguales, porque todas cuestan lo mismo.
void dbscan_p1(const vector<double>& x, const vector<double>& y, vector<int>& labels,
               double eps, int min_samples) {
    int n = x.size();
    double eps2 = eps * eps;
    vector<char> core(n, 0);

    // Fase 1 en paralelo
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int vecinos = 0;
        for (int j = 0; j < n; ++j) {
            if (dist2(x[i], y[i], x[j], y[j]) <= eps2) vecinos++;
        }
        core[i] = (vecinos >= min_samples);
    }
    // Aquí hay una barrera implícita: 'core' ya está completo antes de empezar la fase 2.

    // Fase 2 en paralelo (solo lee 'core')
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (core[i]) { labels[i] = 1; continue; }

        bool tiene_core_cerca = false;
        for (int j = 0; j < n; ++j) {
            if (core[j] && dist2(x[i], y[i], x[j], y[j]) <= eps2) {
                tiene_core_cerca = true;
                break;
            }
        }
        labels[i] = tiene_core_cerca ? 2 : 0;
    }
}

// =====================================================================================
// VERSIÓN P2: matriz dividida en k regiones (cuadrícula)
// =====================================================================================
//
// 1) La caja que contiene a todos los puntos se divide en una cuadrícula de 'filas x columnas'
//    regiones, con filas*columnas = k (lo más cuadrada posible; por ejemplo k=16 -> 4x4).
// 2) Los puntos se REORDENAN en memoria por región: todos los puntos de la región 0 juntos,
//    luego los de la región 1, etc. Así cada región es un bloque seguido de memoria
//    (la región c ocupa las posiciones inicio[c] ... inicio[c+1]-1).
// 3) Un punto solo puede tener vecinos en su región y en las regiones de alrededor (el halo).
//    Cuántas celdas de alrededor hay que mirar: ceil(eps / tamaño_de_celda). Con eso el
//    resultado es correcto para cualquier k (aunque eps sea más grande que una celda).
// 4) Cada hilo toma regiones completas (parallel for sobre las k regiones).

struct Cuadricula {
    int filas, columnas;       // filas * columnas = k
    int halo_f, halo_c;        // cuántas celdas hacia cada lado hay que revisar
    vector<int> inicio;        // inicio[c] = posición donde empieza la región c (tiene k+1 elementos)
    vector<int> orden;         // orden[p] = índice ORIGINAL del punto que quedó en la posición p
    vector<double> xs, ys;     // coordenadas de los puntos ya reordenados por región
};

// Construye la cuadrícula y reordena los puntos por región.
Cuadricula construir_cuadricula(const vector<double>& x, const vector<double>& y,
                                double eps, int k) {
    int n = x.size();
    Cuadricula g;

    // Filas y columnas lo más cuadradas posible: buscamos el divisor de k más cercano a sqrt(k).
    g.filas = 1;
    g.columnas = k;
    for (int f = (int)sqrt((double)k); f >= 1; --f) {
        if (k % f == 0) {
            g.filas = f;
            g.columnas = k / f;
            break;
        }
    }

    // Caja que contiene a todos los puntos
    double min_x = *min_element(x.begin(), x.end());
    double max_x = *max_element(x.begin(), x.end());
    double min_y = *min_element(y.begin(), y.end());
    double max_y = *max_element(y.begin(), y.end());

    // Tamaño de cada celda (el 0.0001 evita que el punto máximo caiga fuera de la cuadrícula)
    double ancho_celda = (max_x - min_x + 0.0001) / g.columnas;
    double alto_celda  = (max_y - min_y + 0.0001) / g.filas;

    g.halo_c = (int)ceil(eps / ancho_celda);
    g.halo_f = (int)ceil(eps / alto_celda);

    // Paso 1: ver en qué región cae cada punto y contar cuántos puntos tiene cada región.
    vector<int> region_de(n);
    g.inicio.assign(k + 1, 0);
    for (int i = 0; i < n; ++i) {
        int col = min((int)((x[i] - min_x) / ancho_celda), g.columnas - 1);
        int fil = min((int)((y[i] - min_y) / alto_celda), g.filas - 1);
        region_de[i] = fil * g.columnas + col;
        g.inicio[region_de[i] + 1]++;
    }

    // Paso 2: sumas acumuladas, así inicio[c] queda como la posición donde empieza la región c.
    for (int c = 0; c < k; ++c) g.inicio[c + 1] += g.inicio[c];

    // Paso 3: colocar cada punto dentro del bloque de su región.
    vector<int> siguiente(g.inicio.begin(), g.inicio.end() - 1);  // próxima posición libre de cada región
    g.orden.resize(n);
    g.xs.resize(n);
    g.ys.resize(n);
    for (int i = 0; i < n; ++i) {
        int p = siguiente[region_de[i]]++;
        g.orden[p] = i;
        g.xs[p] = x[i];
        g.ys[p] = y[i];
    }
    return g;
}

// Para una región dice qué posiciones hay que revisar (ella misma y su halo).
// Las celdas de una misma fila de la cuadrícula quedan una tras otra en memoria, así que
// de cada fila de celdas basta un solo rango [desde, hasta).
void rangos_a_revisar(const Cuadricula& g, int region, vector<pair<int,int>>& rangos) {
    rangos.clear();
    int fila_region = region / g.columnas;
    int col_region  = region % g.columnas;

    int col_ini = max(0, col_region - g.halo_c);
    int col_fin = min(g.columnas - 1, col_region + g.halo_c);

    for (int df = -g.halo_f; df <= g.halo_f; ++df) {
        int fila = fila_region + df;
        if (fila < 0 || fila >= g.filas) continue;           // fuera de la cuadrícula
        int desde = g.inicio[fila * g.columnas + col_ini];
        int hasta = g.inicio[fila * g.columnas + col_fin + 1];
        rangos.push_back({desde, hasta});
    }
}

void dbscan_p2(const vector<double>& x, const vector<double>& y, vector<int>& labels,
               double eps, int min_samples, int k) {
    int n = x.size();
    double eps2 = eps * eps;

    Cuadricula g = construir_cuadricula(x, y, eps, k);
    vector<char> core(n, 0);    // OJO: está indexado por la posición NUEVA (ya reordenada)
    vector<int> etiqueta(n, 0); // etiqueta de cada punto, también por posición nueva

    // Fase 1: cada hilo toma regiones completas.
    // schedule(dynamic, 1): cuando un hilo termina una región, toma la siguiente que quede.
    // Se usa dynamic porque las regiones tienen muy distinta cantidad de puntos.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int region = 0; region < k; ++region) {
        vector<pair<int,int>> rangos;
        rangos_a_revisar(g, region, rangos);

        for (int p = g.inicio[region]; p < g.inicio[region + 1]; ++p) {
            int vecinos = 0;
            for (auto& r : rangos) {
                for (int q = r.first; q < r.second; ++q) {
                    if (dist2(g.xs[p], g.ys[p], g.xs[q], g.ys[q]) <= eps2) vecinos++;
                }
            }
            core[p] = (vecinos >= min_samples);   // solo escribe la posición p de SU región
        }
    }
    // Barrera implícita: aquí "se juntan" los resultados de todas las regiones. Los cores de una
    // región que le hacen falta a la región de al lado ya están listos antes de la fase 2.

    // Fase 2: cores de segundo orden y ruido, con el mismo halo.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int region = 0; region < k; ++region) {
        vector<pair<int,int>> rangos;
        rangos_a_revisar(g, region, rangos);

        for (int p = g.inicio[region]; p < g.inicio[region + 1]; ++p) {
            if (core[p]) { etiqueta[p] = 1; continue; }

            bool tiene_core_cerca = false;
            for (auto& r : rangos) {
                for (int q = r.first; q < r.second; ++q) {
                    if (core[q] && dist2(g.xs[p], g.ys[p], g.xs[q], g.ys[q]) <= eps2) {
                        tiene_core_cerca = true;
                        break;
                    }
                }
                if (tiene_core_cerca) break;
            }
            etiqueta[p] = tiene_core_cerca ? 2 : 0;
        }
    }

    // Regresar las etiquetas al orden original de los puntos.
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n; ++p) labels[g.orden[p]] = etiqueta[p];
}

// =====================================================================================
// PROGRAMA PRINCIPAL
// =====================================================================================
int main(int argc, char* argv[]) {
    if (argc < 8) {
        cerr << "Uso: " << argv[0]
             << " <entrada.csv> <salida.csv> <eps> <min_samples> <serial|p1|p2> <k> <hilos>" << endl;
        return 1;
    }

    // Leer los argumentos de la línea de comandos
    string archivo_entrada = argv[1];
    string archivo_salida  = argv[2];
    double eps             = stod(argv[3]);
    int min_samples        = stoi(argv[4]);
    string version         = argv[5];
    int k                  = stoi(argv[6]);
    int hilos              = stoi(argv[7]);

    if (k < 1 || hilos < 1) {
        cerr << "k y hilos deben ser al menos 1" << endl;
        return 1;
    }
    omp_set_num_threads(hilos);

    // Leer los puntos
    vector<double> x, y;
    leerCSV(archivo_entrada, x, y);
    vector<int> labels(x.size(), 0);

    // Medir solo el algoritmo (sin leer ni escribir archivos)
    double t_inicio = omp_get_wtime();

    if (version == "serial") {
        dbscan_serial(x, y, labels, eps, min_samples);
    } else if (version == "p1") {
        dbscan_p1(x, y, labels, eps, min_samples);
    } else if (version == "p2") {
        dbscan_p2(x, y, labels, eps, min_samples, k);
    } else {
        cerr << "Versión no reconocida. Use serial, p1 o p2." << endl;
        return 1;
    }

    double t_fin = omp_get_wtime();

    // Se imprime SOLO el tiempo (en segundos), para que los scripts lo puedan leer fácil.
    cout << t_fin - t_inicio << endl;

    escribirCSV(archivo_salida, x, y, labels);
    return 0;
}
