#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <random>
#include <thread>
#include <future>

using namespace std;

class AlgoritmoGenetico {
private:
    vector<vector<int>> cromosomas;
    const vector<int> coeficientes;
    const int valor_objetivo;
    const double tasa_crossover;
    const double tasa_mutacion;
    const int tam_poblacion;
    const int max_generaciones;
    const int num_coeficientes;

    mt19937 gen; //alternativa de generacion de randoms
    uniform_real_distribution<> dis;

    double calcularFitness(int i) {
        double objetivo = valor_objetivo - inner_product(cromosomas[i].begin(), cromosomas[i].end(), coeficientes.begin(), 0);
        return 1.0 / (1.0 + fabs(objetivo));
    }

    int seleccionRuleta(const vector<double>& prob_cum) {
        double r = dis(gen);
        return lower_bound(prob_cum.begin(), prob_cum.end(), r) - prob_cum.begin() - 1;
    }

    vector<vector<int>> inicializarPoblacion() {
        vector<vector<int>> poblacion(tam_poblacion, vector<int>(num_coeficientes));
        for (auto& cromosoma : poblacion) {
            for (auto& gen : cromosoma) {
                gen = rand() % (valor_objetivo + 1);
            }
        }
        return poblacion;
    }

    void crossover(vector<vector<int>>& nuevos_cromosomas, const vector<int>& cruzar_indices) {
        for (size_t i = 0; i < cruzar_indices.size(); i += 2) {
            if (i + 1 < cruzar_indices.size()) {
                int punto_crossover = rand() % num_coeficientes + 1;
                for (int j = punto_crossover; j < num_coeficientes; ++j) {
                    swap(nuevos_cromosomas[cruzar_indices[i]][j], nuevos_cromosomas[cruzar_indices[i + 1]][j]);
                }
            }
        }
    }

    void mutacion(vector<vector<int>>& nuevos_cromosomas, int num_mutaciones) {
        for (int i = 0; i < num_mutaciones; ++i) {
            int fila = rand() % tam_poblacion;
            int columna = rand() % num_coeficientes;
            nuevos_cromosomas[fila][columna] = rand() % (valor_objetivo + 1);
        }
    }

public:
    AlgoritmoGenetico(const vector<int>& coeficientes, const int valor_objetivo, const double tasa_crossover, const double tasa_mutacion, const int tam_poblacion, const int max_generaciones)
        : coeficientes(coeficientes), valor_objetivo(valor_objetivo), tasa_crossover(tasa_crossover), tasa_mutacion(tasa_mutacion), tam_poblacion(tam_poblacion), max_generaciones(max_generaciones), num_coeficientes(coeficientes.size()), gen(random_device{}()), dis(0, 1) {
        cromosomas = inicializarPoblacion();
    }

    void ejecutar() {
        for (int generacion = 0; generacion < max_generaciones; ++generacion) {
            vector<double> fitness(tam_poblacion);
            double suma_fitness = 0;
            for (int i = 0; i < tam_poblacion; ++i) {
                fitness[i] = calcularFitness(i);
                suma_fitness += fitness[i];
            }

            vector<double> prob_cum(tam_poblacion + 1);
            for (int i = 0; i < tam_poblacion; ++i) {
                prob_cum[i + 1] = prob_cum[i] + (fitness[i] / suma_fitness);
            }

            vector<vector<int>> nuevos_cromosomas(tam_poblacion, vector<int>(num_coeficientes));
            for (int i = 0; i < tam_poblacion; ++i) {
                nuevos_cromosomas[i] = cromosomas[seleccionRuleta(prob_cum)];
            }

            vector<int> cruzar_indices;
            for (int i = 0; i < tam_poblacion; ++i) {
                if (dis(gen) < tasa_crossover) {
                    cruzar_indices.push_back(i);
                }
            }

            crossover(nuevos_cromosomas, cruzar_indices);

            int total_genes = tam_poblacion * num_coeficientes;
            int num_mutaciones = static_cast<int>(round(tasa_mutacion * total_genes));

            mutacion(nuevos_cromosomas, num_mutaciones);

            cromosomas = move(nuevos_cromosomas);
        }

        mostrarResultados();
    }

    void mostrarResultados() {
        cout << "Cromosomas finales después de " << max_generaciones << " generaciones:" << endl;
        for (const auto& cromosoma : cromosomas) {
            int resultado = 0;
            for (size_t i = 0; i < cromosoma.size(); ++i) {
                resultado += cromosoma[i] * coeficientes[i];
                cout << cromosoma[i] << " ";
            }
            cout << "Resultado = " << resultado << endl;
        }
    }

    void mostrarPoblacion() {
        cout << "Cromosomas:" << endl;
        for (const auto& cromosoma : cromosomas) {
            for (const auto& gen : cromosoma) {
                cout << gen << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    const vector<int> coeficientes = {1, 2, 3, 4};
    const int valor_objetivo = 30;
    const double crossover = 0.25;
    const double mutacion = 0.1;
    const int poblacion = 6;
    const int generaciones = 1000000;

    AlgoritmoGenetico ag(coeficientes, valor_objetivo, crossover, mutacion, poblacion, generaciones);

    cout << "----Población Inicial----" << endl;
    ag.mostrarPoblacion();
    ag.ejecutar();
    delete (coeficientes);
    return 0;
}
