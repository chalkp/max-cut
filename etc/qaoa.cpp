#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>

// Configs
const int layer_count = 5;
const int max_eval = 1000;
const int seed = 42;

using namespace std::complex_literals;
using complex = std::complex<cudaq::complex>;

struct edge {
    size_t u, v;
};


// Utils

cudaq::spin_op get_hamiltonian(std::vector<struct edge> edge_list) {
    cudaq::spin_op ret_hamiltonian;
    for (auto &[u, v]: edge_list) {
        ret_hamiltonian += cudaq::spin::z(u) * cudaq::spin::z(v);
        ret_hamiltonian -= cudaq::spin::i(u) * cudaq::spin::i(v);
    }
    return ret_hamiltonian / 2;
}

double rand_doube() {
    double normalized = ((double)rand() / RAND_MAX);
    return normalized * 2.0 * M_PI - M_PI;
}


// Kernels

__qpu__ void uc(cudaq::qubit &u, cudaq::qubit &v, double alpha) {
    cudaq::cx(u, v);
    cudaq::rz(2.0 * alpha, v);
    cudaq::cx(u, v);
}

__qpu__ void um(cudaq::qubit &qubit, double beta) {
    cudaq::rx(2.0 * beta, qubit);
}

__qpu__ void qaoa_kernel(
    size_t qubit_count,
    size_t layer_count,
    std::vector<size_t> sources,
    std::vector<size_t> targets,
    std::vector<double> alphas,
    std::vector<double> betas
) {
    cudaq::qvector qvector(qubit_count);
    cudaq::h(qvector);

    for (size_t i = 0; i < layer_count; i++) {
        for (size_t k = 0; k < sources.size(); k++) {
            uc(qvector[sources[k]], qvector[targets[k]], alphas[i]);
        }
        for (int j = 0; j < qubit_count; j++) {
            um(qvector[j], betas[i]);
        }
    }
}


int main() {
    size_t n, m;
    size_t shots = 100000;

    std::vector<struct edge> edge_list;
    std::vector<size_t> sources;
    std::vector<size_t> targets;
    
    std::vector<double> losses;
    std::vector<std::string> bests;
    std::vector<cudaq::sample_result> results;
    
    srand(seed);

    if (scanf("%lu %lu", &n, &m) != 2) return 1;

    edge_list.reserve(m);
    sources.reserve(m);
    targets.reserve(m);

    for (size_t i = 0; i < m; i++) {
        size_t u, v;
        if (scanf("%lu %lu", &u, &v) != 2) return 1;
        edge_list.push_back((struct edge){u, v});
    }


    cudaq::spin_op hamil = get_hamiltonian(edge_list);

    std::vector<double> init_params(2 * layer_count);
    for (auto &param: init_params) {
        param = rand_doube();
    }
    
    cudaq::optimizers::cobyla optimizer; 
    optimizer.max_eval = max_eval;
    optimizer.initial_parameters = init_params;
    
    
    auto [min_energy, opt_params] = optimizer.optimize(
        2 * layer_count,
        [&](const std::vector<double>& params) {
            std::vector<double> alphas(layer_count);
            std::vector<double> betas(layer_count);

            for (size_t i = 0; i < layer_count; i++) {
                alphas[i] = params[i];
                betas[i] = params[i + layer_count];
            }

            double energy = cudaq::observe(
                shots, qaoa_kernel,
                hamil,
                n, layer_count, sources, targets, alphas, betas
            ).expectation();

            losses.push_back(energy);

            return energy;
        }
    );


    if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
        printf("Optimization Steps: %lu\n", losses.size());
        printf("Loss Landscape (Energy per step):\n");
        for (size_t i = 0; i < losses.size(); i++) {
            printf("[%lu]: %f\n", i, losses[i]);
        }

        printf("\nMinimum Energy: %f\n", min_energy);

        std::vector<double> opt_alphas(layer_count), opt_betas(layer_count);
        for (size_t i = 0; i < layer_count; i++) {
            opt_alphas[i] = opt_params[i];
            opt_betas[i] = opt_params[i + layer_count];
        }

        auto counts = cudaq::sample(
            shots, qaoa_kernel,
            n, layer_count, sources, targets, opt_alphas, opt_betas
        );

        std::string best;
        size_t max_count = 0;
        size_t total_shots = 0;

        for (auto &[bits, count]: counts) {
            total_shots += count;
            if (count > max_count) {
                max_count = count;
                best = bits;
            }
        }

        printf("Best Solution: %s, (prob: %lf)\n", best.c_str(), (double)(max_count / total_shots));
    }

    return 0;
}
