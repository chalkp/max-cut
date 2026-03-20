import networkx as nx
import cudaq
import pandas as pd
import numpy as np
import threading

from utils import (
    generate_graph,
    process_max_cut,
    brute_force
)
from rqaoa import solve_rqaoa
from tqdm.auto import tqdm

cudaq.set_target("nvidia", option='mgpu')

shots = 2000
maxiter = 300
data = list()

for n in range(3, 21):
    print("=" * 15 + f"{n}" + "=" * 15)
    threads = list()
    rqaoa_results = dict()
    brute_force_results = dict()
    
    def run_brute_force(G, seed):
        ground_truth_max_cut_value, _, _ = brute_force(G)
        brute_force_results[seed] = ground_truth_max_cut_value
    

    for seed in tqdm(range(50)):
        G_for_bf = generate_graph(n, (n + 1) // 2, seed)
        thread = threading.Thread(target=run_brute_force, args=(G_for_bf, seed))
        threads.append(thread)
        thread.start()
        
        for layer_count in [1, 2, 3]:
            cudaq.set_random_seed(seed)
            np.random.seed(seed)

            G = generate_graph(n, (n + 1) // 2, seed)

            final_solution, _, losses = solve_rqaoa(G, layer_count, shots=shots, seed=seed, method='COBYLA', maxiter=maxiter)

            for node, color in final_solution.items():
                if color == 1:
                    nx.nodes(G)[node]['color'] = 1
                else:
                    nx.nodes(G)[node]['color'] = 0

            max_cut_value, _ = process_max_cut(G)
            rqaoa_results[(seed, layer_count)] = (max_cut_value, losses)

        t.join()

        ground_truth_max_cut_value = brute_force_results[seed]
        for layer_count in [1, 2, 3]:
            max_cut_value, losses = rqaoa_results[(seed, layer_count)]
            data.append((n, seed, ground_truth_max_cut_value, layer_count, max_cut_value, losses))
            # print(f"run #{seed} (p={layer_count}): {max_cut_value} / {ground_truth_max_cut_value}")
            # if ground_truth_max_cut_value > 0:
            #     print(f"performance = {max_cut_value / ground_truth_max_cut_value:.4f}")
            # else:
            #     print(f"performance = N/A")

pd.DataFrame(data, columns=['n_nodes', 'seed', 'ground_truth', 'layer_count', 'rqaoa', 'losses']).to_csv('experiment_layer_count.csv', index=False)
