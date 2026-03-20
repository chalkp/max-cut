import networkx as nx
import cudaq
import pandas as pd
import numpy as np
import threading
from tqdm.auto import tqdm

from utils import (
    generate_graph,
    process_max_cut,
    brute_force
)
from rqaoa import solve_rqaoa
cudaq.set_target("nvidia", option='mgpu')

layer_count = 1
shots = 3000
maxiter = 100
data = list()

for n in range(3, 21):
    print("=" * 15 + f"{n}" + "=" * 15)
    threads = list()
    rqaoa_results = dict()
    brute_force_results = dict()
    
    def run_brute_force(G, seed):
        ground_truth_max_cut_value, _, _ = brute_force(G)
        brute_force_results[seed] = ground_truth_max_cut_value
    
    for seed in tqdm(range(100)):
        cudaq.set_random_seed(seed)
        np.random.seed(seed)

        G = generate_graph(n, (n + 1) // 2, seed)

        thread = threading.Thread(target = run_brute_force, args=(G, seed))
        threads.append(thread)
        thread.start()

        final_solution, _, losses = solve_rqaoa(G, layer_count, shots=shots, seed=seed, method='COBYLA', maxiter=maxiter)

        for node, color in final_solution.items():
            if color == 1:
                nx.nodes(G)[node]['color'] = 1
            else:
                nx.nodes(G)[node]['color'] = 0

        max_cut_value, _ = process_max_cut(G)
        rqaoa_results[seed] = (max_cut_value, losses)
        t.join()

        ground_truth_max_cut_value = brute_force_results[seed]
        max_cut_value, losses = rqaoa_results[seed]
        data.append((n, seed, ground_truth_max_cut_value, max_cut_value, losses))
        # print(f"run #{seed}: {max_cut_value} / {ground_truth_max_cut_value}, performance = {max_cut_value / ground_truth_max_cut_value:.4f}")

pd.DataFrame(data, columns=['n_nodes', 'seed', 'ground_truth', 'rqaoa', 'losses']).to_csv('experiment.csv', index=False)
