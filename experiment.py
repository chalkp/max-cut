import networkx as nx
import matplotlib.pyplot as plt
import cudaq
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from utils import (
    generate_graph,
    process_max_cut,
    brute_force
)
from rqaoa import solve_rqaoa
cudaq.set_target("nvidia", option='mgpu')

data = list()

for n in tqdm(range(3, 21)):
    for seed in range(10):
        cudaq.set_random_seed(seed)
        np.random.seed(seed)
        p = 1/2

        layer_count = 1
        shots = 2000
        maxiter = 100
        G = generate_graph(n, p)

        ground_truth_max_cut_value, _, _ = brute_force(G)

        final_solution, _, losses = solve_rqaoa(G, layer_count, shots=shots, seed=seed, method='COBYLA', maxiter=maxiter)

        for node, color in final_solution.items():
            if color == 1:
                nx.nodes(G)[node]['color'] = 1
            else:
                nx.nodes(G)[node]['color'] = 0


        max_cut_value, _ = process_max_cut(G)
        
        data.append((n, seed, ground_truth_max_cut_value, max_cut_value, losses))
        print("=" * 30)
        print(f"n = {n}, seed = {seed}")
        print(f"ground_truth = {ground_truth_max_cut_value}, ")
        print(f"rqaoa = {max_cut_value}")

pd.DataFrame(data, columns=['n_nodes', 'seed', 'ground_truth', 'rqaoa', 'losses']).to_csv('experiment.csv', index=False)
