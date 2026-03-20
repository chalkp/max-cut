import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from typing import Union, Tuple, List
from itertools import chain, combinations


def generate_graph_legacy(
        n: int, p: float,
        rng: np.random._generator.Generator = None,
        seed: int = 42) -> nx.Graph:
    if rng == None:
        rng = np.random.default_rng(seed)
    while True:
        G = nx.gnp_random_graph(n, p, seed=rng)
        if nx.is_connected(G):
            nx.set_node_attributes(G, values=0, name='color')
            nx.set_edge_attributes(G, values=1.0, name='weight')
            return G


def generate_graph(n: int, m: int, seed: int = 42) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=rng)
    nx.set_node_attributes(G, values=0, name='color')
    nx.set_edge_attributes(G, values=1.0, name='weight')
    return G


def color_graph(
        G: nx.Graph,
        nodes: Union[int, Tuple],
        color: int) -> nx.Graph:
    if type(nodes) == int:
        G.nodes[nodes]['color'] = color
        return G
    attrs = {node: color for node in nodes}
    nx.set_node_attributes(G, attrs, 'color')
    return G


def mark_edges(G: nx.Graph, edges: List[Tuple[int, int]]) -> nx.Graph:
    nx.set_edge_attributes(G, False, "is_cut")
    
    for u, v in edges:
        if G.has_edge(u, v):
            G[u][v]["is_cut"] = True
    
    return G


def draw_init_graph(G: nx.Graph, seed: int = 42) -> None:
    color_map = ['#8c8c8c'] * G.number_of_nodes()
    edge_color_map = ['#acacac'] * G.number_of_edges()
    
    nx.draw(
        G,
        with_labels=True,
        pos=nx.spring_layout(G, seed=seed),
        node_color=color_map,
        edge_color=edge_color_map,
    )
    plt.show()


def draw_graph(
        G: nx.Graph,
        color_0: str = '#8c8c8c',
        color_1: str = '#76b900',
        cut_edge_color: str = '#8b2332',
        normal_edge_color: str = '#acacac',
        seed: int = 42) -> None:
    
    color_map = [
        color_0 if G.nodes[u].get('color', 0) == 0
        else color_1 for u in G]
    
    edge_color_map = list()
    edge_width_map = list()
    
    for u, v in G.edges():
        if G[u][v].get('is_cut', False):
            edge_color_map.append(cut_edge_color)
            edge_width_map.append(1.35)
        else:
            edge_color_map.append(normal_edge_color)
            edge_width_map.append(1.0)
    
    nx.draw(
        G,
        with_labels=True,
        pos=nx.spring_layout(G, seed=seed),
        node_color=color_map,
        edge_color=edge_color_map,
        width=edge_width_map
    )
    plt.show()


def powerset(G: nx.Graph) -> chain:
    s = list(G.nodes())
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def brute_force(
        G: nx.Graph)-> Tuple[int, List[Tuple[int, int]], Tuple[int, ...]]:
    max_cut_value = 0
    max_cut_edges = list()
    max_subset = None

    G_powerset = powerset(G)
    for subset in G_powerset:
        subset_cut_value = 0
        subset_cut_edges = list()

        subset_set = set(subset)

        for (u, v) in G.edges():
            if (u in subset_set) != (v in subset_set):
                subset_cut_value += 1
                subset_cut_edges.append((u, v))

        if subset_cut_value > max_cut_value:
            max_cut_value = subset_cut_value
            max_cut_edges = subset_cut_edges
            max_subset = subset

    return (max_cut_value, max_cut_edges, max_subset)


def process_max_cut(G: nx.Graph) -> Tuple[int, List[int]]:
    max_cut_value = 0
    max_cut_edges = list()

    for (u, v) in G.edges():
        color_u = G.nodes[u].get('color', 0)
        color_v = G.nodes[v].get('color', 0)
        
        if color_u != color_v:
            max_cut_value += 1
            max_cut_edges.append((u, v))

    return (max_cut_value, max_cut_edges)
