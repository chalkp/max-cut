"""
WARNING: HEADLESS-UNSAFE.
"""
import networkx as nx
from matplotlib import pyplot as plt


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
