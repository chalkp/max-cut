from typing import Union, Tuple, List

import networkx as nx


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


def process_max_cut_weighted(G: nx.Graph) -> Tuple[float, List[Tuple[int, int]]]:
    # Same as process_max_cut, but with **weight**
    max_cut_value = 0.0
    max_cut_edges = list()

    for (u, v, data) in G.edges(data=True):
        color_u = G.nodes[u].get('color', 0)
        color_v = G.nodes[v].get('color', 0)

        if color_u != color_v:
            max_cut_value += data.get('weight', 1.0)
            max_cut_edges.append((u, v))

    return (max_cut_value, max_cut_edges)
