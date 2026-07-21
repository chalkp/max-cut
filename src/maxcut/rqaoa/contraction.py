import networkx as nx


def contract_graph(graph: nx.Graph, u, v, correlation_sign: float):
    G_new = graph.copy()

    for edge in G_new.edges():
        if 'weight' not in G_new.edges[edge]:
            G_new.edges[edge]['weight'] = 1.0

    for neighbor in list(G_new.neighbors(v)):
        if neighbor == u:
            continue

        weight_v_neighbor = G_new.edges[v, neighbor]['weight']

        effective_weight = correlation_sign * weight_v_neighbor

        if G_new.has_edge(u, neighbor):
            G_new.edges[u, neighbor]['weight'] += effective_weight
        else:
            G_new.add_edge(u, neighbor, weight=effective_weight)

    G_new.remove_node(v)
    return G_new


def reconstruct_solution(base_assignment: dict, elimination_history: list):
    solution = base_assignment.copy()

    for u, v, _, sign in reversed(elimination_history):
        solution[v] = int(sign * solution[u])

    return solution
