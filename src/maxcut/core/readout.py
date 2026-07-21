def bitstring_to_coloring(bitstring, nodes, node_map):
    return {
        node: (1 if bitstring[node_map[node]] == '1' else 0)
        for node in nodes
    }
