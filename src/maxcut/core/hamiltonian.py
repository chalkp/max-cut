from cudaq import spin

def max_cut_hamiltonian(sources, targets, weights=None):
    hamil = 0
    if weights == None:
        weights = [1.0] * len(sources)

    for i in range(len(sources)):
        qu = sources[i]
        qv = targets[i]
        w = weights[i]
        hamil += w * spin.z(qu) * spin.z(qv)
    
    return hamil
