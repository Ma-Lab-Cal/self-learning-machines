import numpy as np

class Component:
    def __init__(self, name, l, r, val):
        self.name = name
        self.l, self.r = l, r
        self.val = val

    def __repr__(self):
        return f"{self.name} {self.l} {self.r} {self.val}"

def gen_netlist(mat, fname):
    assert mat.shape[0] == mat.shape[1], "circuit matrix should be square"
    n = mat.shape[0] # number of nodes
    res_mask = np.maximum(mat, mat.T) < np.inf
    di_mask = np.minimum(mat, mat.T) < np.inf
    zero_mask = np.maximum(mat, mat.T) > 0
    components = []

    for i in np.arange(n):
        for j in np.arange(i):
            if res_mask[i,j] and zero_mask[i,j]:
                components.append(Component(f"R{n*i+j}", i, j, mat[i, j]))
            elif di_mask[i, j] and zero_mask[i,j]:
                if mat[i, j] < np.inf:
                    components.append(Component(f"D{n*i+j}", i, j, "D"))
                else:
                    components.append(Component(f"D{n*i+j}", j, i, "D"))
    with open(fname, "wt") as f:
        f.write("*\n")
        for comp in components:
            f.write(f"{comp}\n")
        
        f.write("V1 1 0 5\n")
        f.write(".model D D\n")
        f.write(".lib /Users/lancemathias/Library/Application Support/LTspice/lib/cmp/standard.dio\n")
        f.write(".tran 1\n.backanno\n.end\n")