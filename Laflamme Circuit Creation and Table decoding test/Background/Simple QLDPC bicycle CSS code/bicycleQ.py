import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from qiskit.quantum_info import Pauli, PauliList  # Qiskit symplectic Pauli support


# --------------------------
# GF(2) helpers
# --------------------------
def gf2_matmul(A, B):
    """Matrix product over GF(2)."""
    return (A @ B) % 2


# --------------------------
# Bicycle CSS construction
#   HX = [C | C^T]
#   HZ = [C^T | C]
# This guarantees HX HZ^T = 0 (CSS commutation).
# --------------------------
def circulant_from_first_row(first_row):
    first_row = np.array(first_row, dtype=np.uint8) % 2
    r = len(first_row)
    C = np.zeros((r, r), dtype=np.uint8)
    for i in range(r):
        C[i] = np.roll(first_row, i)
    return C

def random_sparse_first_row(r, weight, seed=None):
    if seed is not None:
        random.seed(seed)
    idx = random.sample(range(r), k=weight)
    v = np.zeros(r, dtype=np.uint8)
    v[idx] = 1
    return v

def generate_bicycle_css(r=18, row_weight=3, seed=7):
    """
    Returns (HX, HZ) with shapes (r, 2r) each.
    n = 2r physical qubits.
    """
    random.seed(seed)
    np.random.seed(seed)

    first = random_sparse_first_row(r, row_weight)
    C = circulant_from_first_row(first)
    CT = C.T.copy()

    HX = np.concatenate([C, CT], axis=1).astype(np.uint8)
    HZ = np.concatenate([CT, C], axis=1).astype(np.uint8)

    # Verify CSS commutation
    comm = gf2_matmul(HX, HZ.T)
    ok = np.all(comm == 0)

    return HX, HZ, ok


# --------------------------
# Convert (HX, HZ) rows to Qiskit Paulis
# For CSS:
#   X-stabilizer row h -> Pauli with x=h, z=0
#   Z-stabilizer row h -> Pauli with x=0, z=h
# --------------------------
def hx_rows_to_paulis(HX):
    HX = np.array(HX, dtype=bool)
    m, n = HX.shape
    out = []
    for i in range(m):
        x = HX[i]
        z = np.zeros(n, dtype=bool)
        out.append(Pauli((z, x)))  # Pauli stores (z, x)
    return PauliList(out)

def hz_rows_to_paulis(HZ):
    HZ = np.array(HZ, dtype=bool)
    m, n = HZ.shape
    out = []
    for i in range(m):
        z = HZ[i]
        x = np.zeros(n, dtype=bool)
        out.append(Pauli((z, x)))
    return PauliList(out)


# --------------------------
# Tanner graphs
# --------------------------
def build_tanner_graph(H, check_prefix="c", var_prefix="q"):
    H = np.array(H, dtype=np.uint8) % 2
    m, n = H.shape

    G = nx.Graph()
    checks = [f"{check_prefix}{i}" for i in range(m)]
    vars_  = [f"{var_prefix}{j}" for j in range(n)]
    G.add_nodes_from(checks, bipartite=0)
    G.add_nodes_from(vars_, bipartite=1)

    for i in range(m):
        js = np.where(H[i] == 1)[0]
        for j in js:
            G.add_edge(checks[i], vars_[j])

    return G, checks, vars_

def plot_bipartite(G, checks, vars_, title="", max_labels=60):
    # Simple left/right layout
    pos = {}
    for i, c in enumerate(checks):
        pos[c] = (0, -i)
    for j, v in enumerate(vars_):
        pos[v] = (1, -j)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=220)
    if len(G.nodes) <= max_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.show()


# --------------------------
# Extra relevant graphs
# --------------------------
def build_commutation_graph(HX, HZ):
    """
    Bipartite graph: X-check nodes vs Z-check nodes.
    Edge if anticommute (overlap parity = 1).
    For a valid CSS code, this graph should have 0 edges.
    """
    overlap = gf2_matmul(HX, HZ.T)  # mX x mZ
    mX, mZ = overlap.shape

    G = nx.Graph()
    Xnodes = [f"X{i}" for i in range(mX)]
    Znodes = [f"Z{j}" for j in range(mZ)]
    G.add_nodes_from(Xnodes, bipartite=0)
    G.add_nodes_from(Znodes, bipartite=1)

    for i in range(mX):
        bad = np.where(overlap[i] == 1)[0]
        for j in bad:
            G.add_edge(Xnodes[i], Znodes[j])
    return G

def plot_degree_hist(H, title=""):
    H = np.array(H, dtype=np.uint8) % 2
    col_deg = H.sum(axis=0)
    row_deg = H.sum(axis=1)

    plt.figure(figsize=(10, 4))
    plt.hist(col_deg, bins=np.arange(col_deg.max() + 2) - 0.5)
    plt.title(f"{title} variable-node (qubit) degrees")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(row_deg, bins=np.arange(row_deg.max() + 2) - 0.5)
    plt.title(f"{title} check-node degrees")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()


# --------------------------
# Demo
# --------------------------
if __name__ == "__main__":
    HX, HZ, ok = generate_bicycle_css(r=18, row_weight=3, seed=7)
    n = HX.shape[1]
    print("n qubits:", n)
    print("HX shape:", HX.shape, "HZ shape:", HZ.shape)
    print("CSS commutation HX*HZ^T == 0 ?", ok)

    # Qiskit PauliLists (stabilizer generators)
    Xstabs = hx_rows_to_paulis(HX)
    Zstabs = hz_rows_to_paulis(HZ)

    # Print a couple stabilizers as labels
    print("Example X-stabilizers:", [p.to_label() for p in Xstabs[:3]])
    print("Example Z-stabilizers:", [p.to_label() for p in Zstabs[:3]])

    # Tanner graphs
    GX, cx, qx = build_tanner_graph(HX, check_prefix="x", var_prefix="q")
    GZ, cz, qz = build_tanner_graph(HZ, check_prefix="z", var_prefix="q")
    plot_bipartite(GX, cx, qx, title="Tanner graph for H_X (X-checks vs qubits)")
    plot_bipartite(GZ, cz, qz, title="Tanner graph for H_Z (Z-checks vs qubits)")

    # Commutation (anticommutation) graph
    Gcomm = build_commutation_graph(HX, HZ)
    plt.figure(figsize=(10, 4))
    plt.title(f"X–Z commutation graph (edges = anticommute). Edges: {Gcomm.number_of_edges()}")
    nx.draw(Gcomm, with_labels=True, node_size=300)
    plt.axis("off")
    plt.show()

    # Degree histograms (LDPC-ness)
    plot_degree_hist(HX, title="H_X")
    plot_degree_hist(HZ, title="H_Z")
