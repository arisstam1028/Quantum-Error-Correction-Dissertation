import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


def gf2_matmul(A, B):
    return (A @ B) % 2


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


def generate_commuting_css_bicycle(r=18, a_row_weight=3, seed=7):
    """
    Guaranteed-commuting CSS construction:
      HX  [I | A]
      HZ  [A^T | I]
    where A is sparse (here: circulant from a sparse first row).
    """
    random.seed(seed)
    np.random.seed(seed)

    I = np.eye(r, dtype=np.uint8)

    first = random_sparse_first_row(r, a_row_weight, seed=seed)
    A = circulant_from_first_row(first)  # sparse-ish if first row is sparse

    HX = np.concatenate([I, A], axis=1).astype(np.uint8)      # r x 2r
    HZ = np.concatenate([A.T, I], axis=1).astype(np.uint8)    # r x 2r

    comm = gf2_matmul(HX, HZ.T)
    ok = np.all(comm == 0)
    return HX, HZ, ok, A


def build_combined_css_tanner_graph(HX, HZ):
    """
    One Tanner graph:
      Left: X-check nodes and Z-check nodes
      Right: qubit nodes
    Edges from HX and HZ both included.
    """
    HX = np.array(HX, dtype=np.uint8) % 2
    HZ = np.array(HZ, dtype=np.uint8) % 2
    mx, n = HX.shape
    mz, n2 = HZ.shape
    assert n == n2

    G = nx.Graph()

    x_checks = [f"x{i}" for i in range(mx)]
    z_checks = [f"z{i}" for i in range(mz)]
    qubits   = [f"q{j}" for j in range(n)]

    # Add nodes
    G.add_nodes_from(x_checks, bipartite=0, kind="X")
    G.add_nodes_from(z_checks, bipartite=0, kind="Z")
    G.add_nodes_from(qubits,   bipartite=1, kind="Q")

    # Add edges from HX
    for i in range(mx):
        for j in np.where(HX[i] == 1)[0]:
            G.add_edge(x_checks[i], qubits[j], etype="X")

    # Add edges from HZ
    for i in range(mz):
        for j in np.where(HZ[i] == 1)[0]:
            G.add_edge(z_checks[i], qubits[j], etype="Z")

    return G, x_checks, z_checks, qubits


def plot_combined_tanner_one_figure(G, x_checks, z_checks, qubits, title="Combined CSS Tanner graph"):
    # Layout: left column checks (X then Z), right column qubits
    pos = {}

    # stack X checks then Z checks
    y = 0
    for node in x_checks:
        pos[node] = (0, -y); y += 1
    for node in z_checks:
        pos[node] = (0, -y); y += 1

    for j, q in enumerate(qubits):
        pos[q] = (1, -j)

    plt.figure(figsize=(14, 10))

    # draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=x_checks, node_size=260)
    nx.draw_networkx_nodes(G, pos, nodelist=z_checks, node_size=260)
    nx.draw_networkx_nodes(G, pos, nodelist=qubits,   node_size=220)

    # draw edges (no manual colors required; matplotlib defaults are fine)
    nx.draw_networkx_edges(G, pos, width=1.0)

    # labels only if small
    if len(G.nodes) <= 80:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def build_tanner_graph(H, check_prefix="c", var_prefix="q"):
    """
    Standard Tanner graph for a binary matrix H (m x n):
      checks on one side, variables on the other.
    """
    H = np.array(H, dtype=np.uint8) % 2
    m, n = H.shape
    G = nx.Graph()
    checks = [f"{check_prefix}{i}" for i in range(m)]
    vars_  = [f"{var_prefix}{j}" for j in range(n)]
    G.add_nodes_from(checks, bipartite=0)
    G.add_nodes_from(vars_, bipartite=1)

    for i in range(m):
        for j in np.where(H[i] == 1)[0]:
            G.add_edge(checks[i], vars_[j])
    return G, checks, vars_


def plot_bipartite(G, checks, vars_, title="", max_labels=80):
    # simple left-right layout
    pos = {}
    for i, c in enumerate(checks):
        pos[c] = (0, -i)
    for j, v in enumerate(vars_):
        pos[v] = (1, -j)

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=False, node_size=220)
    if len(G.nodes) <= max_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_degree_hist(H, title=""):
    H = np.array(H, dtype=np.uint8) % 2
    col_deg = H.sum(axis=0)  # variable node degrees
    row_deg = H.sum(axis=1)  # check node degrees

    plt.figure(figsize=(10, 4))
    plt.hist(col_deg, bins=np.arange(col_deg.max() + 2) - 0.5)
    plt.title(f"{title} qubit (variable) degrees")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(row_deg, bins=np.arange(row_deg.max() + 2) - 0.5)
    plt.title(f"{title} check degrees")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def build_anticommutation_graph(HX, HZ):
    """
    Bipartite graph: X-checks vs Z-checks.
    Edge means anticommute (overlap parity  1).
    For a valid CSS code: should have 0 edges.
    """
    HX = np.array(HX, dtype=np.uint8) % 2
    HZ = np.array(HZ, dtype=np.uint8) % 2
    overlap = gf2_matmul(HX, HZ.T)  # mx x mz

    mx, mz = overlap.shape
    G = nx.Graph()
    Xnodes = [f"X{i}" for i in range(mx)]
    Znodes = [f"Z{j}" for j in range(mz)]
    G.add_nodes_from(Xnodes, bipartite=0)
    G.add_nodes_from(Znodes, bipartite=1)

    for i in range(mx):
        for j in np.where(overlap[i] == 1)[0]:
            G.add_edge(Xnodes[i], Znodes[j])
    return G


def estimate_girth(G):
    """
    Estimate girth (length of shortest cycle) in an unweighted graph.
    Returns None if acyclic.
    """
    girth = None
    for start in G.nodes:
        # BFS from start
        dist = {start: 0}
        parent = {start: None}
        queue = [start]
        for v in queue:
            for u in G.neighbors(v):
                if u not in dist:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif parent[v] != u:
                    # found a cycle
                    cycle_len = dist[v] + dist[u] + 1
                    if girth is None or cycle_len < girth:
                        girth = cycle_len
    return girth

def plot_all_graphs_in_one_window(HX, HZ):
    GX, cx, qx = build_tanner_graph(HX, check_prefix="x", var_prefix="q")
    GZ, cz, qz = build_tanner_graph(HZ, check_prefix="z", var_prefix="q")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    #  HX Tanner 
    posX = {}
    for i, c in enumerate(cx):
        posX[c] = (0, -i)
    for j, q in enumerate(qx):
        posX[q] = (1, -j)

    nx.draw(GX, posX, ax=axes[0, 0], node_size=120, with_labels=False)
    axes[0, 0].set_title("Tanner Graph H_X")
    axes[0, 0].axis("off")

    #  HZ Tanner 
    posZ = {}
    for i, c in enumerate(cz):
        posZ[c] = (0, -i)
    for j, q in enumerate(qz):
        posZ[q] = (1, -j)

    nx.draw(GZ, posZ, ax=axes[0, 1], node_size=120, with_labels=False)
    axes[0, 1].set_title("Tanner Graph H_Z")
    axes[0, 1].axis("off")

    #  HX degree histogram 
    col_deg_X = HX.sum(axis=0)
    axes[1, 0].hist(col_deg_X, bins=np.arange(col_deg_X.max()+2)-0.5)
    axes[1, 0].set_title("H_X Variable Degrees")

    #  HZ degree histogram 
    col_deg_Z = HZ.sum(axis=0)
    axes[1, 1].hist(col_deg_Z, bins=np.arange(col_deg_Z.max()+2)-0.5)
    axes[1, 1].set_title("H_Z Variable Degrees")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    HX, HZ, ok, A = generate_commuting_css_bicycle(r=18, a_row_weight=3, seed=7)
    print("HX shape:", HX.shape, "HZ shape:", HZ.shape)
    print('CSS commutation HX*HZ^T  0 ?', ok)

    plot_all_graphs_in_one_window(HX, HZ)