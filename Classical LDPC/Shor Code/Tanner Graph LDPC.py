"""
Generate and plot a simple regular (dv, dc) LDPC Tanner graph.

Requirements:
  pip install networkx matplotlib numpy
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make_regular_ldpc_H(n: int, m: int, dv: int, dc: int, seed: int = 0, max_tries: int = 5000) -> np.ndarray:
    if dv * n != dc * m:
        raise ValueError("Degree constraint violated: dv*n must equal dc*m.")

    rng = np.random.default_rng(seed)

    var_stubs = np.repeat(np.arange(n), dv)
    chk_stubs = np.repeat(np.arange(m), dc)

    for _ in range(max_tries):
        rng.shuffle(chk_stubs)
        H = np.zeros((m, n), dtype=int)

        ok = True
        for v, c in zip(var_stubs, chk_stubs):
            if H[c, v] == 1:
                ok = False
                break
            H[c, v] = 1

        if ok and np.all(H.sum(axis=0) == dv) and np.all(H.sum(axis=1) == dc):
            return H

    raise RuntimeError("Failed to construct regular LDPC matrix.")


def plot_tanner_graph(H: np.ndarray) -> None:
    m, n = H.shape
    G = nx.Graph()

    var_nodes = [f"v{i+1}" for i in range(n)]
    chk_nodes = [f"c{j+1}" for j in range(m)]

    G.add_nodes_from(var_nodes, bipartite=0)
    G.add_nodes_from(chk_nodes, bipartite=1)

    for j in range(m):
        for i in range(n):
            if H[j, i] == 1:
                G.add_edge(var_nodes[i], chk_nodes[j])

    # Layout
    pos = {}
    for i, v in enumerate(var_nodes):
        pos[v] = (0.0, -i)
    for j, c in enumerate(chk_nodes):
        pos[c] = (1.0, -j * (max(1, n - 1) / max(1, m - 1)))

    plt.figure(figsize=(10, 6))

    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_shape="o", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=chk_nodes, node_shape="s", node_size=900)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")

    # Clean descriptive title
    plt.title("Variable nodes (Left), Check nodes (Right)", fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    n = 6
    m = 3
    dv = 2
    dc = 4

    H = make_regular_ldpc_H(n, m, dv, dc, seed=2)
    plot_tanner_graph(H)
