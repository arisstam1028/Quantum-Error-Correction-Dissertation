# Purpose:
#   Builds the Tanner graph data structure used by the binary BP decoder.
#
# Process:
#   1. Read non-zero entries of the parity-check matrix H.
#   2. Build check-to-variable and variable-to-check adjacency lists.
#   3. Store flat edge arrays for efficient message updates.
#
# Theory link:
#   Belief propagation passes messages along Tanner graph edges. Check
#   nodes enforce syndrome/parity constraints, while variable nodes
#   represent candidate bits of the binary error pattern.

from dataclasses import dataclass

import numpy as np


@dataclass
class BPGraph:
    H: np.ndarray
    check_to_var: list[list[int]]
    var_to_check: list[list[int]]

    # Flat edge representation
    edge_var: np.ndarray           # shape (E,)
    edge_check: np.ndarray         # shape (E,)

    # Check-major edge layout
    check_edge_ptr: np.ndarray     # shape (m + 1,)
    check_edges: np.ndarray        # shape (E,)

    # Variable-major edge layout
    var_edge_ptr: np.ndarray       # shape (n + 1,)
    var_edges: np.ndarray          # shape (E,)

    # Fast lookup: edge id for pair (v, c), else -1
    edge_id_of_pair: np.ndarray    # shape (n, m)

    # Local positions
    edge_pos_in_check: np.ndarray  # shape (E,)
    edge_pos_in_var: np.ndarray    # shape (E,)


def build_bp_graph(H: np.ndarray) -> BPGraph:
    """
    Convert a binary parity-check matrix into BP graph arrays.

    Role in pipeline:
        Provides the edge ordering and adjacency information used by the
        check-node and variable-node message updates in BP decoding.
    """
    H = np.asarray(H, dtype=np.uint8)
    if H.ndim != 2:
        raise ValueError("H must be 2D")

    m, n = H.shape

    check_to_var: list[list[int]] = []
    for i in range(m):
        check_to_var.append(np.where(H[i] == 1)[0].tolist())

    var_to_check: list[list[int]] = []
    for j in range(n):
        var_to_check.append(np.where(H[:, j] == 1)[0].tolist())

    # Build edges in check-major order
    edge_var_list: list[int] = []
    edge_check_list: list[int] = []
    edge_pos_in_check_list: list[int] = []

    check_edge_ptr = np.zeros(m + 1, dtype=np.int64)
    for c in range(m):
        check_edge_ptr[c] = len(edge_var_list)
        for local_pos, v in enumerate(check_to_var[c]):
            edge_var_list.append(v)
            edge_check_list.append(c)
            edge_pos_in_check_list.append(local_pos)
    check_edge_ptr[m] = len(edge_var_list)

    edge_var = np.asarray(edge_var_list, dtype=np.int64)
    edge_check = np.asarray(edge_check_list, dtype=np.int64)
    edge_pos_in_check = np.asarray(edge_pos_in_check_list, dtype=np.int64)

    E = edge_var.size
    check_edges = np.arange(E, dtype=np.int64)

    # Variable-major buckets
    var_edge_buckets: list[list[int]] = [[] for _ in range(n)]
    for e in range(E):
        v = int(edge_var[e])
        var_edge_buckets[v].append(e)

    var_edge_ptr = np.zeros(n + 1, dtype=np.int64)
    var_edges_list: list[int] = []
    edge_pos_in_var = np.zeros(E, dtype=np.int64)

    for v in range(n):
        var_edge_ptr[v] = len(var_edges_list)
        for local_pos, e in enumerate(var_edge_buckets[v]):
            var_edges_list.append(e)
            edge_pos_in_var[e] = local_pos
    var_edge_ptr[n] = len(var_edges_list)
    var_edges = np.asarray(var_edges_list, dtype=np.int64)

    edge_id_of_pair = -np.ones((n, m), dtype=np.int64)
    for e in range(E):
        v = int(edge_var[e])
        c = int(edge_check[e])
        edge_id_of_pair[v, c] = e

    return BPGraph(
        H=H,
        check_to_var=check_to_var,
        var_to_check=var_to_check,
        edge_var=edge_var,
        edge_check=edge_check,
        check_edge_ptr=check_edge_ptr,
        check_edges=check_edges,
        var_edge_ptr=var_edge_ptr,
        var_edges=var_edges,
        edge_id_of_pair=edge_id_of_pair,
        edge_pos_in_check=edge_pos_in_check,
        edge_pos_in_var=edge_pos_in_var,
    )
