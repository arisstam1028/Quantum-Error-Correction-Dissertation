"""
Purpose:
    Build Tanner-graph lookup structures for binary BP decoders.

Process:
    Convert nonzero entries of H into flat edge arrays plus check-major and
    variable-major index ranges.

Theory link:
    BP sends messages along Tanner-graph edges. Precomputed edge layouts keep
    flooding, variable-scheduled, and check-scheduled updates consistent.
"""

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
    Convert a binary parity-check matrix into a BPGraph.

    Role in pipeline:
        Gives decoders fast access to all variables attached to a check and
        all checks attached to a variable.
    """
    H = np.asarray(H, dtype=np.uint8)
    if H.ndim != 2:
        raise ValueError("H must be 2D")

    m, n = H.shape
    rows, cols = np.nonzero(H)
    edge_count = int(rows.size)

    check_counts = np.bincount(rows, minlength=m).astype(np.int64)
    var_counts = np.bincount(cols, minlength=n).astype(np.int64)

    check_to_var: list[list[int]] = [[] for _ in range(m)]
    var_to_check: list[list[int]] = [[] for _ in range(n)]
    for r, c in zip(rows.tolist(), cols.tolist()):
        check_to_var[r].append(c)
        var_to_check[c].append(r)

    check_edge_ptr = np.empty(m + 1, dtype=np.int64)
    check_edge_ptr[0] = 0
    np.cumsum(check_counts, out=check_edge_ptr[1:])

    edge_var = cols.astype(np.int64, copy=True)
    edge_check = rows.astype(np.int64, copy=True)
    edge_pos_in_check = np.empty(edge_count, dtype=np.int64)
    if edge_count:
        edge_pos_in_check[:] = np.concatenate(
            [np.arange(count, dtype=np.int64) for count in check_counts if count > 0]
        )

    E = edge_count
    check_edges = np.arange(E, dtype=np.int64)

    var_edge_ptr = np.empty(n + 1, dtype=np.int64)
    var_edge_ptr[0] = 0
    np.cumsum(var_counts, out=var_edge_ptr[1:])

    var_edges_list: list[int] = [0] * E
    edge_pos_in_var = np.zeros(E, dtype=np.int64)
    var_offsets = var_edge_ptr[:-1].copy()
    for e in range(E):
        v = int(edge_var[e])
        slot = int(var_offsets[v])
        var_edges_list[slot] = e
        edge_pos_in_var[e] = slot - int(var_edge_ptr[v])
        var_offsets[v] += 1
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
