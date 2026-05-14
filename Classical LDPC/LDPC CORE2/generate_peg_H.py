import numpy as np

def build_peg_ldpc(n=1000, col_w=3, row_w=6, seed=12345):
    """
    Simple PEG-style construction for a regular (col_w, row_w) LDPC matrix.

    This is not a full industrial PEG implementation, but it follows
    the idea: we connect edges one by one, trying to avoid short cycles,
    and enforce exact degree per variable and per check.

    Returns:
        H : (m x n) numpy array of 0/1
    """
    rng = np.random.default_rng(seed)

    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n * col_w must be divisible by row_w")

    m = total_ones // row_w

    # target degrees
    var_deg_target = np.full(n, col_w, dtype=int)
    chk_deg_target = np.full(m, row_w, dtype=int)

    # current degrees
    var_deg = np.zeros(n, dtype=int)
    chk_deg = np.zeros(m, dtype=int)

    # adjacency lists we are building
    var_neigh = [[] for _ in range(n)]
    chk_neigh = [[] for _ in range(m)]

    # H to fill
    H = np.zeros((m, n), dtype=int)

    # Helper: compute a "distance" heuristic for a candidate edge (v, c)
    # We want to avoid connecting v to checks that are already
    # close in the graph to v (avoid short cycles).
    def local_penalty(v, c):
        # degree-based heuristic: prefer checks with lower degree
        # plus small penalty if v is already indirectly linked
        # to that check via neighbors-of-neighbors.
        deg_pen = chk_deg[c]

        # small cycle-avoidance heuristic: if c is neighbor of neighbors of v
        # or v is neighbor of neighbors of c, add penalty.
        penalty = 0
        for c2 in var_neigh[v]:
            # neighbors of c2
            for v2 in chk_neigh[c2]:
                if c == v2:
                    penalty += 3  # discourage 2-cycles (parallel edges)
        return deg_pen + penalty

    # Build edges progressively
    for v in range(n):
        for _ in range(col_w):
            # candidate checks that still have capacity
            candidates = np.where(chk_deg < chk_deg_target)[0]
            if len(candidates) == 0:
                raise RuntimeError("No available check nodes left to connect.")

            # evaluate penalty for each candidate
            penalties = np.array([local_penalty(v, c) for c in candidates])
            # choose the candidate with minimal penalty (PEG-like choice)
            best_idx = np.argmin(penalties)
            c = candidates[best_idx]

            # add edge (c, v)
            H[c, v] = 1
            var_deg[v] += 1
            chk_deg[c] += 1
            var_neigh[v].append(c)
            chk_neigh[c].append(v)

    # sanity checks
    assert np.all(H.sum(axis=0) == col_w), "Column weights not correct"
    assert np.all(H.sum(axis=1) == row_w), "Row weights not correct"
    return H


def write_H_to_py(H, filename="ldpc_H_matrix.py"):
    """
    Write the given H matrix into a python file as a hard-coded numpy array.
    """
    m, n = H.shape
    with open(filename, "w") as f:
        f.write("import numpy as np\n\n")
        f.write(f"# Hard-coded LDPC parity-check matrix H\n")
        f.write(f"# Shape: ({m}, {n})\n")
        f.write("H = np.array([\n")
        for i in range(m):
            row_str = ", ".join(str(int(x)) for x in H[i, :])
            f.write("    [" + row_str + "],\n")
        f.write("], dtype=int)\n")


if __name__ == "__main__":
    n = 1000
    col_w = 3
    row_w = 6
    H = build_peg_ldpc(n=n, col_w=col_w, row_w=row_w, seed=12345)
    write_H_to_py(H)
    print("Generated ldpc_H_matrix.py with H of shape", H.shape)
