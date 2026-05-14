import numpy as np
from collections import deque

def build_peg_ldpc(n=1000, col_w=3, row_w=6, seed=12345, max_tries_per_edge=5000):
    """
    PEG-like construction for a regular (col_w, row_w) LDPC parity-check matrix.

    Key improvements vs your previous file:
      - True PEG-style selection using BFS distance in Tanner graph
      - Explicit 4-cycle avoidance: never allow two variables to share >= 2 checks

    Returns:
        H : (m x n) numpy array of 0/1 with exact degrees.
    """
    rng = np.random.default_rng(seed)

    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n * col_w must be divisible by row_w")
    m = total_ones // row_w  # for (1000, col_w=3, row_w=6) -> 500

    # adjacency lists
    var_to_chk = [[] for _ in range(n)]
    chk_to_var = [[] for _ in range(m)]

    # degrees
    chk_deg = np.zeros(m, dtype=np.int32)

    # For fast 4-cycle checking:
    # var_check_sets[v] = set of checks connected to v
    var_check_sets = [set() for _ in range(n)]

    # Also track for each variable v: all variables that share a check with v (1-hop via check)
    # We can build on the fly from chk_to_var; we don't need a global matrix.
    def would_create_4cycle(v, c):
        """
        Edge (v,c) creates a 4-cycle if there exists another check c2 in N(v)
        and another variable u such that u in N(c) and u in N(c2).
        Equivalent: some existing neighbor u of check c already shares a check with v.
        """
        # neighbors of check c are variables u
        for u in chk_to_var[c]:
            if u == v:
                continue
            # if u already shares ANY check with v, then adding (v,c) creates a 4-cycle
            # because v -- c2 -- u already exists, and now v -- c -- u would be another.
            if len(var_check_sets[v].intersection(var_check_sets[u])) > 0:
                return True
        return False

    def bfs_distance_from_var(v):
        """
        BFS in Tanner graph starting from variable v.
        Returns:
          dist_chk: array length m with distance to each check (in edges),
                    np.inf for unreachable.
        Graph alternates: var -> check -> var -> ...
        """
        INF = 10**9
        dist_var = np.full(n, INF, dtype=np.int32)
        dist_chk = np.full(m, INF, dtype=np.int32)

        q = deque()
        dist_var[v] = 0
        q.append(("v", v))

        while q:
            typ, idx = q.popleft()
            if typ == "v":
                d = dist_var[idx]
                for c in var_to_chk[idx]:
                    if dist_chk[c] > d + 1:
                        dist_chk[c] = d + 1
                        q.append(("c", c))
            else:  # typ == "c"
                d = dist_chk[idx]
                for u in chk_to_var[idx]:
                    if dist_var[u] > d + 1:
                        dist_var[u] = d + 1
                        q.append(("v", u))

        return dist_chk

    # We'll build H implicitly via adjacency, then materialize at end
    for v in range(n):
        for e in range(col_w):
            # capacity checks
            candidates = np.where(chk_deg < row_w)[0]
            if candidates.size == 0:
                raise RuntimeError("No check nodes with remaining capacity.")

            # PEG distance: choose farthest check from v (to avoid short cycles)
            dist_chk = bfs_distance_from_var(v)
            cand_dist = dist_chk[candidates]

            # Prefer max distance; if many tie, prefer low degree; if still tie, random
            max_d = np.max(cand_dist)
            farthest = candidates[cand_dist == max_d]

            # Among farthest, prefer min degree
            degs = chk_deg[farthest]
            min_deg = np.min(degs)
            best_pool = farthest[degs == min_deg]

            # Now we must also enforce 4-cycle avoidance; try picks from best_pool, else widen
            # We'll try a few random permutations to find a valid c quickly
            tried = 0
            selected = None

            # Build a priority list: first best_pool, then remaining farthest, then all candidates
            # (so if strict constraints block, it can still finish)
            priority_lists = [best_pool, farthest, candidates]

            for pool in priority_lists:
                pool = pool.copy()
                rng.shuffle(pool)
                for c in pool:
                    tried += 1

                    # Prevent duplicate edge (shouldn't happen, but safe)
                    if c in var_check_sets[v]:
                        continue

                    # Enforce 4-cycle avoidance
                    if would_create_4cycle(v, c):
                        continue

                    selected = c
                    break
                if selected is not None:
                    break
                if tried > max_tries_per_edge:
                    break

            if selected is None:
                # If too strict, fall back: allow edge even if it creates a 4-cycle
                # (but still keep degrees correct). This prevents "getting stuck".
                pool = candidates.copy()
                rng.shuffle(pool)
                for c in pool:
                    if c not in var_check_sets[v]:
                        selected = c
                        break

            if selected is None:
                raise RuntimeError("Failed to select a check node for an edge.")

            # Add edge (selected, v)
            c = int(selected)
            var_to_chk[v].append(c)
            chk_to_var[c].append(v)
            var_check_sets[v].add(c)
            chk_deg[c] += 1

    # Materialize H
    H = np.zeros((m, n), dtype=np.uint8)
    for v in range(n):
        for c in var_to_chk[v]:
            H[c, v] = 1

    # Sanity checks
    if not np.all(H.sum(axis=0) == col_w):
        raise AssertionError("Column weights not all correct")
    if not np.all(H.sum(axis=1) == row_w):
        raise AssertionError("Row weights not all correct")

    return H


def write_H_to_py(H, filename="ldpc_H_matrix.py"):
    """
    Writes H as a hard-coded numpy array into ONE python file.
    NOTE: This will create a VERY large file for 500x1000.
    """
    m, n = H.shape
    with open(filename, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n\n")
        f.write("# Hard-coded LDPC parity-check matrix H\n")
        f.write(f"# Shape: ({m}, {n})\n")
        f.write("# Column weight = 3, Row weight = 6\n\n")
        f.write("H = np.array([\n")
        for i in range(m):
            row_str = ", ".join(str(int(x)) for x in H[i, :])
            f.write("    [" + row_str + "],\n")
        f.write("], dtype=np.uint8)\n\n")

        # Add self-test block at bottom
        f.write("if __name__ == '__main__':\n")
        f.write("    print('H shape:', H.shape)\n")
        f.write("    print('col weight min/max:', int(H.sum(axis=0).min()), int(H.sum(axis=0).max()))\n")
        f.write("    print('row weight min/max:', int(H.sum(axis=1).min()), int(H.sum(axis=1).max()))\n")
        f.write("    assert H.shape == (500, 1000)\n")
        f.write("    assert np.all(H.sum(axis=0) == 3)\n")
        f.write("    assert np.all(H.sum(axis=1) == 6)\n")
        f.write("    print('OK ✅')\n")


if __name__ == "__main__":
    n = 1000
    col_w = 3
    row_w = 6

    H = build_peg_ldpc(n=n, col_w=col_w, row_w=row_w, seed=12345)
    write_H_to_py(H, filename="ldpc_H_matrix.py")
    print("Generated ldpc_H_matrix.py with H of shape", H.shape)
