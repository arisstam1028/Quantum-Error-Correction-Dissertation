"""
ldpc_min_sum_numpy.py
Optimized NumPy implementation using sparse adjacency arrays and vectorized operations
Keeps the same Min-Sum math; no Numba required.
Requires: numpy, scipy.sparse, matplotlib, scipy.special.erfc
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy import sparse

def build_regular_h(n, col_w, row_w, rng=None, max_attempts=1000):
    if rng is None:
        rng = np.random
    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n * col_w must be divisible by row_w")
    m = total_ones // row_w
    for _ in range(max_attempts):
        col_stubs = np.repeat(np.arange(n), col_w)
        row_stubs = np.repeat(np.arange(m), row_w)
        rng.shuffle(row_stubs)
        pairs = np.vstack((row_stubs, col_stubs)).T
        if len(np.unique(pairs, axis=0)) != pairs.shape[0]:
            continue
        H = np.zeros((m, n), dtype=np.int8)
        for r, c in pairs:
            H[r, c] = 1
        if not (np.all(H.sum(axis=0) == col_w) and np.all(H.sum(axis=1) == row_w)):
            continue
        # store adjacency as lists of arrays
        checks_to_vars = [np.where(H[i, :] == 1)[0].astype(np.int32) for i in range(m)]
        vars_to_checks = [np.where(H[:, j] == 1)[0].astype(np.int32) for j in range(n)]
        return H, checks_to_vars, vars_to_checks
    raise RuntimeError("Failed to build H")

def min_sum_decode_numpy(H, checks_to_vars, vars_to_checks, y, sigma2, max_iter=50):
    m, n = H.shape
    q = (2.0 * y) / sigma2
    # initialize Q matrix only for edges using dict of arrays (list of arrays)
    Q = [None]*m
    for i in range(m):
        deg = checks_to_vars[i].size
        Q[i] = q[checks_to_vars[i]].astype(float)  # vector of length deg
    R = [np.zeros_like(Qi) for Qi in Q]
    for it in range(1, max_iter+1):
        # horizontal: for each check, compute messages to connected variables
        for i in range(m):
            neighbors = checks_to_vars[i]
            vals = Q[i]  # q for that check to each variable
            if vals.size <= 1:
                R[i][:] = 0.0
                continue
            signs = np.sign(vals)
            mags = np.abs(vals)
            min_all = np.min(mags)
            # For each outgoing edge j_index, we need min over all except j_index; we can compute using:
            # min_except = minimum of (prefix_min, suffix_min) trick for speed
            prefix = np.minimum.accumulate(mags)
            suffix = np.minimum.accumulate(mags[::-1])[::-1]
            min_except = np.empty_like(mags)
            for k in range(vals.size):
                if k == 0:
                    min_except[k] = suffix[1]
                elif k == vals.size-1:
                    min_except[k] = prefix[-2]
                else:
                    min_except[k] = min(prefix[k-1], suffix[k+1])
            # product of signs except self: use total_sign / sign[k]
            total_sign = np.prod(signs)
            prod_except = total_sign / signs
            R[i] = prod_except * min_except
        # vertical: build Q for each check i from variable messages
        # For each variable j, find all checks and sum R from those checks (exclude target)
        # Precompute q_hat per variable
        qhat = np.zeros(n)
        for j in range(n):
            checks = vars_to_checks[j]
            # gather R values for this variable from each check
            vals = np.array([ R[i][ np.where(checks_to_vars[i]==j)[0][0] ] for i in checks ])
            qhat[j] = q[j] + vals.sum()
        decoded = (qhat < 0).astype(int)
        # check syndrome
        s = (H.dot(decoded) % 2)
        if np.all(s == 0):
            return decoded, it, True, qhat
        # update Q for next iter: for each check i, update Q[i] entries
        for i in range(m):
            neighbors = checks_to_vars[i]
            for idx, j in enumerate(neighbors):
                # Q[i][idx] = q[j] + sum_{i' in M_j \ i} R[i', j]
                other_checks = vars_to_checks[j]
                # gather R values for variable j from other checks
                vals = []
                for ip in other_checks:
                    if ip == i:
                        continue
                    # find index of j in checks_to_vars[ip]
                    pos = np.where(checks_to_vars[ip]==j)[0][0]
                    vals.append(R[ip][pos])
                Q[i][idx] = q[j] + (np.sum(vals) if len(vals)>0 else 0.0)
    # not converged
    return decoded, max_iter, False, qhat

def simulate_numpy(n=100, col_w=3, row_w=6, ebn0_dB_range=np.arange(0,7,1), frames_per_snr=2000, max_iter=50):
    H, checks_to_vars, vars_to_checks = build_regular_h(n, col_w, row_w, rng=np.random)
    m = len(checks_to_vars)
    k = n - m
    Rc = k/n
    x0 = np.ones(n)
    for eb in ebn0_dB_range:
        EbN0_lin = 10**(eb/10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        frames = frames_per_snr
        bit_errors = 0
        total_iters = 0
        t0 = time.time()
        for f in range(frames):
            noise = np.sqrt(sigma2)*np.random.randn(n)
            y = x0 + noise
            decoded, it_used, conv, qhat = min_sum_decode_numpy(H, checks_to_vars, vars_to_checks, y, sigma2, max_iter=max_iter)
            bit_errors += np.sum(decoded != 0)
            total_iters += it_used
        t1 = time.time()
        print(f"Eb/N0={eb:.1f} dB BER={bit_errors/(frames*n):.3e} avg_it={total_iters/frames:.2f} time={t1-t0:.2f}s")
    # plotting omitted

if __name__ == "__main__":
    simulate_numpy(n=100, col_w=3, row_w=6, frames_per_snr=200)
