"""
ldpc_min_sum_numba.py
Numba-accelerated Min-Sum LDPC BER simulator.
Keeps the same Min-Sum math (sgn * min) and LLR scaling q = 2*y/sigma2.
Requires: numba, numpy, matplotlib, scipy (for erfc)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import erfc
from numba import njit, prange

# ---- Utility: build regular H with adjacency lists ----
def build_regular_h_adj(n, col_w, row_w, rng=None, max_attempts=1000):
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
        # check duplicates quickly
        if len(np.unique(pairs, axis=0)) != pairs.shape[0]:
            continue
        H = np.zeros((m, n), dtype=np.int8)
        for r, c in pairs:
            H[r, c] = 1
        if not (np.all(H.sum(axis=0) == col_w) and np.all(H.sum(axis=1) == row_w)):
            continue
        # build adjacency lists (Python lists of arrays)
        checks_to_vars = [np.where(H[i, :] == 1)[0].astype(np.int32) for i in range(m)]
        vars_to_checks = [np.where(H[:, j] == 1)[0].astype(np.int32) for j in range(n)]
        return H.astype(np.int8), np.array([len(a) for a in checks_to_vars], dtype=np.int32), checks_to_vars, vars_to_checks
    raise RuntimeError("Failed to build H")

# ---- Numba-friendly helper: convert adjacency lists to arrays with offsets ----
def adjlists_to_flat(checks_to_vars, vars_to_checks):
    # flatten checks_to_vars
    m = len(checks_to_vars)
    c_ptr = np.zeros(m+1, dtype=np.int32)
    c_flat = np.zeros(sum(len(a) for a in checks_to_vars), dtype=np.int32)
    idx = 0
    for i in range(m):
        arr = checks_to_vars[i]
        c_ptr[i] = idx
        c_flat[idx:idx+arr.size] = arr
        idx += arr.size
    c_ptr[m] = idx
    # vars_to_checks
    n = len(vars_to_checks)
    v_ptr = np.zeros(n+1, dtype=np.int32)
    v_flat = np.zeros(sum(len(a) for a in vars_to_checks), dtype=np.int32)
    idx = 0
    for j in range(n):
        arr = vars_to_checks[j]
        v_ptr[j] = idx
        v_flat[idx:idx+arr.size] = arr
        idx += arr.size
    v_ptr[n] = idx
    return c_flat.astype(np.int32), c_ptr.astype(np.int32), v_flat.astype(np.int32), v_ptr.astype(np.int32)

# ---- Numba-accelerated core functions ----
@njit(fastmath=True)
def horizontal_min_sum_numba(m, n, c_flat, c_ptr, Q):
    # Q: m x n array (only entries where H=1 are meaningful; non-connected entries are 0)
    R = np.zeros_like(Q)
    for i in range(m):
        start = c_ptr[i]
        end = c_ptr[i+1]
        deg = end - start
        for k in range(start, end):
            j = c_flat[k]
            # compute sign and min over others
            sgn = 1.0
            min_abs = 1e100
            for kk in range(start, end):
                jj = c_flat[kk]
                if jj == j:
                    continue
                val = Q[i, jj]
                a = abs(val)
                if a < min_abs:
                    min_abs = a
                if val < 0:
                    sgn = -sgn
            if deg <= 1:
                R[i, j] = 0.0
            else:
                R[i, j] = sgn * min_abs
    return R

@njit(fastmath=True)
def vertical_numba(m, n, v_flat, v_ptr, R, q):
    Qnew = np.zeros_like(R)
    for j in range(n):
        start = v_ptr[j]
        end = v_ptr[j+1]
        for idx in range(start, end):
            i = v_flat[idx]
            s = 0.0
            for idx2 in range(start, end):
                ip = v_flat[idx2]
                if ip == i:
                    continue
                s += R[ip, j]
            Qnew[i, j] = q[j] + s
    return Qnew

@njit(fastmath=True)
def compute_qhat_numba(m, n, v_flat, v_ptr, R, q):
    qhat = np.zeros(n)
    for j in range(n):
        s = 0.0
        start = v_ptr[j]
        end = v_ptr[j+1]
        for idx in range(start, end):
            i = v_flat[idx]
            s += R[i, j]
        qhat[j] = q[j] + s
    return qhat

@njit(fastmath=True)
def syndrome_numba(m, n, v_flat, v_ptr, decoded):
    # returns True if syndrome all zeros (fast check using adjacency)
    for j in range(n):
        # do nothing here; compute directly below
        pass
    # Instead compute H*decoded (we need H but we can compute using v_ptr/c_flat)
    # We will not use this convenience; user can check outside numba if needed.
    return False

@njit(fastmath=True, parallel=True)
def decode_many_frames_numba(m, n, c_flat, c_ptr, v_flat, v_ptr, x0, sigma2, frames, max_iter, out_decoded, out_iters, out_conv):
    rng_state = 123456789  # deterministic LCG inside Numba, simple but deterministic
    for f in prange(frames):
        # Simple LCG for randomness (not cryptographic) because numpy RNG isn't available inside njit easily
        # initialize xorshift-like state from f
        state = (rng_state + f * 1103515245) & 0x7fffffff
        # generate noise
        y = np.empty(n)
        for j in range(n):
            # simple Gaussian via Box-Muller pair: generate two normals per loop (but here we use a cheap approx)
            # For clarity we use a uniform->normal via inverse transform approx (not high quality); OK for speed tests
            # We'll produce noise using a simple normal approximation using sum of uniforms
            u = 0.0
            for t in range(12):
                state = (1664525 * state + 1013904223) & 0xffffffff
                u += (state & 0xffff) / 65535.0
            z = (u - 6.0)  # approx N(0,1)
            y[j] = x0[j] + np.sqrt(sigma2) * z
        # q initialization
        q = np.empty(n)
        for j in range(n):
            q[j] = 2.0 * y[j] / sigma2
        Q = np.zeros((m, n))
        for i in range(m):
            start = c_ptr[i]
            end = c_ptr[i+1]
            for k in range(start, end):
                j = c_flat[k]
                Q[i, j] = q[j]
        decoded = np.zeros(n, dtype=np.int32)
        converged = False
        it_used = 0
        for it in range(1, max_iter+1):
            R = horizontal_min_sum_numba(m, n, c_flat, c_ptr, Q)
            Q = vertical_numba(m, n, v_flat, v_ptr, R, q)
            qhat = compute_qhat_numba(m, n, v_flat, v_ptr, R, q)
            for j in range(n):
                decoded[j] = 1 if qhat[j] < 0 else 0
            # check syndrome quickly (H*decoded mod2): use v_ptr to sum bits and check parity
            ok = True
            for i in range(m):
                start = c_ptr[i]
                end = c_ptr[i+1]
                s = 0
                for k in range(start, end):
                    jj = c_flat[k]
                    s += decoded[jj]
                if (s % 2) != 0:
                    ok = False
                    break
            if ok:
                converged = True
                it_used = it
                break
            it_used = it
        # store outputs
        for j in range(n):
            out_decoded[f, j] = decoded[j]
        out_iters[f] = it_used
        out_conv[f] = 1 if converged else 0

# ---- Simulation wrapper (uses the compiled decode_many_frames) ----
def simulate_numba(n=100, col_w=3, row_w=6, frames_per_snr=2000, ebn0_dB_range=np.arange(0,7,1), max_iter=50):
    H, _, checks_to_vars, vars_to_checks = build_regular_h_adj(n, col_w, row_w, rng=np.random)
    c_flat, c_ptr, v_flat, v_ptr = adjlists_to_flat(checks_to_vars, vars_to_checks)
    m = len(checks_to_vars)
    k = n - m
    Rc = k / n
    x0 = np.ones(n)  # all-zero codeword -> BPSK +1
    for eb in ebn0_dB_range:
        EbN0_lin = 10**(eb/10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        frames = frames_per_snr
        out_decoded = np.zeros((frames, n), dtype=np.int32)
        out_iters = np.zeros(frames, dtype=np.int32)
        out_conv = np.zeros(frames, dtype=np.int32)
        t0 = time.time()
        decode_many_frames_numba(m, n, c_flat, c_ptr, v_flat, v_ptr, x0, sigma2, frames, max_iter, out_decoded, out_iters, out_conv)
        t1 = time.time()
        bit_errors = out_decoded.sum()  # since transmitted all zero
        ber = bit_errors / (frames * n)
        avg_it = out_iters.mean()
        print(f"Eb/N0={eb:.1f} dB BER={ber:.3e} avg_iter={avg_it:.2f} time={t1-t0:.2f}s")
    # plotting omitted for brevity

# Example quick test (small frames)
if __name__ == "__main__":
    simulate_numba(n=100, col_w=3, row_w=6, frames_per_snr=200, ebn0_dB_range=np.arange(0,6,1), max_iter=50)
