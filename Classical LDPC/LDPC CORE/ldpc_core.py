import numpy as np
import time

# Step 1: Build a regular LDPC parity-check matrix

def build_regular_h(n=1000, col_w=3, row_w=6, max_attempts=5000, seed=12345):
    """
    Build a regular (m x n) parity-check matrix H with:
      - each column has weight = col_w
      - each row has weight = row_w
    For n=1000, col_w=3, row_w=6 -> m = n*col_w/row_w = 500.
    A fixed RNG seed is used so the same H is generated every run.
    """
    rng = np.random.RandomState(seed)
    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n * col_w must be divisible by row_w")
    m = total_ones // row_w

    col_stubs_base = np.repeat(np.arange(n), col_w)

    for attempt in range(max_attempts):
        col_stubs = col_stubs_base.copy()
        row_stubs = np.repeat(np.arange(m), row_w)
        rng.shuffle(row_stubs)
        pairs = np.vstack((row_stubs, col_stubs)).T  # (total_ones, 2)

        # check duplicates (avoid parallel edges)
        if len({tuple(rc) for rc in pairs}) != pairs.shape[0]:
            continue

        H = np.zeros((m, n), dtype=int)
        H[pairs[:, 0], pairs[:, 1]] = 1

        # check degrees
        if np.all(H.sum(axis=0) == col_w) and np.all(H.sum(axis=1) == row_w):
            return H, m

    raise RuntimeError("Failed to build regular H after many attempts")


# Step 2: Hard decision and syndrome

def hard_decision(q_hat):
    """
    Hard decision from LLRs:
      q_hat >= 0 -> bit = 0
      q_hat <  0 -> bit = 1
    """
    return (q_hat < 0).astype(int)


def syndrome(H, c):
    """
    Compute binary syndrome vector s = H * c^T (mod 2).
    If s == 0 -> all parity checks satisfied (valid codeword).
    """
    return H.dot(c) % 2


# Step 3: Min-Sum decoding (iterative)

def min_sum_decode(H, y, sigma2, max_iter=50, c_true=None, ber_traj=None):
    """
    Min-Sum decoding on a single received vector y.

    Parameters:
      H        : (m x n) parity-check matrix
      y        : received BPSK samples (length n)
      sigma2   : noise variance
      max_iter : maximum number of iterations
      c_true   : optional true codeword (0/1) for tracking BER vs iteration
      ber_traj : optional array of length max_iter to store BER per iteration

    Returns:
      decoded  : final hard-decoded bits (0/1, length n)
      it_used  : number of iterations actually used
      converged: True if syndrome became zero before max_iter
    """
    m, n = H.shape
    # Channel LLRs (same math as MATLAB lab): q_j = 2*y_j / sigma^2
    q = 2.0 * y / sigma2

    # Initialize variable-to-check messages Q
    Q = np.zeros((m, n), dtype=float)
    rows, cols = np.where(H == 1)
    Q[rows, cols] = q[cols]

    decoded = np.zeros(n, dtype=int)
    converged = False
    it_used = 0

    for it in range(1, max_iter + 1):
        # Horizontal step: check nodes → variable nodes
        R = np.zeros((m, n), dtype=float)
        for i in range(m):
            idx_vars = np.flatnonzero(H[i, :])
            vals = Q[i, idx_vars]
            signs = np.sign(vals)
            signs[signs == 0] = 1.0  # treat zero as +1 sign to avoid NaNs
            mags = np.abs(vals)

            if len(mags) == 1:
                R[i, idx_vars[0]] = 0.0
            else:
                total_sign = np.prod(signs)
                # min excluding current element for each edge (safe, small degree)
                min_except = np.array([np.min(np.delete(mags, k))
                                       for k in range(len(mags))])
                prod_except = total_sign / signs
                R[i, idx_vars] = prod_except * min_except

        # Vertical step: variable nodes → check nodes
        Q_new = np.zeros((m, n), dtype=float)
        for j in range(n):
            idx_checks = np.flatnonzero(H[:, j])
            if len(idx_checks) == 0:
                continue
            for i in idx_checks:
                other_checks = idx_checks[idx_checks != i]
                Q_new[i, j] = q[j] + np.sum(R[other_checks, j])
        Q = Q_new

        # Posterior LLR and hard decision
        q_hat = q + np.sum(R, axis=0)
        decoded = hard_decision(q_hat)

        # Track BER vs iteration for this frame if requested
        if (c_true is not None) and (ber_traj is not None):
            ber_traj[it - 1] = np.sum(decoded != c_true) / float(len(c_true))

        # Convergence check via syndrome
        s = syndrome(H, decoded)
        if np.all(s == 0):
            converged = True
            it_used = it
            break
        it_used = it

    return decoded, it_used, converged


# Step 4: Monte Carlo simulation (no plotting)

def simulate_min_sum_ber(
    n=1000,
    col_w=3,
    row_w=6,
    ebn0_dB=None,
    frames=None,
    max_iter=50
):
    """
    Monte Carlo BER simulation for LDPC Min-Sum (no plotting).
    Uses same AWGN / Eb/N0 mathematics as the MATLAB Hamming lab.

    Parameters:
      n        : codeword length
      col_w    : column weight of H
      row_w    : row weight of H
      ebn0_dB  : array of Eb/N0 values in dB (0..10 by default)
      frames   : list of number of frames per SNR (same length as ebn0_dB)
      max_iter : maximum number of iterations in Min-Sum

    Returns:
      ebn0_dB       : SNR array
      ber           : BER for each SNR
      avg_iters     : average iterations used per SNR
      ber_vs_iter   : per-iteration BER trajectory for first frame at each SNR
      H             : parity-check matrix used
      info          : dict with meta info (n, k, Rc, col_w, row_w)
    """
    if ebn0_dB is None:
        ebn0_dB = np.arange(0, 11, 1)  # 0..10 dB
        # ebn0_dB = np.arange(0, 7, 1)  # 0..6 dB

    ebn0_dB = np.array(ebn0_dB, dtype=float)
    num_snr = len(ebn0_dB)

    if frames is None:
        # default frames per SNR – your last LDPC list
        #frames = [10, 100, 1000, 1000, 10000, 10000, 20000, 50000, 100000, 100000, 200000]
        frames = [10, 100, 1000, 10000, 50000, 100000, 200000]
        #frames = [10, 100, 1000, 10000, 20000, 30000, 50000, 100000, 200000, 300000, 400000]
    frames = np.array(frames, dtype=int)
    assert len(frames) == num_snr, "frames must match length of ebn0_dB"

    # Build H once and reuse for all SNRs
    H, m = build_regular_h(n=n, col_w=col_w, row_w=row_w)
    k = n - m
    Rc = k / float(n)

    print(f"H shape: {H.shape} (rows = N-K = {m}, cols = N = {n})")
    print(f"Code rate Rc = {Rc:.3f}")
    print(f"Column weight = {col_w}, Row weight = {row_w}\n")

    # All-zero codeword (0..0) and BPSK mapping (0->+1)
    c = np.zeros(n, dtype=int)
    x0 = 1 - 2 * c

    ber = np.zeros(num_snr, dtype=float)
    avg_iters = np.zeros(num_snr, dtype=float)

    # BER vs iteration for the first frame at each SNR
    ber_vs_iter = np.full((num_snr, max_iter), np.nan)

    rng = np.random.default_rng(2025)

    for idx, eb in enumerate(ebn0_dB):
        frames_here = frames[idx]
        EbN0_lin = 10.0 ** (eb / 10.0)

        # same variance formula as MATLAB:
        # sigma^2 = 1 / (2 * Rc * Eb/N0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        sigma = np.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0
        total_bits = frames_here * n

        t0 = time.time()
        for f in range(frames_here):
            noise = sigma * rng.standard_normal(n)
            y = x0 + noise

            # For the first frame at this SNR, record BER vs iteration
            if f == 0:
                traj = np.zeros(max_iter, dtype=float)
                decoded, it_used, converged = min_sum_decode(
                    H, y, sigma2, max_iter=max_iter,
                    c_true=c, ber_traj=traj
                )
                ber_vs_iter[idx, :] = traj
            else:
                decoded, it_used, converged = min_sum_decode(
                    H, y, sigma2, max_iter=max_iter
                )

            bit_errors += np.sum(decoded != c)
            total_iters += it_used

        elapsed = time.time() - t0
        ber[idx] = bit_errors / float(total_bits)
        avg_iters[idx] = total_iters / float(frames_here)

        # syndrome of last decoded word (just to see how many checks violated)
        s_last = syndrome(H, decoded)
        unsat_checks = np.sum(s_last != 0)

        print(
            f"Eb/N0={eb:.1f} dB, frames={frames_here}, "
            f"bit_errors={bit_errors}, BER={ber[idx]:.3e}, "
            f"avg_it={avg_iters[idx]:.2f}, unsat_checks_last={unsat_checks}, "
            f"time={elapsed:.2f}s"
        )

    info = {
        "n": n,
        "k": k,
        "m": m,
        "Rc": Rc,
        "col_w": col_w,
        "row_w": row_w,
        "frames": frames
    }

    return ebn0_dB, ber, avg_iters, ber_vs_iter, H, info
