"""
Optimized LDPC Min-Sum Monte Carlo BER simulation (vectorised, fast)

Key ideas used for speed:
 - Use adjacency mask E = (H == 1) so we only compute on actual edges.
 - Compute check-node outputs (horizontal step) with vectorised row-wise
   operations: first-min, second-min and total sign product. This
   avoids Python loops per edge.
 - Compute variable-node messages (vertical step) using matrix-sums:
     Q_new = q + sum_over_checks(R) - R
   which yields the "sum excluding i" for every (i,j) in one vectorised op.
 - Compute final LLRs by summing R over checks per variable (axis sum).
 - Keep H builder mostly the same but efficient; adjacency lists are not necessary
   because everything is done with masks/sums.
"""

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc


# Build regular H (kept similar to your original but slightly more vectorised)
def build_regular_h(n, col_w, row_w, max_attempts=2000, rng=None):
    """
    Build a (m x n) binary parity-check matrix H with:
      - each column has weight col_w
      - each row has weight row_w
    """
    if rng is None:
        rng = np.random

    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n * col_w must be divisible by row_w")
    m = total_ones // row_w

    # Pre-generate column stubs
    col_stubs = np.repeat(np.arange(n), col_w)

    for attempt in range(max_attempts):
        row_stubs = np.repeat(np.arange(m), row_w)
        rng.shuffle(row_stubs)

        pairs = np.vstack((row_stubs, col_stubs)).T    # shape (total_ones, 2)

        # --- FIX: safe duplicate detection ---
        if len({tuple(r) for r in pairs}) != pairs.shape[0]:
            continue  # collision → retry

        # Build H
        H = np.zeros((m, n), dtype=np.int8)
        H[pairs[:, 0], pairs[:, 1]] = 1

        # Final structural check
        if np.all(H.sum(axis=0) == col_w) and np.all(H.sum(axis=1) == row_w):
            return H, m

    raise RuntimeError("Failed to build regular H after many attempts")


# Vectorised horizontal step: compute R (check-to-variable messages)
def horizontal_step_min_sum(H, Q):
    """
    Vectorised Min-Sum horizontal step.
    Inputs:
      H : (m x n) binary parity-check matrix
      Q : (m x n) variable-to-check messages (only meaningful where H==1)
    Returns:
      R : (m x n) check-to-variable messages (only nonzero where H==1)
    NOTES / MATH:
      For check node i and variable node j in N(i):
        R[i,j] = prod_{j' in N(i)\{j}} sign(Q[i,j']) * min_{j' in N(i)\{j}} |Q[i,j']|
      We compute per-row:
        - first minimum (min1) and index pos1
        - second minimum (min2) to handle the "exclude self" case
        - total sign product over the row (total_sign)
      Then R_sign(i,j) = total_sign(i) * sign(Q[i,j])  (because for ±1, 1/sign=sign)
      R_mag(i,j)  = min2(i) if j==pos1(i) else min1(i)
      Finally: R = R_sign * R_mag for all edges.
    """

    # Edge mask: True where edges exist
    E = (H == 1)

    # Degrees per check node (rows)
    degs = E.sum(axis=1)                       # shape (m,)

    # If a check node degree <= 1, its outputs are all zeros; handle later.
    m, n = H.shape

    # Safe mag array: set non-edges to +inf so they never affect minima
    mags_masked = np.where(E, np.abs(Q), np.inf)   # shape (m, n)

    # Row-wise first min and its argmin index
    min1 = np.min(mags_masked, axis=1)             # (m,)
    pos1 = np.argmin(mags_masked, axis=1)          # (m,)

    # Prepare second-min by replacing min1 entries with +inf
    mags2 = mags_masked.copy()
    rows_idx = np.arange(m)
    mags2[rows_idx, pos1] = np.inf
    min2 = np.min(mags2, axis=1)                   # (m,)

    # Signs masked: for non-edges use 1 (neutral for product)
    sign_masked = np.where(E, np.sign(Q), 1.0)     # (m, n)
    # Total product of signs per row (1 for non-edges)
    total_sign = np.prod(sign_masked, axis=1)      # (m,)

    # Compute sign of each outgoing message (prod of others' signs)
    # For ±1 elements the inverse equals itself, so:
    # prod_except = total_sign / sign_ij  -> equals total_sign * sign_ij
    # using multiplication avoids division by zero issues (we kept non-edges=1)
    R_sign = total_sign[:, None] * np.where(E, np.sign(Q), 1.0)  # (m, n)

    # Determine where each column is the first-min for its row.
    # Build a boolean mask (m x n) that's True at (i, pos1[i]) if that is an edge.
    is_first_min = (np.arange(n)[None, :] == pos1[:, None]) & E

    # R magnitude: min2 for the first-min positions, otherwise min1
    R_mag = np.where(is_first_min, min2[:, None], min1[:, None])  # (m,n)

    # For rows with deg <= 1, set magnitudes to 0 (no meaningful check message)
    low_deg_rows = (degs <= 1)
    if np.any(low_deg_rows):
        R_mag[low_deg_rows, :] = 0.0
        R_sign[low_deg_rows, :] = 0.0

    # Final R: apply sign and magnitude, then zero-out non-edges
    R = R_sign * R_mag
    R[~E] = 0.0

    return R


# Vectorised vertical step: compute Q_new (variable-to-check messages)
def vertical_step(H, R, q):
    """
    Vectorised vertical step.
    Inputs:
      H : (m x n) binary matrix
      R : (m x n) check-to-variable messages (nonzero where H==1)
      q : (n,) channel LLRs
    Output:
      Q_new : (m x n) variable-to-check messages (only meaningful where H==1)
    MATH TRICK:
      Q_new[i,j] = q[j] + sum_{i' in M(j) \ {i}} R[i', j]
      Let total_R_col[j] = sum_{i' in M(j)} R[i', j]
      Then Q_new[i,j] = q[j] + total_R_col[j] - R[i,j]
      This computes the "sum excluding i" for all edges in one vectorised op.
    """

    E = (H == 1)
    total_R_col = R.sum(axis=0)                     # (n,)
    # Broadcast q and total_R_col to (m, n), then subtract R to exclude current check
    Q_new = q[None, :] + total_R_col[None, :] - R   # (m, n)
    # Zero-out non-edges
    Q_new[~E] = 0.0
    return Q_new


# Vectorised final LLR computation
def compute_q_hat(H, R, q):
    """
    Compute final LLRs:
      q_hat[j] = q[j] + sum_{i in M(j)} R[i, j]
    Vectorised by summing R along axis=0.
    """
    q_hat = q + R.sum(axis=0)
    return q_hat


# Single frame Min-Sum decode using the vectorised steps
def min_sum_decode_single(H, y, sigma2, max_iter=50):
    """
    Decode one received vector y using the vectorised Min-Sum algorithm.
    Returns: decoded_bits, it_used, converged_flag, q_hat_final, R_first
    """

    m, n = H.shape
    # Channel LLRs
    q = (2.0 * y) / sigma2          # (n,)

    # Initialize Q (variable-to-check): replicate q on each connected check
    Q = np.where(H, q[None, :], 0.0)   # broadcast q into shape (m,n) but zero where H==0

    R_first = None
    converged = False
    it_used = 0
    decoded = np.zeros(n, dtype=int)
    for it in range(1, max_iter + 1):
        # Horizontal: compute check-to-variable messages R (vectorised)
        R = horizontal_step_min_sum(H, Q)
        if it == 1:
            R_first = R.copy()

        # Vertical: compute Q for next iteration (vectorised)
        Q = vertical_step(H, R, q)

        # Final LLRs and decision
        q_hat = compute_q_hat(H, R, q)
        decoded = (q_hat < 0).astype(int)   # hard decision like your original

        # Check syndrome: if zero -> converged
        s = (H.dot(decoded) % 2)
        if np.all(s == 0):
            converged = True
            it_used = it
            break
        it_used = it

    return decoded, it_used, converged, q_hat, R_first


# Monte-Carlo simulation harness (unchanged workflow, but uses the faster decode)
def simulate_min_sum_ber(n=100,
                         col_w=3,
                         row_w=6,
                         ebn0_dB_range=np.arange(0.0, 6.1, 0.5),
                         frames_per_snr=2000,
                         max_iter=50,
                         seed=None,
                         verbose=True):
    """
    Vectorised Monte-Carlo Min-Sum simulation. Same semantics and returned values
    as your original script.
    """

    rng = np.random.RandomState(seed)

    # Build parity-check matrix
    H, m = build_regular_h(n, col_w, row_w, rng=rng)
    k = n - m
    Rc = float(k) / n

    # Pre-allocate
    ebn0_dB_range = np.array(ebn0_dB_range, dtype=float)
    ber = np.zeros_like(ebn0_dB_range)
    avg_iters = np.zeros_like(ebn0_dB_range)

    start_time = time.time()

    # All-zero codeword (BPSK +1)
    c = np.zeros(n, dtype=int)
    x0 = 1 - 2 * c

    for idx, ebn0_db in enumerate(ebn0_dB_range):
        EbN0_lin = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        sigma = math.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0
        total_bits = frames_per_snr * n

        if verbose:
            print(f"\nEb/N0 = {ebn0_db:.2f} dB, sigma^2={sigma2:.5e}, frames={frames_per_snr}")

        # Run frames
        for f in range(frames_per_snr):
            noise = sigma * rng.randn(n)
            y = x0 + noise

            decoded, it_used, converged, q_hat, R_first = min_sum_decode_single(H, y, sigma2, max_iter=max_iter)

            bit_errors += np.sum(decoded != c)
            total_iters += it_used

            # occasional progress
            if verbose and (f + 1) % (max(1, frames_per_snr // 10)) == 0:
                print(f"  frame {f+1}/{frames_per_snr}  current BER estimate = {bit_errors / ((f+1)*n):.3e}")

        ber[idx] = bit_errors / float(total_bits)
        avg_iters[idx] = float(total_iters) / frames_per_snr

        if verbose:
            elapsed = time.time() - start_time
            print(f"-> Eb/N0 {ebn0_db:.2f} dB: BER = {ber[idx]:.3e}, avg iters = {avg_iters[idx]:.2f}, elapsed={elapsed:.1f}s")

    # Plot
    plt.figure(figsize=(8, 5))
    ebn0_lin = 10.0 ** (ebn0_dB_range / 10.0)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))
    plt.semilogy(ebn0_dB_range, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')
    plt.semilogy(ebn0_dB_range, ber, 'o-', label=f'LDPC Min-Sum (n={n}, rate={Rc:.2f})')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0 - Min-Sum LDPC (vectorised)')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return ebn0_dB_range, ber, avg_iters


# Quick correctness test utility (optional)
def _self_test_small():
    """
    Run a tiny consistency test comparing the new vectorised functions
    against a straightforward nested-loop reference on a tiny H.
    """
    np.random.seed(0)
    n = 12
    col_w = 3
    row_w = 6
    H, m = build_regular_h(n, col_w, row_w, rng=np.random)
    # small random y
    y = 1.0 + 0.1 * np.random.randn(n)
    sigma2 = 0.1

    # Build Q manually (like original)
    Q_loop = np.zeros((m, n))
    q = (2.0 * y) / sigma2
    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                Q_loop[i, j] = q[j]

    # Compute loop-based R
    def horiz_loop(H, Q):
        m, n = H.shape
        Rl = np.zeros_like(Q)
        for i in range(m):
            for j in range(n):
                if H[i, j] == 1:
                    others = []
                    for jj in range(n):
                        if jj != j and H[i, jj] == 1:
                            others.append(Q[i, jj])
                    if len(others) == 0:
                        Rl[i, j] = 0.0
                    else:
                        sgn = 1.0
                        for val in others:
                            sgn *= np.sign(val) if val != 0 else 1.0
                        minimum = np.min(np.abs(others))
                        Rl[i, j] = sgn * minimum
        return Rl

    R_loop = horiz_loop(H, Q_loop)
    R_vec = horizontal_step_min_sum(H, Q_loop)

    # Compare
    err = np.max(np.abs(R_loop - R_vec))
    print("Self-test: max abs difference between loop R and vector R =", err)
    assert err < 1e-12, "Vectorised horizontal_step does not match loop reference!"

    print("Self-test passed.")


# Entrypoint for script execution
if __name__ == "__main__":
    # Optional quick check (uncomment to run small verification)
    # _self_test_small()

    # Simulation parameters
    n = 100
    col_w = 3
    row_w = 6
    ebn0_dB_range = np.arange(0.0, 6.1, 1.0)  # 0..6 dB
    frames_per_snr = 2000   # lower for a quick run; increase for production
    max_iter = 50
    seed = 12345

    ebn0s, bers, avg_iters = simulate_min_sum_ber(n=n,
                                                  col_w=col_w,
                                                  row_w=row_w,
                                                  ebn0_dB_range=ebn0_dB_range,
                                                  frames_per_snr=frames_per_snr,
                                                  max_iter=max_iter,
                                                  seed=seed,
                                                  verbose=True)

    # Print numeric table
    print("\nEb/N0 (dB)    BER           Avg Iters")
    for e, b, it in zip(ebn0s, bers, avg_iters):
        print(f"{e:6.2f}      {b:.3e}      {it:.2f}")