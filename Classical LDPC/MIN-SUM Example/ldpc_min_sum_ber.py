"""
LDPC Min-Sum Monte Carlo BER simulation (Python)

- Builds a regular (column weight = 3, row weight = 6) sparse parity-check matrix H
  for n = 100 variable nodes -> m = n*3/6 = 50 check nodes.
- Transmits all-zero codeword over AWGN.
- Performs Min-Sum decoding with nested-loop implementation.
- Repeats for multiple Eb/N0 points and counts bit errors to estimate BER.
- Stops early if the decoder converges (syndrome == 0).
- Limits max iterations (default 50) and records iteration usage.

Author: (adapted for your lab workflow)
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.special import erfc


def build_regular_h(n, col_w, row_w, max_attempts=1000, rng=None):
    """
    Build a binary m x n parity-check matrix H with:
      - each column has weight = col_w
      - each row has weight = row_w
    Returns H (numpy array) and m (number of rows).
    Uses configuration-model pairing of stubs; retries if duplicates appear.
    """
    if rng is None:
        rng = np.random

    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n * col_w must be divisible by row_w")
    m = total_ones // row_w

    for attempt in range(max_attempts):
        # create stubs: column indices repeated col_w times
        col_stubs = np.repeat(np.arange(n), col_w)
        # create row stubs: row indices repeated row_w times
        row_stubs = np.repeat(np.arange(m), row_w)

        # shuffle row stubs and pair with column stubs
        rng.shuffle(row_stubs)
        pairs = np.vstack((row_stubs, col_stubs)).T

        # detect duplicates (same (row,col) pair repeated)
        # build a set of (row, col)
        # if collisions occur, retry
        seen = set()
        collision = False
        for (r, c) in pairs:
            if (r, c) in seen:
                collision = True
                break
            seen.add((r, c))
        if collision:
            continue

        # Build H
        H = np.zeros((m, n), dtype=int)
        for (r, c) in pairs:
            H[r, c] = 1

        # double-check row and column weights
        if not (np.all(H.sum(axis=0) == col_w) and np.all(H.sum(axis=1) == row_w)):
            # something off — retry
            continue

        return H, m

    raise RuntimeError("Failed to build a regular H without collisions after attempts")


def hard_decision_from_llr(q_hat):
    """Return decoded bits 0/1 from LLR vector q_hat (0 if q_hat >= 0, 1 otherwise)."""
    return (q_hat < 0).astype(int)


def syndrome(H, v):
    """Compute binary syndrome vector s = H * v^T (mod 2)."""
    return (H.dot(v) % 2)


def horizontal_step_min_sum(H, Q):
    """
    Min-Sum horizontal step (nested-loop explicit).
    H: m x n parity-check matrix (0/1)
    Q: m x n matrix of variable-to-check messages (only entries where H==1 are used)
    Returns R: m x n check-to-variable messages (only H==1 entries set)
    """
    m, n = H.shape
    R = np.zeros_like(Q)
    for i in range(m):  # for each check node
        for j in range(n):  # for each variable node
            if H[i, j] == 1:
                # gather all other connected Q[i, jj] where jj != j
                others = []
                for jj in range(n):
                    if jj != j and H[i, jj] == 1:
                        others.append(Q[i, jj])
                if len(others) == 0:
                    # no other connected nodes -> message zero
                    R[i, j] = 0.0
                else:
                    # product of signs
                    sgn = 1.0
                    for val in others:
                        sgn *= np.sign(val) if val != 0 else 1.0
                    # minimum magnitude
                    minimum = np.min(np.abs(others))
                    R[i, j] = sgn * minimum
    return R


def vertical_step(H, R, q):
    """
    Vertical step: compute Q_new (variable-to-check messages).
    Q_new[i,j] = q[j] + sum_{i' in M_j \ i} R[i', j]
    """
    m, n = H.shape
    Q_new = np.zeros((m, n))
    for j in range(n):  # variable nodes
        for i in range(m):  # check nodes
            if H[i, j] == 1:
                sum_others = 0.0
                for ip in range(m):
                    if ip != i and H[ip, j] == 1:
                        sum_others += R[ip, j]
                Q_new[i, j] = q[j] + sum_others
    return Q_new


def compute_q_hat(H, R, q):
    """
    Final LLR update: q_hat[j] = q[j] + sum_{i in M_j} R[i, j]
    Uses nested loops for clarity.
    """
    m, n = H.shape
    q_hat = np.zeros_like(q)
    for j in range(n):
        total = 0.0
        for i in range(m):
            if H[i, j] == 1:
                total += R[i, j]
        q_hat[j] = q[j] + total
    return q_hat


def min_sum_decode_single(H, y, sigma2, max_iter=50):
    """
    Decode one received vector y using Min-Sum algorithm.
    Returns decoded_bits, num_iters_used, converged_flag, q_hat_final, R_first (first iteration R if needed)
    - H: parity-check matrix (m x n)
    - y: received signal samples (length n, BPSK: +1/-1 plus noise)
    - sigma2: noise variance
    """

    m, n = H.shape
    # Channel LLRs (probabilistic): q_j = 2*y_j / sigma^2
    q = (2.0 * y) / sigma2

    # Initialize Q: variable-to-check messages
    Q = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                Q[i, j] = q[j]

    R_first = None
    converged = False
    decoded = np.zeros(n, dtype=int)
    it_used = 0

    for it in range(1, max_iter + 1):
        # Horizontal step
        R = horizontal_step_min_sum(H, Q)
        if it == 1:
            R_first = R.copy()

        # Vertical step
        Q = vertical_step(H, R, q)

        # Final LLRs and hard decision
        q_hat = compute_q_hat(H, R, q)
        decoded = hard_decision_from_llr(q_hat)

        # Check syndrome
        s = syndrome(H, decoded)
        if np.all(s == 0):
            converged = True
            it_used = it
            break
        it_used = it

    # If didn't converge, q_hat from last iteration is returned
    return decoded, it_used, converged, q_hat, R_first


def simulate_min_sum_ber(n=100,
                         col_w=3,
                         row_w=6,
                         ebn0_dB_range=np.arange(0.0, 6.1, 0.5),
                         frames_per_snr=2000,
                         max_iter=50,
                         use_random_noise=True,
                         seed=None,
                         verbose=True):
    """
    Monte-Carlo simulation of Min-Sum LDPC decoder BER curve.

    Parameters:
      - n: code length (bits)
      - col_w: column weight (ones per column)
      - row_w: row weight (ones per row)
      - ebn0_dB_range: array-like Eb/N0 values in dB
      - frames_per_snr: number of codewords (frames) to simulate per Eb/N0
      - max_iter: maximum decoder iterations (observe improvement between iterations)
      - use_random_noise: if True sample new noise each frame, else use deterministic sequences
      - seed: RNG seed (None for truly random)
      - verbose: print progress

    Returns:
      - ebn0_dB_range (np.array)
      - ber (np.array) bit error rate for each Eb/N0
      - avg_iters (np.array) average iterations used (only for frames that decoded converged or overall)
    """
    rng = np.random.RandomState(seed)

    # Build H
    H, m = build_regular_h(n, col_w, row_w, rng=rng)
    k = n - m
    Rc = float(k) / n

    # Pre-allocate results
    ebn0_dB_range = np.array(ebn0_dB_range, dtype=float)
    ber = np.zeros_like(ebn0_dB_range, dtype=float)
    avg_iters = np.zeros_like(ebn0_dB_range, dtype=float)

    start_time = time.time()

    # All-zero codeword (length n)
    c = np.zeros(n, dtype=int)
    x0 = 1 - 2 * c  # all ones (+1 for 0)

    for idx, ebn0_db in enumerate(ebn0_dB_range):
        EbN0_lin = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)  # per-bit noise variance (matches MATLAB)
        sigma = np.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0
        total_bits = frames_per_snr * n

        if verbose:
            print(f"\nEb/N0 = {ebn0_db:.2f} dB, sigma^2={sigma2:.5e}, frames={frames_per_snr}")

        for f in range(frames_per_snr):
            # Generate noise and received y
            if use_random_noise:
                noise = sigma * rng.randn(n)
            else:
                # deterministic noise (for reproducibility while not using seed)
                noise = sigma * np.random.randn(n)
            y = x0 + noise  # transmit all-zero codeword in BPSK: +1 + noise

            # Decode
            decoded, it_used, converged, q_hat, R_first = min_sum_decode_single(H, y, sigma2, max_iter=max_iter)

            # Count bit errors (compare decoded codeword to all-zero transmitted)
            bit_errors += np.sum(decoded != c)
            total_iters += it_used

            # optional: record per iteration improvements (not implemented per-frame to save time)

            # Progress feedback occasionally
            if verbose and (f + 1) % (max(1, frames_per_snr // 10)) == 0:
                print(f"  frame {f+1}/{frames_per_snr}  current BER estimate = {bit_errors / ((f+1)*n):.3e}")

        ber[idx] = bit_errors / float(total_bits)
        avg_iters[idx] = float(total_iters) / frames_per_snr

        if verbose:
            elapsed = time.time() - start_time
            print(f"-> Eb/N0 {ebn0_db:.2f} dB: BER = {ber[idx]:.3e}, avg iters = {avg_iters[idx]:.2f}, elapsed={elapsed:.1f}s")

    # Plot BER curve (uncoded vs coded approximation)
    plt.figure(figsize=(8, 5))
    ebn0_lin = 10.0 ** (ebn0_dB_range / 10.0)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))  # erfc/2 ~ (1-erf)/2; same general shape
    plt.semilogy(ebn0_dB_range, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')
    plt.semilogy(ebn0_dB_range, ber, 'o-', label=f'LDPC Min-Sum (n={n}, rate={Rc:.2f})')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0 - Min-Sum LDPC')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return ebn0_dB_range, ber, avg_iters


if __name__ == "__main__":
    # Simulation parameters (you can edit these)
    n = 100
    col_w = 3
    row_w = 6
    ebn0_dB_range = np.arange(0.0, 6.1, 1.0)  # 0,1,...,6 dB
    frames_per_snr = 2000   # lower for quick test; increase for accuracy (e.g. 10000)
    max_iter = 50
    seed = 12345
    use_random_noise = True

    ebn0s, bers, avg_iters = simulate_min_sum_ber(n=n,
                                                  col_w=col_w,
                                                  row_w=row_w,
                                                  ebn0_dB_range=ebn0_dB_range,
                                                  frames_per_snr=frames_per_snr,
                                                  max_iter=max_iter,
                                                  use_random_noise=use_random_noise,
                                                  seed=seed,
                                                  verbose=True)

    # Print numeric table
    print("\nEb/N0 (dB)    BER           Avg Iters")
    for e, b, it in zip(ebn0s, bers, avg_iters):
        print(f"{e:6.2f}      {b:.3e}      {it:.2f}")
