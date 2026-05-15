import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import erfc

# Step 1️⃣: Build a regular LDPC parity-check matrix
# LDPC  Low-Density Parity-Check codes.
# They are linear error-correcting codes defined by a sparse parity-check matrix H of size m x n:
#   - n  number of codeword bits (variable nodes)
#   - m  number of parity-check equations (check nodes)
# Each column of H corresponds to a bit, each row to a parity-check.
# Sparsity is key: few 1’s per row/column → efficient decoding.
def build_regular_h(n, col_w, row_w, max_attempts=1000, rng=None):
    if rng is None:
        rng = np.random
    total_ones = n * col_w
    if total_ones % row_w != 0:
        raise ValueError("n*col_w must be divisible by row_w")
    m = total_ones // row_w

    for attempt in range(max_attempts):
        col_stubs = np.repeat(np.arange(n), col_w)
        row_stubs = np.repeat(np.arange(m), row_w)
        rng.shuffle(row_stubs)
        pairs = np.vstack((row_stubs, col_stubs)).T

        # check duplicates
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            continue

        H = np.zeros((m, n), dtype=int)
        for r, c in pairs:
            H[r, c] = 1

        if np.all(H.sum(axis=0) == col_w) and np.all(H.sum(axis=1) == row_w):
            return H, m
    raise RuntimeError("Failed to build H")

# Step 2️⃣: Hard decision from LLRs
# At the receiver, the log-likelihood ratio (LLR) of each received bit is computed:
# Positive → likely 0
# Negative → likely 1
def hard_decision(q_hat):
    return (q_hat < 0).astype(int)

# Step 3️⃣: Compute syndrome
# Compute binary syndrome vector s  H * decoded^T (mod 2)
# If s  0 → all parity-checks satisfied → decoder converged
# Otherwise, iterate up to max_iter
def syndrome(H, c):
    return H.dot(c) % 2

# Step 4️⃣: Min-Sum decoding
# Min-Sum is an iterative message-passing algorithm on the bipartite graph of H
# Horizontal step (check nodes → variable nodes):
#   - Each check node i sends a message R[i,j] to each connected variable node j
#   - Take the sign product of all incoming messages except j
#   - Take the minimum magnitude of all incoming messages except j
#   - Send this as the outgoing message
# Vertical step (variable nodes → check nodes):
#   - Each variable node j updates its message to check i
#   - Sum the channel LLR + all incoming messages from other checks (excluding i)
#   - Send updated belief to each connected check node
# Posterior LLR:
#   - Compute the posterior LLR for each variable node: q_hat  q + sum(R)
#   - Make a hard decision: q_hat < 0 → 1, q_hat > 0 → 0
def min_sum_decode(H, y, sigma2, max_iter=50):
    m, n = H.shape
    q = 2 * y / sigma2  # Step 2: Channel LLRs

    # Initialize messages: Q  variable-to-check messages
    Q = np.zeros((m, n))
    Q[H == 1] = q[np.where(H == 1)[1]]

    decoded = np.zeros(n, dtype=int)
    converged = False
    it_used = 0

    for it in range(1, max_iter + 1):
        # Horizontal step per check node
        R = np.zeros((m, n))
        for i in range(m):
            idx_vars = np.flatnonzero(H[i, :])
            if len(idx_vars) == 1:
                R[i, idx_vars[0]] = 0.0
            else:
                vals = Q[i, idx_vars]
                signs = np.sign(vals)
                mags = np.abs(vals)
                total_sign = np.prod(signs)
                pref = np.minimum.accumulate(mags)
                suff = np.minimum.accumulate(mags[::-1])[::-1]
                min_except = np.zeros_like(mags)
                for k in range(len(mags)):
                    if k == 0:
                        min_except[k] = suff[k+1]
                    elif k == len(mags)-1:
                        min_except[k] = pref[k-1]
                    else:
                        min_except[k] = min(pref[k-1], suff[k+1])
                prod_except = total_sign / signs
                R[i, idx_vars] = prod_except * min_except

        # Vertical step per variable node
        Q_new = np.zeros((m, n))
        for j in range(n):
            idx_checks = np.flatnonzero(H[:, j])
            for i in idx_checks:
                other_checks = idx_checks[idx_checks != i]
                Q_new[i, j] = q[j] + np.sum(R[other_checks, j])
        Q = Q_new

        # Compute posterior LLRs and hard decision
        q_hat = q + np.sum(R, axis=0)
        decoded = hard_decision(q_hat)

        # Check convergence via syndrome
        if np.all(syndrome(H, decoded) == 0):
            converged = True
            it_used = it
            break
        it_used = it

    return decoded, it_used, converged

# Step 5️⃣: Monte Carlo simulation
# Repeat transmission & decoding for multiple codewords (frames) at each Eb/N0
# Compute BER  total bit errors / total bits transmitted
# Track average iterations used and elapsed time per Eb/N0
def simulate_min_sum_ber(n=100, col_w=3, row_w=6):
    ebn0_dB = np.arange(0, 11, 1)  # 0..10 dB
    frames = [10, 100, 1000, 1000, 10000, 10000, 20000, 50000, 100000, 100000, 200000]  # Frames per Eb/N0

    H, m = build_regular_h(n, col_w, row_w)
    k = n - m
    Rc = k / n
    ber = np.zeros_like(ebn0_dB, dtype=float)
    avg_iters = np.zeros_like(ebn0_dB, dtype=float)

    c = np.zeros(n, dtype=int)
    x0 = 1 - 2*c  # all +1

    for idx, eb in enumerate(ebn0_dB):
        sigma2 = 1 / (2 * Rc * 10**(eb/10))
        sigma = np.sqrt(sigma2)
        bit_errors = 0
        total_iters = 0
        total_bits = frames[idx] * n

        t0 = time.time()
        for f in range(frames[idx]):
            y = x0 + sigma * np.random.randn(n)
            decoded, it_used, converged = min_sum_decode(H, y, sigma2)
            bit_errors += np.sum(decoded != c)
            total_iters += it_used
        elapsed = time.time() - t0

        ber[idx] = bit_errors / total_bits
        avg_iters[idx] = total_iters / frames[idx]

        print(f'Eb/N0{eb} dB, BER{ber[idx]:.3e}, avg_it{avg_iters[idx]:.2f}, time{elapsed:.2f}s')

    # Plot
    ebn0_lin = 10**(ebn0_dB/10)
    ber_uncoded = 0.5*erfc(np.sqrt(ebn0_lin))
    plt.semilogy(ebn0_dB, ber_uncoded, 'k--', label='Uncoded BPSK')
    plt.semilogy(ebn0_dB, ber, 'bo-', label='LDPC Min-Sum')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    return ebn0_dB, ber, avg_iters

if __name__ == "__main__":
    simulate_min_sum_ber()
