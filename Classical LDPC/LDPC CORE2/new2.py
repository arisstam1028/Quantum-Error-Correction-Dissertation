"""
Optimized LDPC Min-Sum Monte Carlo BER simulation (vectorised, fast)
NOW WITH: Normalized Min-Sum (α-Min-Sum)

Updates included (per your requests):
 - Uses a FIXED parity-check matrix H loaded from ldpc_H_matrix.py (same H for ALL SNRs, frames, and iterations).
 - Uses an EXPLICIT frames-per-SNR LIST (frames_per_snr) that you provide.
 - Uses Eb/N0 range 0..10 dB (11 points) matched to your frames list.
 - Fixes SyntaxWarning by using raw docstrings where backslashes appear.
 - Fixes Min-Sum sign edge-case: treats sign(0) as +1.
 - Adds α normalization in the check-node update:
       R  α * (sign product) * min(|Q|)
   Typical α for (3,6) LDPC is ~0.75..0.9. Start with α0.8.

Requirements:
 - ldpc_H_matrix.py must define H (numpy array) of shape (500, 1000) with col weight3 and row weight6.
"""

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

# Load your fixed LDPC H matrix ONCE (used for all iterations / frames / SNRs)
from ldpc_H_matrix import H  # must be a numpy array, shape (500, 1000)

# Optional but strongly recommended sanity checks
assert H.shape == (500, 1000), f"Expected H shape (500,1000), got {H.shape}"
assert np.all(H.sum(axis=0) == 3), "Column weights are not all 3"
assert np.all(H.sum(axis=1) == 6), "Row weights are not all 6"


# Helper: sign with 0 mapped to +1 (prevents sign-product collapsing to 0)
def nzsign(x):
    """Non-zero sign: returns +1 for x>0, -1 for x<0 (so sign(0)+1)."""
    return np.where(x < 0, -1.0, 1.0)


# Vectorised horizontal step: compute R (check-to-variable messages) with α
def horizontal_step_min_sum(H, Q, alpha=0.8):
    r"""
    Vectorised Normalized Min-Sum (α-Min-Sum) horizontal step.

    For check node i and variable node j in N(i):
      R[i,j]  α * prod_{j' in N(i)\{j}} sign(Q[i,j']) * min_{j' in N(i)\{j}} |Q[i,j']|

    alpha (0<alpha<1) scales magnitudes to compensate Min-Sum overestimation.
    """

    E = (H == 1)
    m, n = H.shape

    degs = E.sum(axis=1)

    # Magnitudes: non-edges are +inf so they don't affect minima
    mags_masked = np.where(E, np.abs(Q), np.inf)

    # First min and its position
    min1 = np.min(mags_masked, axis=1)
    pos1 = np.argmin(mags_masked, axis=1)

    # Second min
    mags2 = mags_masked.copy()
    rows_idx = np.arange(m)
    mags2[rows_idx, pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    # Signs: use nzsign so zeros behave as +1
    sign_masked = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(sign_masked, axis=1)

    R_sign = total_sign[:, None] * np.where(E, nzsign(Q), 1.0)

    is_first_min = (np.arange(n)[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first_min, min2[:, None], min1[:, None])

    # Rows with deg < 1: no meaningful outgoing messages
    low_deg_rows = (degs <= 1)
    if np.any(low_deg_rows):
        R_mag[low_deg_rows, :] = 0.0
        R_sign[low_deg_rows, :] = 0.0

    #  α normalization (key change) 
    R = R_sign * (alpha * R_mag)
    R[~E] = 0.0
    return R


# Vectorised vertical step: compute Q_new (variable-to-check messages)
def vertical_step(H, R, q):
    r"""
    Vectorised vertical step.

    Q_new[i,j]  q[j] + sum_{i' in M(j) \ {i}} R[i', j]
               q[j] + (sum over column j) - R[i,j]
    """

    E = (H == 1)
    total_R_col = R.sum(axis=0)
    Q_new = q[None, :] + total_R_col[None, :] - R
    Q_new[~E] = 0.0
    return Q_new


# Vectorised final LLR computation
def compute_q_hat(R, q):
    """q_hat[j]  q[j] + sum_i R[i,j]"""
    return q + R.sum(axis=0)


# Single frame decode (α-Min-Sum)
def min_sum_decode_single(H, y, sigma2, max_iter=50, alpha=0.8):
    """
    Decode one received vector y using vectorised α-Min-Sum.
    Returns: decoded_bits, it_used, converged_flag
    """
    m, n = H.shape

    q = (2.0 * y) / sigma2
    Q = np.where(H, q[None, :], 0.0)

    for it in range(1, max_iter + 1):
        R = horizontal_step_min_sum(H, Q, alpha=alpha)
        Q = vertical_step(H, R, q)

        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(int)

        s = (H.dot(decoded) % 2)
        if np.all(s == 0):
            return decoded, it, True

    return decoded, max_iter, False


# Monte-Carlo simulation harness using FIXED H, frames LIST, and α
def simulate_min_sum_ber(H,
                         ebn0_dB_range,
                         frames_per_snr,
                         max_iter=50,
                         alpha=0.8,
                         seed=None,
                         verbose=True):
    """
    Vectorised Monte-Carlo α-Min-Sum simulation using:
      - fixed parity-check matrix H (provided)
      - explicit frames_per_snr list/array (same length as ebn0_dB_range)
      - alpha normalization parameter
    """

    rng = np.random.RandomState(seed)

    ebn0_dB_range = np.asarray(ebn0_dB_range, dtype=float)
    frames_per_snr = np.asarray(frames_per_snr, dtype=int)

    if len(frames_per_snr) != len(ebn0_dB_range):
        raise ValueError("frames_per_snr must have the same length as ebn0_dB_range")

    m, n = H.shape
    k = n - m
    Rc = float(k) / n

    ber = np.zeros_like(ebn0_dB_range, dtype=float)
    avg_iters = np.zeros_like(ebn0_dB_range, dtype=float)

    c = np.zeros(n, dtype=int)
    x0 = 1 - 2 * c

    start_time = time.time()

    for idx, ebn0_db in enumerate(ebn0_dB_range):
        frames_this = int(frames_per_snr[idx])

        EbN0_lin = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        sigma = math.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0
        total_bits = frames_this * n

        if verbose:
            print(f'\nEb/N0  {ebn0_db:.2f} dB, sigma^2{sigma2:.5e}, frames{frames_this}, alpha{alpha}')

        for f in range(frames_this):
            noise = sigma * rng.randn(n)
            y = x0 + noise

            decoded, it_used, converged = min_sum_decode_single(
                H, y, sigma2, max_iter=max_iter, alpha=alpha
            )

            bit_errors += np.sum(decoded != c)
            total_iters += it_used

            if verbose and (f + 1) % max(1, frames_this // 10) == 0:
                print(f'  frame {f + 1}/{frames_this}  current BER estimate  {bit_errors / ((f + 1) * n):.3e}')

        ber[idx] = bit_errors / float(total_bits)
        avg_iters[idx] = float(total_iters) / frames_this

        if verbose:
            elapsed = time.time() - start_time
            print(f'-> Eb/N0 {ebn0_db:.2f} dB: BER  {ber[idx]:.3e}, avg iters  {avg_iters[idx]:.2f}, elapsed{elapsed:.1f}s')

    # Plot BER curve
    plt.figure(figsize=(8, 5))
    ebn0_lin = 10.0 ** (ebn0_dB_range / 10.0)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))
    plt.semilogy(ebn0_dB_range, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')
    plt.semilogy(ebn0_dB_range, ber, 'o-', label=f'LDPC α-Min-Sum (H={m}x{n}, rate={Rc:.2f}, α={alpha:.2f})')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0 - Normalized Min-Sum LDPC (vectorised, fixed H)')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return ebn0_dB_range, ber, avg_iters


# Entrypoint
if __name__ == "__main__":

    # Eb/N0 points: 0..10 dB (11 points)
    ebn0_dB_range = np.arange(0.0, 11.0, 1.0)

    # Your explicit frames list (must have length 11)
    frames = [10, 100, 1000, 1000, 10000, 10000, 20000, 50000, 100000, 100000, 100000]

    # Choose α (try 0.75, 0.8, 0.85, 0.9)
    alpha = 0.80

    ebn0s, bers, avg_iters = simulate_min_sum_ber(
        H=H,
        ebn0_dB_range=ebn0_dB_range,
        frames_per_snr=frames,
        max_iter=50,
        alpha=alpha,
        seed=12345,
        verbose=True
    )

    print("\nEb/N0 (dB)    Frames      BER           Avg Iters")
    for e, f, b, it in zip(ebn0s, frames, bers, avg_iters):
        print(f"{e:6.2f}     {f:7d}    {b:.3e}      {it:.2f}")
