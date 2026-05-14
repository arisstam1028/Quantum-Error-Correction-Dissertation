# ================================================================
# LDPC: BER vs Iteration curves for MULTIPLE Eb/N0 values (same axes)
# - Uses your fixed H from ldpc_H_matrix.py (500x1000, col_w=3, row_w=6)
# - Normalized Min-Sum (alpha-Min-Sum)
# - For each Eb/N0 point: run Monte Carlo frames and average BER after each iteration
# - Early stop per Eb/N0 using: (frames_done >= min_frames) AND (final_bit_errors >= error_limit)
# - Prints progress while running
# ================================================================

import numpy as np
import time
import math
import matplotlib.pyplot as plt

from ldpc_H_matrix import H

# ============================ sanity checks ============================
assert H.shape == (500, 1000), f"Expected (500,1000), got {H.shape}"
assert np.all(H.sum(axis=0) == 3), "Column weights not all 3"
assert np.all(H.sum(axis=1) == 6), "Row weights not all 6"

E = (H == 1)
m, n = H.shape
degs_row = E.sum(axis=1)
rows_idx = np.arange(m)
cols = np.arange(n)

# ============================ helpers ============================
def nzsign(x):
    # sign(0)=+1
    return np.where(x < 0, -1.0, 1.0)

# ============================ horizontal step (NMS) ============================
def horizontal_step_nms(Q, alpha=0.8):
    mags = np.where(E, np.abs(Q), np.inf)

    min1 = np.min(mags, axis=1)
    pos1 = np.argmin(mags, axis=1)

    mags2 = mags.copy()
    mags2[rows_idx, pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    signs = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(signs, axis=1)
    R_sign = total_sign[:, None] * signs

    is_first = (cols[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first, min2[:, None], min1[:, None])

    low_deg = (degs_row <= 1)
    if np.any(low_deg):
        R_mag[low_deg, :] = 0.0
        R_sign[low_deg, :] = 0.0

    R = R_sign * (alpha * R_mag)
    R[~E] = 0.0
    return R

# ============================ vertical + final LLR ============================
def vertical_step(R, q):
    total = R.sum(axis=0)
    Q = q[None, :] + total[None, :] - R
    Q[~E] = 0.0
    return Q

def compute_q_hat(R, q):
    return q + R.sum(axis=0)

# ============================ decoder: per-iteration bit errors ============================
def decode_single_with_iter_errors(y, sigma2, c, max_iter=50, alpha=0.8):
    """
    Returns:
      iter_bit_errors[it] = #bit errors after iteration (it+1)
      decoded_final       = final decoded bits (at early stop iteration or max_iter)
      it_used             = iterations used
      converged           = True if syndrome==0 at early stop
    """
    q = (2.0 * y) / sigma2
    Q = np.where(E, q[None, :], 0.0)

    iter_bit_errors = np.zeros(max_iter, dtype=np.int32)

    for it in range(max_iter):
        R = horizontal_step_nms(Q, alpha)
        Q = vertical_step(R, q)
        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(np.int8)

        iter_bit_errors[it] = int(np.sum(decoded != c))

        # syndrome check
        if np.all(H.dot(decoded) % 2 == 0):
            # keep curve flat after convergence
            iter_bit_errors[it + 1:] = iter_bit_errors[it]
            return iter_bit_errors, decoded, (it + 1), True

    return iter_bit_errors, decoded, max_iter, False

# ============================ simulation for one Eb/N0 ============================
def simulate_iteration_curve(
    ebn0_db,
    max_frames,
    min_frames,
    error_limit,
    alpha=0.8,
    max_iter=50,
    seed=12345,
    verbose=True,
):
    """
    Produces iter_ber[it] = average BER after iteration (it+1) for a fixed Eb/N0.
    Early stop uses FINAL decision bit-errors, not per-iteration.
    """
    rng = np.random.default_rng(seed)

    # rate ~ (n-m)/n as in your code
    k = n - m
    Rc = k / n

    EbN0 = 10.0 ** (ebn0_db / 10.0)
    sigma2 = 1.0 / (2.0 * Rc * EbN0)
    sigma = math.sqrt(sigma2)

    c = np.zeros(n, dtype=np.int8)
    x0 = 1 - 2 * c

    iter_err_sum = np.zeros(max_iter, dtype=np.float64)
    frames_done = 0
    bit_errors_final = 0
    iters_sum = 0

    t0 = time.time()
    step = max(1, max_frames // 10)

    if verbose:
        print(
            f"\nEb/N0 = {ebn0_db:.2f} dB | max_frames={max_frames} | min_frames={min_frames} | "
            f"error_limit={error_limit} | alpha={alpha} | max_iter={max_iter}"
        )

    for f in range(max_frames):
        noise = sigma * rng.standard_normal(n)
        y = x0 + noise

        iter_bit_errs, decoded, it_used, _ = decode_single_with_iter_errors(
            y=y, sigma2=sigma2, c=c, max_iter=max_iter, alpha=alpha
        )

        iter_err_sum += iter_bit_errs
        err_final = int(np.sum(decoded != c))
        bit_errors_final += err_final
        iters_sum += it_used
        frames_done += 1

        if verbose and (frames_done == 1 or frames_done % step == 0):
            cur_ber_final = bit_errors_final / float(frames_done * n)
            print(f"  frame {frames_done}/{max_frames}  current FINAL-BER = {cur_ber_final:.3e}")

        if frames_done >= min_frames and bit_errors_final >= error_limit:
            if verbose:
                print(f"  Stopped early at frame {frames_done}: reached {bit_errors_final} final bit errors.")
            break

    if frames_done == 0:
        return np.full(max_iter, np.nan)

    iter_ber = iter_err_sum / float(frames_done * n)
    avg_iters = iters_sum / float(frames_done)

    if verbose:
        elapsed = time.time() - t0
        final_ber = bit_errors_final / float(frames_done * n)
        print(
            f"→ Done Eb/N0={ebn0_db:.2f} dB | frames={frames_done} | final bit errors={bit_errors_final} | "
            f"FINAL BER={final_ber:.3e} | avg iters={avg_iters:.2f} | elapsed={elapsed:.1f}s"
        )

    return iter_ber

# ============================ entrypoint: MULTIPLE Eb/N0 curves ============================
if __name__ == "__main__":

    # Pick the Eb/N0 values you want iteration-curves for (multiple curves on same axes)
    EBNO_LIST = [0.9, 1.5, 2.1, 2.4]   # edit this list

    # Decoder params
    ALPHA = 0.8
    MAX_ITER = 50

    # Monte Carlo controls per Eb/N0 (keep runtime reasonable)
    # You can make these dictionaries if you want per-SNR tuning, but scalar is simplest:
    MAX_FRAMES = 5000
    MIN_FRAMES = 200
    ERROR_LIMIT = 200  # stop once you have this many FINAL bit errors (after MIN_FRAMES)

    # Plot all curves on the same axes
    plt.figure(figsize=(8, 5))
    it_axis = np.arange(1, MAX_ITER + 1)

    for i, ebno in enumerate(EBNO_LIST):
        # Different seed per Eb/N0 so curves aren’t identical due to same RNG stream
        seed = 12345 + i * 1000

        iter_ber = simulate_iteration_curve(
            ebn0_db=float(ebno),
            max_frames=int(MAX_FRAMES),
            min_frames=int(MIN_FRAMES),
            error_limit=int(ERROR_LIMIT),
            alpha=ALPHA,
            max_iter=MAX_ITER,
            seed=seed,
            verbose=True,
        )

        plt.semilogy(it_axis, iter_ber, label=f"{ebno:.2f} dB")

    plt.xlabel("Iteration")
    plt.ylabel("BER after that iteration")
    plt.title("LDPC BER vs Iteration (multiple Eb/N0 curves)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()
