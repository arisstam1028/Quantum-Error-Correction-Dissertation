# ============================ imports ============================
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

from ldpc_H_matrix import H

# ============================ sanity checks ============================
assert H.shape == (500, 1000)
assert np.all(H.sum(axis=0) == 3)
assert np.all(H.sum(axis=1) == 6)

E = (H == 1)
m, n = H.shape
degs_row = E.sum(axis=1)
rows_idx = np.arange(m)
cols = np.arange(n)

# ============================ helpers ============================
def nzsign(x):
    return np.where(x < 0, -1.0, 1.0)

# ============================ horizontal steps ============================
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
    R_mag[low_deg, :] = 0.0
    R_sign[low_deg, :] = 0.0

    R = R_sign * (alpha * R_mag)
    R[~E] = 0.0
    return R

# ============================ vertical step ============================
def vertical_step(R, q):
    total = R.sum(axis=0)
    Q = q[None, :] + total[None, :] - R
    Q[~E] = 0.0
    return Q

def compute_q_hat(R, q):
    return q + R.sum(axis=0)

# ============================ decoder with per-iteration BER ============================
def decode_single_with_iter_errors(y, sigma2, c, max_iter=50, alpha=0.8):
    """
    Returns:
      iter_bit_errors[it] = number of bit errors after iteration (it+1)
      final_decoded = decoded bits at the stopping iteration (or max_iter)
      it_used = iterations used
      converged = True/False (syndrome is zero)
    """
    q = (2.0 * y) / sigma2
    Q = np.where(E, q[None, :], 0.0)

    iter_bit_errors = np.zeros(max_iter, dtype=int)

    for it in range(max_iter):
        R = horizontal_step_nms(Q, alpha)
        Q = vertical_step(R, q)
        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(np.int8)

        iter_bit_errors[it] = int(np.sum(decoded != c))

        # early success (syndrome = 0)
        if np.all(H @ decoded % 2 == 0):
            iter_bit_errors[it + 1:] = iter_bit_errors[it]
            return iter_bit_errors, decoded, (it + 1), True

    return iter_bit_errors, decoded, max_iter, False

# ============================ main simulation ============================
def simulate_ldpc_with_iter_curves(
    snr_dB_range,
    max_frames,
    error_limit,
    min_frames,
    alpha=0.8,
    max_iter=50,
    seed=12345,
    verbose=True
):
    """
    For each SNR point:
      - runs Monte Carlo with early stop rule (min_frames + error_limit OR max_frames)
      - returns:
          ber[idx]               = final BER estimate at that SNR
          iter_ber[idx, it]      = average BER after iteration (it+1) at that SNR
          used_frames[idx]       = frames actually simulated
          used_errors[idx]       = bit errors counted (final decision)
    """
    rng = np.random.default_rng(seed)
    k = n - m
    Rc = k / n

    snr_dB_range = np.asarray(snr_dB_range, dtype=float)
    L = len(snr_dB_range)

    ber = np.zeros(L, dtype=float)
    used_frames = np.zeros(L, dtype=int)
    used_errors = np.zeros(L, dtype=int)

    # accumulate per-iteration errors across frames per SNR
    iter_err_sum = np.zeros((L, max_iter), dtype=np.float64)

    c = np.zeros(n, dtype=np.int8)
    x0 = 1 - 2 * c

    t0 = time.time()

    for idx, snr_db in enumerate(snr_dB_range):
        EbN0 = 10 ** (snr_db / 10)
        sigma2 = 1 / (2 * Rc * EbN0)
        sigma = math.sqrt(sigma2)

        bit_errors = 0
        frames_done = 0

        if verbose:
            print(
                f"\nEb/N0 = {snr_db:.2f} dB | max_frames={max_frames[idx]} | "
                f"min_frames={min_frames[idx]} | error_limit={error_limit[idx]} | "
                f"alpha={alpha} | max_iter={max_iter}"
            )

        step = max(1, int(max_frames[idx]) // 10)

        for _ in range(int(max_frames[idx])):
            noise = sigma * rng.standard_normal(n)
            y = x0 + noise

            iter_bit_errs, decoded, it_used, _ = decode_single_with_iter_errors(
                y=y, sigma2=sigma2, c=c, max_iter=max_iter, alpha=alpha
            )

            # accumulate per-iteration errors (for the iteration-curves plot)
            iter_err_sum[idx, :] += iter_bit_errs

            # accumulate final BER stats
            err_final = int(np.sum(decoded != c))
            bit_errors += err_final
            frames_done += 1

            # progress printing ~10 times
            if verbose and (frames_done == 1 or frames_done % step == 0):
                cur_ber = bit_errors / float(frames_done * n)
                print(f"  frame {frames_done}/{max_frames[idx]}  current BER = {cur_ber:.3e}")

            # early stop rule (only after min_frames)
            if frames_done >= int(min_frames[idx]) and bit_errors >= int(error_limit[idx]):
                if verbose:
                    print(f"  Stopped early at frame {frames_done}: reached {bit_errors} bit errors.")
                break

        used_frames[idx] = frames_done
        used_errors[idx] = bit_errors
        ber[idx] = bit_errors / float(frames_done * n)

        if verbose:
            elapsed = time.time() - t0
            print(f"→ BER={ber[idx]:.3e} | frames used={frames_done} | bit errors={bit_errors} | elapsed={elapsed:.1f}s")

    # Convert per-iteration error sums into per-iteration BER curves
    iter_ber = np.zeros_like(iter_err_sum, dtype=np.float64)
    for idx in range(L):
        if used_frames[idx] > 0:
            iter_ber[idx, :] = iter_err_sum[idx, :] / float(used_frames[idx] * n)
        else:
            iter_ber[idx, :] = np.nan

    return ber, iter_ber, used_frames, used_errors

# ============================ plotting ============================
def plot_ber_curve(snr_dB, ber):
    snr_dB = np.asarray(snr_dB, dtype=float)
    ebn0_lin = 10 ** (snr_dB / 10)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_dB, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')
    plt.semilogy(snr_dB, ber, 'o-', label='LDPC BER')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('LDPC BER vs Eb/N0')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_iter_curves(iter_ber, snr_dB, max_iter, plot_every=2):
    """
    iter_ber shape: (len(snr_dB), max_iter)
    plot_every: plot only every k-th SNR curve to reduce clutter (set 1 for all)
    """
    snr_dB = np.asarray(snr_dB, dtype=float)
    it_axis = np.arange(1, max_iter + 1)

    plt.figure(figsize=(8, 5))
    for idx in range(0, len(snr_dB), int(plot_every)):
        plt.semilogy(it_axis, iter_ber[idx, :], label=f'{snr_dB[idx]:.2f} dB')

    plt.xlabel('Iteration')
    plt.ylabel('BER after that iteration')
    plt.title('BER improvement vs iteration (one curve per SNR)')
    plt.grid(True, which='both', linestyle=':')

    # Legend can get huge if you plot every SNR; still works, but you can reduce clutter with plot_every > 1
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

# ============================ entrypoint ============================
if __name__ == "__main__":

    # Same meaning as MATLAB: Eb/N0(dB)
    snr_dB = np.arange(0.0, 3.3, 0.3)

    max_frames = [10, 100, 100, 100, 100, 1000, 1000, 10000, 50000, 100000, 100000]
    error_limit = [100] * len(snr_dB)
    min_frames = [10, 20, 20, 30, 50, 80, 100, 150, 200, 200, 200]

    alpha = 0.8
    max_iter = 50

    ber, iter_ber, used_frames, used_errors = simulate_ldpc_with_iter_curves(
        snr_dB_range=snr_dB,
        max_frames=max_frames,
        error_limit=error_limit,
        min_frames=min_frames,
        alpha=alpha,
        max_iter=max_iter,
        seed=12345,
        verbose=True
    )

    # --- Plot 1: BER vs Eb/N0 ---
    plot_ber_curve(snr_dB, ber)

    # --- Plot 2: BER vs iteration for each SNR ---
    # If the legend is too crowded, change plot_every to 2 or 3.
    plot_iter_curves(iter_ber, snr_dB, max_iter=max_iter, plot_every=1)

    print("\nEb/N0(dB)  UsedFrames  BitErrors   BER")
    for i in range(len(snr_dB)):
        print(f"{snr_dB[i]:7.2f}  {used_frames[i]:9d}  {used_errors[i]:9d}  {ber[i]:.3e}")
