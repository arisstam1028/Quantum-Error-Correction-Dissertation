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

# ============================ decoder with iteration BER ============================
def decode_single_with_iter_ber(y, sigma2, c, max_iter=50, alpha=0.8):
    """
    Returns an array iter_bit_errors[iter] = number of bit errors after that iteration.
    """
    q = (2.0 * y) / sigma2
    Q = np.where(E, q[None, :], 0.0)

    iter_bit_errors = np.zeros(max_iter, dtype=int)

    for it in range(max_iter):
        R = horizontal_step_nms(Q, alpha)
        Q = vertical_step(R, q)
        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(np.int8)

        # since c is all-zeros, this equals number of 1s, but this is the correct definition:
        iter_bit_errors[it] = int(np.sum(decoded != c))

        # early success
        if np.all(H @ decoded % 2 == 0):
            iter_bit_errors[it + 1:] = iter_bit_errors[it]
            break

    return iter_bit_errors

# ============================ main simulation ============================
def simulate_ldpc_and_iter_curve(
    snr_dB_range,
    max_frames,
    error_limit,
    min_frames,
    alpha=0.8,
    max_iter=50,
    iter_ber_snr_db=2.0,
    seed=12345,
    verbose=True
):
    rng = np.random.default_rng(seed)
    k = n - m
    Rc = k / n

    ber = np.zeros(len(snr_dB_range))
    used_frames = np.zeros(len(snr_dB_range), dtype=int)
    used_errors = np.zeros(len(snr_dB_range), dtype=int)

    iter_ber_sum = np.zeros(max_iter, dtype=float)
    iter_frames = 0

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
            print(f"\nEb/N0 = {snr_db:.2f} dB | max_frames={max_frames[idx]} | "
                  f"min_frames={min_frames[idx]} | error_limit={error_limit[idx]} | "
                  f"alpha={alpha} | max_iter={max_iter}")

        for f in range(max_frames[idx]):
            noise = sigma * rng.standard_normal(n)
            y = x0 + noise

            # Track iteration-curve only at one chosen SNR
            if abs(snr_db - iter_ber_snr_db) < 1e-12:
                errs_iter = decode_single_with_iter_ber(y, sigma2, c, max_iter, alpha)
                iter_ber_sum += errs_iter
                iter_frames += 1
                bit_errors += int(errs_iter[-1])  # still count BER at that SNR too
            else:
                errs_iter = decode_single_with_iter_ber(y, sigma2, c, max_iter, alpha)
                bit_errors += int(errs_iter[-1])

            frames_done += 1

            # progress print ~10 times
            if verbose:
                step = max(1, max_frames[idx] // 10)
                if frames_done % step == 0 or frames_done == 1:
                    cur_ber = bit_errors / float(frames_done * n)
                    print(f"  frame {frames_done}/{max_frames[idx]}  current BER = {cur_ber:.3e}")

            # early stop rule (only after min_frames)
            if frames_done >= min_frames[idx] and bit_errors >= error_limit[idx]:
                if verbose:
                    print(f"  Stopped early at frame {frames_done}: reached {bit_errors} bit errors.")
                break

        used_frames[idx] = frames_done
        used_errors[idx] = bit_errors
        ber[idx] = bit_errors / float(frames_done * n)

        if verbose:
            elapsed = time.time() - t0
            print(f"→ BER={ber[idx]:.3e} | frames used={frames_done} | bit errors={bit_errors} | elapsed={elapsed:.1f}s")

    # final iteration BER curve (averaged over frames at that SNR)
    if iter_frames > 0:
        iter_ber = iter_ber_sum / float(iter_frames * n)
        if verbose:
            print(f"\nIteration-BER curve collected at Eb/N0={iter_ber_snr_db:.2f} dB using {iter_frames} frames.")
    else:
        iter_ber = np.full(max_iter, np.nan)
        if verbose:
            print("\nWARNING: No frames matched iter_ber_snr_db. Iteration curve is NaN.")

    return ber, iter_ber, used_frames, used_errors

# ============================ entrypoint ============================
if __name__ == "__main__":

    snr_dB = np.arange(0, 3.3, 0.3)

    max_frames = [10, 100, 100, 100, 100, 1000, 1000, 10000, 50000, 100000, 100000]
    error_limit = [100] * 11
    #error_limit = [200, 200, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
    min_frames = [10, 20, 20, 30, 50, 80, 100, 150, 200, 200, 200]

    alpha = 0.8
    max_iter = 50
    iter_ber_snr_db = 2.1   # <<< iteration curve SNR

    ber, iter_ber, used_frames, used_errors = simulate_ldpc_and_iter_curve(
        snr_dB_range=snr_dB,
        max_frames=max_frames,
        error_limit=error_limit,
        min_frames=min_frames,
        alpha=alpha,
        max_iter=max_iter,
        iter_ber_snr_db=iter_ber_snr_db,
        seed=12345,
        verbose=True
    )

    # Uncoded BPSK reference
    ebn0_lin = 10 ** (snr_dB / 10)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_dB, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')
    plt.semilogy(snr_dB, ber, 'o-', label='LDPC BER')

    # Put iteration curve on same axes (mapped across x-range for display)
    plt.semilogy(
        np.linspace(snr_dB[0], snr_dB[-1], max_iter),
        iter_ber,
        'r-.',
        label=f'BER vs Iteration @ {iter_ber_snr_db:.2f} dB'
    )

    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nEb/N0(dB)  UsedFrames  BitErrors   BER")
    for i in range(len(snr_dB)):
        print(f"{snr_dB[i]:7.2f}  {used_frames[i]:9d}  {used_errors[i]:9d}  {ber[i]:.3e}")
