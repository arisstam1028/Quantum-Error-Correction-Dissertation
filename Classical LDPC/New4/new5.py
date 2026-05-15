"""
LDPC (500x1000, rate 1/2) Monte-Carlo simulation with:
1) Error-limit per SNR (so “BER0” is less likely / more meaningful)
2) Minimum-frames per SNR (prevents stopping after 1–2 frames at low SNR)
3) Clean decoder “mode” switching (MS / NMS / OMS)
4) Precomputed edge mask E (avoid recomputing H1 every iteration)
5) FER (Frame Error Rate) in addition to BER

NOTE about your MATLAB repetition-code example:
  snr  [0..10] dB is used as Eb/N0(dB) in the BER formula and sigma expression.
This Python code treats snr_dB_range exactly the same way: it is Eb/N0 in dB.

Requires:
  - ldpc_H_matrix.py defining H (numpy array) shape (500,1000) with col_w3, row_w6
"""

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

from ldpc_H_matrix import H

#  Sanity checks 
assert H.shape == (500, 1000), f"Expected H shape (500,1000), got {H.shape}"
assert np.all(H.sum(axis=0) == 3), "Column weights are not all 3"
assert np.all(H.sum(axis=1) == 6), "Row weights are not all 6"

#  Precompute edges (Change #4) 
E = (H == 1)
m, n = H.shape
degs_row = E.sum(axis=1)
rows_idx = np.arange(m)
cols = np.arange(n)

#  Helpers 
def nzsign(x):
    """Non-zero sign: returns +1 for x>0, -1 for x<0 (so sign(0)+1)."""
    return np.where(x < 0, -1.0, 1.0)

#  Horizontal steps 
def horizontal_step_nms(Q, alpha=0.8):
    """
    Normalized Min-Sum (α-Min-Sum) horizontal step using precomputed E.
    """
    # Magnitudes: non-edges are +inf so they don't affect minima
    mags_masked = np.where(E, np.abs(Q), np.inf)

    # First min and its position per row
    min1 = np.min(mags_masked, axis=1)
    pos1 = np.argmin(mags_masked, axis=1)

    # Second min per row
    mags2 = mags_masked.copy()
    mags2[rows_idx, pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    # Signs (sign(0)+1)
    sign_masked = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(sign_masked, axis=1)

    # Outgoing sign for edge (i,j): total_sign * sign(Q_ij)
    R_sign = total_sign[:, None] * np.where(E, nzsign(Q), 1.0)

    # Outgoing magnitude: min over neighbors excluding j
    is_first_min = (cols[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first_min, min2[:, None], min1[:, None])

    # Rows with deg < 1: no meaningful outgoing messages
    low_deg = (degs_row <= 1)
    if np.any(low_deg):
        R_mag[low_deg, :] = 0.0
        R_sign[low_deg, :] = 0.0

    R = R_sign * (alpha * R_mag)
    R[~E] = 0.0
    return R

def horizontal_step_oms(Q, beta=0.15):
    """
    Offset Min-Sum (OMS) horizontal step:
      R  sign * max(min(|Q|)-beta, 0)
    """
    mags_masked = np.where(E, np.abs(Q), np.inf)

    min1 = np.min(mags_masked, axis=1)
    pos1 = np.argmin(mags_masked, axis=1)

    mags2 = mags_masked.copy()
    mags2[rows_idx, pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    sign_masked = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(sign_masked, axis=1)
    R_sign = total_sign[:, None] * np.where(E, nzsign(Q), 1.0)

    is_first_min = (cols[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first_min, min2[:, None], min1[:, None])

    low_deg = (degs_row <= 1)
    if np.any(low_deg):
        R_mag[low_deg, :] = 0.0
        R_sign[low_deg, :] = 0.0

    R_mag = np.maximum(R_mag - beta, 0.0)
    R = R_sign * R_mag
    R[~E] = 0.0
    return R

#  Vertical + final LLR 
def vertical_step(R, q):
    """
    Q_new[i,j]  q[j] + sum_{i' in M(j)} R[i',j] - R[i,j]
    """
    total_R_col = R.sum(axis=0)
    Q_new = q[None, :] + total_R_col[None, :] - R
    Q_new[~E] = 0.0
    return Q_new

def compute_q_hat(R, q):
    return q + R.sum(axis=0)

#  Decoder (Change #3) 
def decode_single(y, sigma2, max_iter=50, mode="NMS", alpha=0.8, beta=0.15):
    """
    mode:
      "MS"  : Min-Sum (alpha forced to 1.0)
      "NMS" : Normalized Min-Sum (uses alpha)
      "OMS" : Offset Min-Sum (uses beta)
    """
    q = (2.0 * y) / sigma2
    Q = np.where(E, q[None, :], 0.0)

    for it in range(1, max_iter + 1):
        if mode == "MS":
            R = horizontal_step_nms(Q, alpha=1.0)
        elif mode == "NMS":
            R = horizontal_step_nms(Q, alpha=alpha)
        elif mode == "OMS":
            R = horizontal_step_oms(Q, beta=beta)
        else:
            raise ValueError("mode must be one of: MS, NMS, OMS")

        Q = vertical_step(R, q)
        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(np.int8)

        # syndrome check
        if np.all(H.dot(decoded) % 2 == 0):
            return decoded, it, True

    return decoded, max_iter, False

#  Simulation harness (Changes #1, #2, #5) 
def simulate_ldpc(
    snr_dB_range,
    max_frames_per_snr,
    error_limit_per_snr,
    min_frames_per_snr,
    max_iter=50,
    mode="NMS",
    alpha=0.8,
    beta=0.15,
    seed=12345,
    verbose=True,
):
    """
    snr_dB_range is Eb/N0(dB) (same meaning as your MATLAB snr[0..10]).
    Early stop rule:
      stop at SNR point when (frames_used > min_frames) AND (bit_errors > error_limit),
      or when frames_used  max_frames.
    Also computes FER.
    """
    rng = np.random.RandomState(seed)

    snr_dB_range = np.asarray(snr_dB_range, dtype=float)
    L = len(snr_dB_range)

    def _as_vec(x, name):
        x = np.asarray(x)
        if x.size == 1:
            return np.full(L, int(x), dtype=int)
        if x.size != L:
            raise ValueError(f"{name} must be scalar or length {L}")
        return x.astype(int)

    max_frames_per_snr = _as_vec(max_frames_per_snr, "max_frames_per_snr")
    error_limit_per_snr = _as_vec(error_limit_per_snr, "error_limit_per_snr")
    min_frames_per_snr = _as_vec(min_frames_per_snr, "min_frames_per_snr")

    # code rate (assumes full-rank-ish; for your setup kn-m is what you used)
    k = n - m
    Rc = float(k) / n

    ber = np.zeros(L, dtype=float)
    fer = np.zeros(L, dtype=float)
    avg_iters = np.zeros(L, dtype=float)
    used_frames = np.zeros(L, dtype=int)
    used_errors = np.zeros(L, dtype=int)

    c = np.zeros(n, dtype=np.int8)      # all-zero codeword bits
    x0 = 1 - 2 * c                      # BPSK mapping: 0 -> +1

    t0 = time.time()

    for idx, snr_db in enumerate(snr_dB_range):
        max_frames = int(max_frames_per_snr[idx])
        err_limit = int(error_limit_per_snr[idx])
        min_frames = int(min_frames_per_snr[idx])

        EbN0_lin = 10.0 ** (snr_db / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)     # same form as MATLAB: sigma  sqrt(1/(2*Rc*EbN0))
        sigma = math.sqrt(sigma2)

        bit_errors = 0
        frame_errors = 0
        total_iters = 0
        frames_done = 0

        if verbose:
            extra = f"alpha={alpha}" if mode in ("MS", "NMS") else f"beta={beta}"
            print(f'\nEb/N0  {snr_db:.2f} dB | max frames{max_frames} | min_frames{min_frames} | error_limit{err_limit} | mode{mode} | {extra} | max_iter{max_iter}')

        for f in range(max_frames):
            noise = sigma * rng.randn(n)         # same as MATLAB sigma*randn(1,N)
            y = x0 + noise

            decoded, it_used, _ = decode_single(
                y=y, sigma2=sigma2, max_iter=max_iter, mode=mode, alpha=alpha, beta=beta
            )

            err = int(np.sum(decoded != c))
            bit_errors += err
            frame_errors += (1 if err > 0 else 0)
            total_iters += it_used
            frames_done += 1

            # early stop (Change #1 + #2)
            if frames_done >= min_frames and bit_errors >= err_limit:
                if verbose:
                    print(f"  Stopped early at frame {frames_done}: reached {bit_errors} bit errors.")
                break

            if verbose and frames_done % max(1, max_frames // 10) == 0:
                cur_ber = bit_errors / float(frames_done * n)
                print(f'  frame {frames_done}/{max_frames}  current BER{cur_ber:.3e}  frame_errs{frame_errors}')

        used_frames[idx] = frames_done
        used_errors[idx] = bit_errors

        total_bits = frames_done * n
        ber[idx] = bit_errors / float(total_bits) if total_bits > 0 else 0.0
        fer[idx] = frame_errors / float(frames_done) if frames_done > 0 else 0.0
        avg_iters[idx] = total_iters / float(frames_done) if frames_done > 0 else 0.0

        if verbose:
            elapsed = time.time() - t0
            print(f'→ BER{ber[idx]:.3e}, FER{fer[idx]:.3e}, avg iters{avg_iters[idx]:.2f}, frames used{frames_done}, errors{bit_errors}, elapsed{elapsed:.1f}s')

    return snr_dB_range, ber, fer, avg_iters, used_frames, used_errors

#  Plotting 
def plot_curves(snr_dB_range, ber, fer, label_main="LDPC"):
    snr_dB_range = np.asarray(snr_dB_range, dtype=float)
    ebn0_lin = 10.0 ** (snr_dB_range / 10.0)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_dB_range, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')
    plt.semilogy(snr_dB_range, ber, 'o-', label=f'{label_main} BER')
    plt.semilogy(snr_dB_range, fer, 's-', label=f'{label_main} FER')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Error rate')
    plt.title('LDPC performance (BER + FER)')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

#  Entrypoint 
if __name__ == "__main__":

    # Same scaling/meaning as your MATLAB: snr[0..10] is Eb/N0(dB)
    snr_dB = np.arange(0.0, 3.3, 0.3)

    # Example: max frames like your original idea (edit as you like)
    max_frames = [10, 100, 1000, 1000, 1000, 10000, 10000, 10000, 100000, 100000, 100000]

    # Change #1: larger error targets at higher SNR so you don’t get BER0 “fake points”
    # (Tweak these to match your runtime budget.)
    error_limit = [200, 200, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
    # Change #2: avoid stopping after 1–2 frames at low SNR
    min_frames = [20, 20, 20, 30, 50, 80, 100, 150, 200, 200, 200]

    # Decoder settings (Change #3)
    mode = "NMS"     # "MS", "NMS", or "OMS"
    alpha = 0.80     # used for NMS (and MS ignored)
    beta = 0.15      # used for OMS

    snr_dB_range, ber, fer, avg_iters, used_frames, used_errors = simulate_ldpc(
        snr_dB_range=snr_dB,
        max_frames_per_snr=max_frames,
        error_limit_per_snr=error_limit,
        min_frames_per_snr=min_frames,
        max_iter=50,
        mode=mode,
        alpha=alpha,
        beta=beta,
        seed=12345,
        verbose=True
    )

    label = f"LDPC {mode} " + (f"(a={alpha})" if mode in ("MS", "NMS") else f"(b={beta})")
    plot_curves(snr_dB_range, ber, fer, label_main=label)

    print("\nEb/N0(dB)  MaxFrames  MinFrames  ErrLimit  UsedFrames  BitErrors   BER        FER      AvgIters")
    for i in range(len(snr_dB_range)):
        print(f"{snr_dB_range[i]:7.2f}  {max_frames[i]:9d}  {min_frames[i]:9d}  {error_limit[i]:8d}  "
              f"{used_frames[i]:9d}  {used_errors[i]:9d}  {ber[i]:.3e}  {fer[i]:.3e}  {avg_iters[i]:7.2f}")
