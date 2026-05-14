"""
LDPC Monte Carlo simulation (BER-only) with multiple curves on one plot.

Features included:
1) Fixed H loaded from ldpc_H_matrix.py (same matrix for all SNRs/frames/curves)
2) AWGN channel with Eb/N0 scaling: sigma^2 = 1/(2*Rc*Eb/N0)
3) Decoder supports:
   - MS  : plain Min-Sum
   - NMS : Normalized Min-Sum (alpha)
   - OMS : Offset Min-Sum (beta)
4) Early stop per SNR using BOTH:
   - min_frames (must run at least this many frames)
   - error_limit (stop once bit_errors >= error_limit, but only AFTER min_frames reached)
   - also never exceed max_frames
5) Multiple BER curves on one graph (FER removed)

Requirements:
- ldpc_H_matrix.py must define:
    H : numpy array shape (500,1000) with column weight=3 and row weight=6
"""

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

# -----------------------------------------------------------------------------
# Load fixed LDPC H matrix
# -----------------------------------------------------------------------------
from ldpc_H_matrix import H  # must be numpy array (500, 1000)

# Sanity checks (keep these!)
assert H.shape == (500, 1000), f"Expected H shape (500,1000), got {H.shape}"
assert np.all(H.sum(axis=0) == 3), "Column weights are not all 3"
assert np.all(H.sum(axis=1) == 6), "Row weights are not all 6"


# -----------------------------------------------------------------------------
# Helper: sign with 0 mapped to +1
# -----------------------------------------------------------------------------
def nzsign(x):
    """Non-zero sign: +1 for x>=0, -1 for x<0 (so sign(0)=+1)."""
    return np.where(x < 0, -1.0, 1.0)


# -----------------------------------------------------------------------------
# Horizontal step: Min-Sum / Normalized Min-Sum / Offset Min-Sum
# -----------------------------------------------------------------------------
def horizontal_step_min_sum(H, Q, mode="NMS", alpha=0.8, beta=0.15):
    """
    Vectorised horizontal step.

    mode:
      - "MS"  : plain Min-Sum
      - "NMS" : Normalized Min-Sum: magnitude *= alpha
      - "OMS" : Offset Min-Sum: magnitude = max(magnitude - beta, 0)
    """
    E = (H == 1)
    m, n = H.shape

    degs = E.sum(axis=1)

    mags_masked = np.where(E, np.abs(Q), np.inf)

    # First min and its position
    min1 = np.min(mags_masked, axis=1)
    pos1 = np.argmin(mags_masked, axis=1)

    # Second min
    mags2 = mags_masked.copy()
    rows_idx = np.arange(m)
    mags2[rows_idx, pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    # Signs
    sign_masked = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(sign_masked, axis=1)

    # Outgoing sign for each edge (exclude self => total_sign * sign(Q_ij))
    R_sign = total_sign[:, None] * np.where(E, nzsign(Q), 1.0)

    is_first_min = (np.arange(n)[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first_min, min2[:, None], min1[:, None])

    # Handle rows with deg <= 1
    low_deg_rows = (degs <= 1)
    if np.any(low_deg_rows):
        R_mag[low_deg_rows, :] = 0.0
        R_sign[low_deg_rows, :] = 0.0

    # Apply MS / NMS / OMS magnitude rule
    mode = mode.upper()
    if mode == "MS":
        pass
    elif mode == "NMS":
        R_mag = alpha * R_mag
    elif mode == "OMS":
        R_mag = np.maximum(R_mag - beta, 0.0)
    else:
        raise ValueError("mode must be one of: 'MS', 'NMS', 'OMS'")

    R = R_sign * R_mag
    R[~E] = 0.0
    return R


# -----------------------------------------------------------------------------
# Vertical step
# -----------------------------------------------------------------------------
def vertical_step(H, R, q):
    """Q_new[i,j] = q[j] + sum_{i' in M(j)\{i}} R[i',j] = q + colsum(R) - R"""
    E = (H == 1)
    total_R_col = R.sum(axis=0)
    Q_new = q[None, :] + total_R_col[None, :] - R
    Q_new[~E] = 0.0
    return Q_new


def compute_q_hat(R, q):
    """q_hat[j] = q[j] + sum_i R[i,j]"""
    return q + R.sum(axis=0)


# -----------------------------------------------------------------------------
# Decode one frame
# -----------------------------------------------------------------------------
def min_sum_decode_single(H, y, sigma2, max_iter=50, mode="NMS", alpha=0.8, beta=0.15):
    """
    Returns: decoded_bits, it_used, converged_flag
    """
    q = (2.0 * y) / sigma2
    Q = np.where(H, q[None, :], 0.0)

    for it in range(1, max_iter + 1):
        R = horizontal_step_min_sum(H, Q, mode=mode, alpha=alpha, beta=beta)
        Q = vertical_step(H, R, q)

        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(np.uint8)

        s = (H.dot(decoded) % 2)
        if np.all(s == 0):
            return decoded, it, True

    return decoded, max_iter, False


# -----------------------------------------------------------------------------
# Simulation BER-only (with enforced error_limit)
# -----------------------------------------------------------------------------
def simulate_ldpc_ber(
    H,
    ebn0_dB_range,
    max_frames_per_snr,
    min_frames_per_snr,
    error_limit_per_snr,
    max_iter=50,
    mode="NMS",
    alpha=0.8,
    beta=0.15,
    seed=12345,
    verbose=True
):
    """
    BER-only Monte Carlo simulation.

    Enforced stopping rule per SNR:
      - Always run at least min_frames
      - Stop early if bit_errors >= error_limit AND frames_done >= min_frames
      - Never exceed max_frames
    """
    rng = np.random.RandomState(seed)

    ebn0_dB_range = np.asarray(ebn0_dB_range, dtype=float)
    max_frames_per_snr = np.asarray(max_frames_per_snr, dtype=int)
    min_frames_per_snr = np.asarray(min_frames_per_snr, dtype=int)
    error_limit_per_snr = np.asarray(error_limit_per_snr, dtype=int)

    L = len(ebn0_dB_range)
    if not (len(max_frames_per_snr) == len(min_frames_per_snr) == len(error_limit_per_snr) == L):
        raise ValueError("max_frames_per_snr, min_frames_per_snr, error_limit_per_snr must match ebn0_dB_range length")

    # Ensure min_frames <= max_frames for all points
    min_frames_per_snr = np.minimum(min_frames_per_snr, max_frames_per_snr)

    m, n = H.shape
    k = n - m
    Rc = float(k) / n

    ber = np.zeros(L, dtype=float)
    avg_iters = np.zeros(L, dtype=float)
    used_frames = np.zeros(L, dtype=int)
    bit_errors_out = np.zeros(L, dtype=int)

    # all-zero codeword
    c = np.zeros(n, dtype=np.uint8)
    x0 = 1 - 2 * c

    start_time = time.time()

    for idx, ebn0_db in enumerate(ebn0_dB_range):
        max_frames = int(max_frames_per_snr[idx])
        min_frames = int(min_frames_per_snr[idx])
        err_limit = int(error_limit_per_snr[idx])

        EbN0_lin = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        sigma = math.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0

        if verbose:
            extra = f"alpha={alpha}" if mode.upper() == "NMS" else (f"beta={beta}" if mode.upper() == "OMS" else "")
            print(f"\nEb/N0 = {ebn0_db:.2f} dB | max frames={max_frames} | min_frames={min_frames} | "
                  f"error_limit={err_limit} | mode={mode} {extra} | max_iter={max_iter}")

        frames_done = 0
        for f in range(max_frames):
            noise = sigma * rng.randn(n)
            y = x0 + noise

            decoded, it_used, _ = min_sum_decode_single(
                H, y, sigma2, max_iter=max_iter, mode=mode, alpha=alpha, beta=beta
            )

            errs_this = int(np.sum(decoded != c))
            bit_errors += errs_this
            total_iters += it_used
            frames_done += 1

            # progress print occasionally (lightweight)
            if verbose and (frames_done % max(1, max_frames // 10) == 0):
                curr_ber = bit_errors / float(frames_done * n)
                print(f"  frame {frames_done}/{max_frames}  current BER={curr_ber:.3e}  bit_errors={bit_errors}")

            # ---- EARLY STOP: enforce error limit, but only after min_frames ----
            if (frames_done >= min_frames) and (bit_errors >= err_limit):
                if verbose:
                    print(f"  Stopped early at frame {frames_done}: reached {bit_errors} bit errors (limit={err_limit}).")
                break

        total_bits = frames_done * n
        ber[idx] = bit_errors / float(total_bits) if total_bits > 0 else 0.0
        avg_iters[idx] = float(total_iters) / frames_done if frames_done > 0 else 0.0
        used_frames[idx] = frames_done
        bit_errors_out[idx] = bit_errors

        if verbose:
            elapsed = time.time() - start_time
            print(f"→ BER={ber[idx]:.3e}, avg iters={avg_iters[idx]:.2f}, frames used={frames_done}, "
                  f"errors={bit_errors}, elapsed={elapsed:.1f}s")

    return ebn0_dB_range, ber, avg_iters, used_frames, bit_errors_out


# -----------------------------------------------------------------------------
# Entrypoint: MULTIPLE BER CURVES (FER removed)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Same Eb/N0 style as your MATLAB repetition example:
    ebn0_dB_range = np.arange(0.0, 11.0, 1.0)

    # You can keep your existing lists (edit as you like)
    max_frames = [10, 100, 1000, 1000, 10000, 10000, 20000, 50000, 100000, 100000, 100000]
    min_frames = [10, 20, 20, 30, 50, 80, 100, 150, 200, 200, 200]  # must be <= max_frames
    error_limit = [200, 200, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]

    max_iter = 50
    seed = 12345

    # ---- MULTIPLE CURVES CHOICE ----
    # A) NMS with different alphas (recommended)
    mode = "NMS"
    alpha_list = [0.70, 0.75, 0.80, 0.85, 0.90]

    # If you ever want B) compare modes instead, use for example:
    # curves = [("MS", None, None), ("NMS", 0.8, None), ("OMS", None, 0.15)]

    all_bers = {}

    for a in alpha_list:
        ebn0s, ber, avg_iters, used_frames, bit_errors = simulate_ldpc_ber(
            H=H,
            ebn0_dB_range=ebn0_dB_range,
            max_frames_per_snr=max_frames,
            min_frames_per_snr=min_frames,
            error_limit_per_snr=error_limit,
            max_iter=max_iter,
            mode=mode,
            alpha=a,
            beta=0.15,
            seed=seed,
            verbose=True
        )
        all_bers[a] = ber

        print("\nEb/N0(dB)  MaxFrames  MinFrames  ErrLimit  UsedFrames  BitErrors   BER        AvgIters")
        for i in range(len(ebn0s)):
            print(f"{ebn0s[i]:7.2f}  {max_frames[i]:9d}  {min_frames[i]:9d}  {error_limit[i]:8d}  "
                  f"{used_frames[i]:9d}  {bit_errors[i]:9d}  {ber[i]:.3e}   {avg_iters[i]:7.2f}")

    # ---- Plot: Uncoded + multiple BER curves ----
    plt.figure(figsize=(8, 5))

    ebn0_lin = 10.0 ** (ebn0_dB_range / 10.0)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))
    plt.semilogy(ebn0_dB_range, ber_uncoded, 'k--', label='Uncoded BPSK (approx)')

    for a in alpha_list:
        plt.semilogy(ebn0_dB_range, all_bers[a], marker='o', label=f'LDPC {mode} BER (α={a:.2f})')

    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('LDPC BER vs Eb/N0 (multiple curves, FER removed)')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
