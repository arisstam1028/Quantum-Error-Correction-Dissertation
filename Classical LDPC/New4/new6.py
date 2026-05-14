import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---------------------------------------------------------------------
# Load fixed LDPC parity-check matrix
# ---------------------------------------------------------------------
from ldpc_H_matrix import H   # shape (500, 1000)

assert H.shape == (500, 1000)
assert np.all(H.sum(axis=0) == 3)
assert np.all(H.sum(axis=1) == 6)

E = (H == 1)   # edge mask (precomputed)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def nzsign(x):
    """Sign with sign(0)=+1"""
    return np.where(x < 0, -1.0, 1.0)


# ---------------------------------------------------------------------
# Check node update (MS / NMS / OMS)
# ---------------------------------------------------------------------
def horizontal_step(H, Q, mode="NMS", alpha=0.8, beta=0.15):
    mags = np.where(E, np.abs(Q), np.inf)

    min1 = np.min(mags, axis=1)
    pos1 = np.argmin(mags, axis=1)

    mags2 = mags.copy()
    rows = np.arange(H.shape[0])
    mags2[rows, pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    signs = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(signs, axis=1)
    R_sign = total_sign[:, None] * signs

    is_first = (np.arange(H.shape[1])[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first, min2[:, None], min1[:, None])

    if mode == "NMS":
        R_mag = alpha * R_mag
    elif mode == "OMS":
        R_mag = np.maximum(R_mag - beta, 0.0)

    R = R_sign * R_mag
    R[~E] = 0.0
    return R


# ---------------------------------------------------------------------
# Variable node update
# ---------------------------------------------------------------------
def vertical_step(R, q):
    total = R.sum(axis=0)
    Q = q[None, :] + total[None, :] - R
    Q[~E] = 0.0
    return Q


def compute_q_hat(R, q):
    return q + R.sum(axis=0)


# ---------------------------------------------------------------------
# Decode one frame
# ---------------------------------------------------------------------
def decode_single(y, sigma2, max_iter, mode, alpha, beta):
    q = (2.0 * y) / sigma2
    Q = np.where(E, q[None, :], 0.0)

    for it in range(1, max_iter + 1):
        R = horizontal_step(H, Q, mode, alpha, beta)
        Q = vertical_step(R, q)

        q_hat = compute_q_hat(R, q)
        decoded = (q_hat < 0).astype(int)

        if np.all((H @ decoded) % 2 == 0):
            return decoded, it, True

    return decoded, max_iter, False


# ---------------------------------------------------------------------
# Monte-Carlo simulation
# ---------------------------------------------------------------------
def simulate_ldpc(
    snr_dB_range,
    max_frames,
    min_frames,
    error_limit,
    max_iter=50,
    mode="NMS",
    alpha=0.8,
    beta=0.15,
    seed=12345,
    verbose=True
):
    rng = np.random.default_rng(seed)

    n = H.shape[1]
    k = n - H.shape[0]
    Rc = k / n

    ber = []
    fer = []
    avg_iters = []
    used_frames = []
    used_errors = []

    x0 = np.ones(n)   # all-zero codeword → BPSK +1

    for i, snr_db in enumerate(snr_dB_range):
        EbN0 = 10 ** (snr_db / 10)
        sigma2 = 1 / (2 * Rc * EbN0)
        sigma = math.sqrt(sigma2)

        bit_errs = 0
        frame_errs = 0
        it_sum = 0

        t0 = time.time()

        if verbose:
            print(f"\nEb/N0 = {snr_db:.2f} dB | max frames={max_frames[i]} "
                  f"| min_frames={min_frames[i]} | error_limit={error_limit[i]} "
                  f"| mode={mode} | alpha={alpha} | max_iter={max_iter}")

        for f in range(1, max_frames[i] + 1):
            noise = sigma * rng.standard_normal(n)
            y = x0 + noise

            decoded, it_used, conv = decode_single(
                y, sigma2, max_iter, mode, alpha, beta
            )

            errs = np.sum(decoded)
            bit_errs += errs
            frame_errs += int(errs > 0)
            it_sum += it_used

            if verbose and f % max(1, max_frames[i] // 10) == 0:
                print(f"  frame {f}/{max_frames[i]}  current BER={bit_errs/(f*n):.3e}  frame_errs={frame_errs}")

            # ---------------- ERROR LIMIT STOP ----------------
            if f >= min_frames[i] and bit_errs >= error_limit[i]:
                if verbose:
                    print(f"  Stopped early at frame {f}: reached {bit_errs} bit errors.")
                break

        total_bits = f * n

        ber_i = bit_errs / total_bits if bit_errs > 0 else 0.5 / total_bits
        fer_i = frame_errs / f if frame_errs > 0 else 0.5 / f

        ber.append(ber_i)
        fer.append(fer_i)
        avg_iters.append(it_sum / f)
        used_frames.append(f)
        used_errors.append(bit_errs)

        if verbose:
            print(f"→ BER={ber_i:.3e}, FER={fer_i:.3e}, avg iters={avg_iters[-1]:.2f}, "
                  f"frames used={f}, errors={bit_errs}, elapsed={time.time()-t0:.1f}s")

    return (
        np.array(ber),
        np.array(fer),
        np.array(avg_iters),
        np.array(used_frames),
        np.array(used_errors),
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":

    snr_dB = np.arange(0, 11, 1)

    max_frames = [10, 100, 1000, 50000, 50000, 50000, 100000, 200000, 1000000, 1000000, 1000000]
    min_frames = [20, 20, 20, 30, 50, 80, 100, 150, 200, 200, 200]
    error_limit = [200, 200, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]

    ber, fer, avg_it, used_f, used_e = simulate_ldpc(
        snr_dB,
        max_frames,
        min_frames,
        error_limit,
        max_iter=50,
        mode="NMS",
        alpha=0.8,
        beta=0.15,
        verbose=True
    )

    # Plot
    ebn0_lin = 10 ** (snr_dB / 10)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_dB, ber_uncoded, "k--", label="Uncoded BPSK (approx)")
    plt.semilogy(snr_dB, ber, "o-", label="LDPC NMS BER")
    plt.semilogy(snr_dB, fer, "s-", label="LDPC NMS FER")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Error rate")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()