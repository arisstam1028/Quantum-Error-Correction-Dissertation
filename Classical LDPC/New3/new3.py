import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.special import erfc
import argparse

# Load fixed LDPC H
from ldpc_H_matrix import H

assert H.shape == (500, 1000), f"Expected (500,1000), got {H.shape}"
assert np.all(H.sum(axis=0) == 3), "Column weights are not all 3"
assert np.all(H.sum(axis=1) == 6), "Row weights are not all 6"

# Helper: non-zero sign
def nzsign(x):
    return np.where(x < 0, -1.0, 1.0)

# Horizontal step (α-Min-Sum)
def horizontal_step_min_sum(H, Q, alpha=0.8):
    E = (H == 1)
    m, n = H.shape

    mags = np.where(E, np.abs(Q), np.inf)
    min1 = np.min(mags, axis=1)
    pos1 = np.argmin(mags, axis=1)

    mags2 = mags.copy()
    mags2[np.arange(m), pos1] = np.inf
    min2 = np.min(mags2, axis=1)

    sign_masked = np.where(E, nzsign(Q), 1.0)
    total_sign = np.prod(sign_masked, axis=1)
    R_sign = total_sign[:, None] * sign_masked

    is_first = (np.arange(n)[None, :] == pos1[:, None]) & E
    R_mag = np.where(is_first, min2[:, None], min1[:, None])

    R = alpha * R_sign * R_mag
    R[~E] = 0.0
    return R

# Vertical step
def vertical_step(H, R, q):
    E = (H == 1)
    Q = q[None, :] + R.sum(axis=0)[None, :] - R
    Q[~E] = 0.0
    return Q

# Decode one frame
def min_sum_decode_single(H, y, sigma2, max_iter=50, alpha=0.8):
    q = (2.0 * y) / sigma2
    Q = np.where(H, q[None, :], 0.0)

    for it in range(1, max_iter + 1):
        R = horizontal_step_min_sum(H, Q, alpha)
        Q = vertical_step(H, R, q)

        q_hat = q + R.sum(axis=0)
        decoded = (q_hat < 0).astype(int)

        if np.all(H.dot(decoded) % 2 == 0):
            return decoded, it, True

    return decoded, max_iter, False

# Monte-Carlo simulation WITH error-limit stopping rule
def simulate_min_sum_ber(H,
                         ebn0_dB_range,
                         frames_per_snr,
                         max_iter=50,
                         alpha=0.8,
                         error_limit=100,
                         seed=None,
                         verbose=True):

    rng = np.random.RandomState(seed)

    m, n = H.shape
    Rc = (n - m) / n

    ber = np.zeros(len(ebn0_dB_range))
    avg_iters = np.zeros(len(ebn0_dB_range))
    frames_used = np.zeros(len(ebn0_dB_range), dtype=int)
    errors_used = np.zeros(len(ebn0_dB_range), dtype=int)

    c = np.zeros(n, dtype=int)
    x0 = 1 - 2 * c

    start_time = time.time()

    for idx, ebn0_db in enumerate(ebn0_dB_range):
        frames_max = int(frames_per_snr[idx])

        EbN0 = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0)
        sigma = math.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0
        frames_run = 0

        if verbose:
            print(f"\nEb/N0 = {ebn0_db:.2f} dB | max frames={frames_max} | error_limit={error_limit} | alpha={alpha} | max_iter={max_iter}")

        for f in range(frames_max):
            noise = sigma * rng.randn(n)
            y = x0 + noise

            decoded, it_used, _ = min_sum_decode_single(H, y, sigma2, max_iter=max_iter, alpha=alpha)

            bit_errors += int(np.sum(decoded != c))
            total_iters += it_used
            frames_run += 1

            if bit_errors >= error_limit:
                if verbose:
                    print(f"  Stopped early at frame {frames_run}: reached {bit_errors} bit errors.")
                break

        ber[idx] = bit_errors / (frames_run * n)
        avg_iters[idx] = total_iters / frames_run
        frames_used[idx] = frames_run
        errors_used[idx] = bit_errors

        if verbose:
            elapsed = time.time() - start_time
            print(f"→ BER={ber[idx]:.3e}, avg iters={avg_iters[idx]:.2f}, frames used={frames_run}, errors={bit_errors}, elapsed={elapsed:.1f}s")

    # Plot
    plt.figure(figsize=(8, 5))
    ebn0_lin = 10.0 ** (ebn0_dB_range / 10.0)
    plt.semilogy(ebn0_dB_range, 0.5 * erfc(np.sqrt(ebn0_lin)), 'k--', label='Uncoded BPSK')
    plt.semilogy(ebn0_dB_range, ber, 'o-', label=f'LDPC α-Min-Sum (α={alpha})')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\nEb/N0(dB)   MaxFrames   UsedFrames   Errors   BER        AvgIters")
    for e, mf, uf, err, b, it in zip(ebn0_dB_range, frames_per_snr, frames_used, errors_used, ber, avg_iters):
        print(f"{e:7.2f}   {mf:9d}   {uf:9d}   {err:6d}   {b:8.3e}   {it:7.2f}")

    return ber, avg_iters

# Entrypoint with CLI arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDPC α-Min-Sum BER simulation with error-limit stopping.")
    parser.add_argument("--alpha", type=float, default=0.80, help="Normalized Min-Sum alpha (e.g. 0.75..0.9)")
    parser.add_argument("--max_iter", type=int, default=50, help="Max decoding iterations per frame")
    parser.add_argument("--error_limit", type=int, default=100, help="Stop each SNR after this many bit errors")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Less terminal printing")
    args = parser.parse_args()

    ebn0_dB_range = np.arange(0.0, 11.0, 1.0)
    frames = [10, 100, 1000, 1000, 10000, 10000, 20000, 50000, 100000, 100000, 100000]

    simulate_min_sum_ber(
        H=H,
        ebn0_dB_range=ebn0_dB_range,
        frames_per_snr=frames,
        max_iter=args.max_iter,
        alpha=args.alpha,
        error_limit=args.error_limit,
        seed=args.seed,
        verbose=not args.quiet
    )
