import numpy as np
import time

from ldpc_H_matrix import H  # ← hard-coded PEG LDPC matrix


# Step 2: Hard decision and syndrome

def hard_decision(q_hat):
    return (q_hat < 0).astype(int)


def syndrome(H, c):
    return H.dot(c) % 2


# Step 3: Build neighbour lists for layered decoding

def build_neighbors(H):
    m, n = H.shape
    checks_to_vars = []
    for i in range(m):
        checks_to_vars.append(np.where(H[i, :] == 1)[0])

    vars_to_checks = []
    for j in range(n):
        vars_to_checks.append(np.where(H[:, j] == 1)[0])

    return checks_to_vars, vars_to_checks


# Step 4: Layered Normalized Min-Sum decoding

def min_sum_decode_layered(H,
                           checks_to_vars,
                           vars_to_checks,
                           y,
                           sigma2,
                           max_iter=50,
                           alpha=0.75,
                           c_true=None,
                           ber_traj=None):
    m, n = H.shape
    L_ch = 2.0 * y / sigma2  # channel LLRs
    L_app = L_ch.copy()

    # Check-to-variable messages
    R = [np.zeros(len(neigh), dtype=float) for neigh in checks_to_vars]

    decoded = np.zeros(n, dtype=int)
    converged = False
    it_used = 0

    for it in range(1, max_iter + 1):
        # Layered schedule: process each check node
        for i in range(m):
            var_indices = checks_to_vars[i]
            if len(var_indices) == 0:
                continue

            # Extrinsic messages from variables:
            L_extr = np.empty(len(var_indices), dtype=float)
            for idx_local, j in enumerate(var_indices):
                L_extr[idx_local] = L_app[j] - R[i][idx_local]

            signs = np.sign(L_extr)
            signs[signs == 0] = 1.0
            mags = np.abs(L_extr)

            if len(mags) == 1:
                new_R = np.array([0.0])
            else:
                # min1, min2 trick
                min1 = np.inf
                min2 = np.inf
                min1_idx = -1
                for k, mag in enumerate(mags):
                    if mag < min1:
                        min2 = min1
                        min1 = mag
                        min1_idx = k
                    elif mag < min2:
                        min2 = mag

                total_sign = np.prod(signs)
                new_R = np.empty_like(R[i])
                for k in range(len(mags)):
                    sign_excl = total_sign * signs[k]
                    if k == min1_idx:
                        mag_excl = min2
                    else:
                        mag_excl = min1
                    new_R[k] = alpha * sign_excl * mag_excl

            # Update L_app in place (layered)
            for idx_local, j in enumerate(var_indices):
                L_app[j] += new_R[idx_local] - R[i][idx_local]
            R[i] = new_R

        decoded = hard_decision(L_app)

        if (c_true is not None) and (ber_traj is not None):
            ber_traj[it - 1] = np.sum(decoded != c_true) / float(len(c_true))

        s = syndrome(H, decoded)
        if np.all(s == 0):
            converged = True
            it_used = it
            break
        it_used = it

    return decoded, it_used, converged


# Step 5: Monte Carlo simulation (no plotting)

def simulate_min_sum_ber(
    ebn0_dB=None,
    max_iter=50,
    alpha=0.75,
    target_errors=100,
    max_frames_per_snr=200000
):
    """
    Uses the hard-coded H from ldpc_H_matrix.py.
    """
    m, n = H.shape
    k = n - m
    Rc = k / float(n)

    if ebn0_dB is None:
        ebn0_dB = np.arange(0.0, 11.0, 1.0)
    ebn0_dB = np.array(ebn0_dB, dtype=float)
    num_snr = len(ebn0_dB)

    print(f'H shape: {H.shape} (rows  N-K  {m}, cols  N  {n})')
    print(f'Code rate Rc  {Rc:.3f}')
    print(f'Using PEG-style regular (3,6) LDPC, layered Norm-Min-Sum (alpha{alpha})\n')

    checks_to_vars, vars_to_checks = build_neighbors(H)

    c = np.zeros(n, dtype=int)
    x0 = 1 - 2 * c

    ber = np.zeros(num_snr, dtype=float)
    avg_iters = np.zeros(num_snr, dtype=float)
    ber_vs_iter = np.full((num_snr, max_iter), np.nan)

    rng = np.random.default_rng(2025)

    for idx, eb in enumerate(ebn0_dB):
        EbN0_lin = 10.0 ** (eb / 10.0)
        sigma2 = 1.0 / (2.0 * Rc * EbN0_lin)
        sigma = np.sqrt(sigma2)

        bit_errors = 0
        total_iters = 0
        total_bits = 0
        frame_count = 0

        t0 = time.time()
        while bit_errors < target_errors and frame_count < max_frames_per_snr:
            frame_count += 1
            noise = sigma * rng.standard_normal(n)
            y = x0 + noise

            if idx == 0 and frame_count == 1:
                traj = np.zeros(max_iter, dtype=float)
                decoded, it_used, converged = min_sum_decode_layered(
                    H,
                    checks_to_vars,
                    vars_to_checks,
                    y,
                    sigma2,
                    max_iter=max_iter,
                    alpha=alpha,
                    c_true=c,
                    ber_traj=traj,
                )
                ber_vs_iter[idx, :] = traj
            else:
                decoded, it_used, converged = min_sum_decode_layered(
                    H,
                    checks_to_vars,
                    vars_to_checks,
                    y,
                    sigma2,
                    max_iter=max_iter,
                    alpha=alpha,
                )

            errors_this = np.sum(decoded != c)
            bit_errors += errors_this
            total_iters += it_used
            total_bits += n

            if frame_count >= max_frames_per_snr:
                break

        elapsed = time.time() - t0
        if total_bits > 0:
            ber[idx] = bit_errors / float(total_bits)
            avg_iters[idx] = total_iters / float(frame_count)

        s_last = syndrome(H, decoded)
        unsat_checks = np.sum(s_last != 0)

        print(f'Eb/N0{eb:.1f} dB, frames{frame_count}, bit_errors{bit_errors}, BER{ber[idx]:.3e}, avg_it{avg_iters[idx]:.2f}, unsat_checks_last{unsat_checks}, time{elapsed:.2f}s')

    info = {
        "n": n,
        "k": k,
        "m": m,
        "Rc": Rc,
        "col_w": int(H.sum(axis=0)[0]),
        "row_w": int(H.sum(axis=1)[0]),
        "target_errors": target_errors,
        "max_frames_per_snr": max_frames_per_snr,
    }

    return ebn0_dB, ber, avg_iters, ber_vs_iter, H, info