import numpy as np
from LDPC1.display import show_results_popup  # Make sure display.py is in the same folder


# Helper functions

def hard_decision(llr):
    """Return 0/1 decoded bits based on the sign of LLR"""
    return (llr < 0).astype(int)


def compute_syndrome(H, c):
    """Compute the syndrome vector s = H * c mod 2"""
    return H.dot(c) % 2


def horizontal_step_min_sum(H, Q):
    """
    Min-Sum Horizontal Step (check-to-variable messages)
    R[i,j] = product of signs of other connected Q * minimum magnitude
    """
    rows, cols = H.shape
    R = np.zeros_like(Q)
    for i in range(rows):
        for j in range(cols):
            if H[i, j] == 1:
                # All other variable nodes connected to this check node
                others = []
                for jj in range(cols):
                    if H[i, jj] == 1 and jj != j:
                        others.append(Q[i, jj])
                if others:
                    sgn = 1
                    for val in others:
                        sgn *= np.sign(val)
                    minimum = np.min(np.abs(others))
                    R[i, j] = sgn * minimum
    return R


def vertical_step(H, R, q):
    """
    Vertical step: update variable-to-check messages Q_new
    Q_ij = q_j + sum of other R_i'j
    """
    rows, cols = H.shape
    Q_new = np.zeros((rows, cols))
    for j in range(cols):
        for i in range(rows):
            if H[i, j] == 1:
                others_sum = 0
                for ii in range(rows):
                    if ii != i and H[ii, j] == 1:
                        others_sum += R[ii, j]
                Q_new[i, j] = q[j] + others_sum
    return Q_new


def update_llr(H, R, q):
    """
    Compute final LLRs q_hat for all variable nodes
    q_hat_j = q_j + sum of all connected R_ij
    """
    rows, cols = H.shape
    q_hat = np.zeros_like(q)
    for j in range(cols):
        total = 0
        for i in range(rows):
            if H[i, j] == 1:
                total += R[i, j]
        q_hat[j] = q[j] + total
    return q_hat


def run_ldpc_min_sum(iterations=2, use_random_noise=False):
    # Parity-check matrix H
    H = np.array([
        [1, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 0]
    ])

    # Transmitted codeword and BPSK modulation
    c = np.array([0, 0, 0, 0, 0, 0])
    x = 1 - 2 * c  # 0 -> 1, 1 -> -1

    # Noise parameters
    sigma2 = 1.0
    sigma = np.sqrt(sigma2)

    # Received signal
    if use_random_noise:
        y = x + np.random.normal(0, sigma, size=x.shape)
    else:
        # Fixed example to match worked example
        y = np.array([0.339, 1.492, 0.6115, 2.4295, 1.5465, -0.368])

    # Initial hard decision and syndrome
    c_hat = hard_decision(y)
    errors = np.sum(c_hat != c)
    syndrome = compute_syndrome(H, c_hat)

    # Initial LLRs (Min-Sum initialization: q_j = 2*y/sigma^2)
    #q = (2 * y) / sigma2
    # For some reason when I initialised q as is depicted in the example all the values of R were halfed
    q = y.copy()
    Q = H * q  # initial variable-to-check messages

    # Iterative decoding
    for it in range(iterations):
        R = horizontal_step_min_sum(H, Q)
        Q = vertical_step(H, R, q)
        q_hat = update_llr(H, R, q)
        decoded = hard_decision(q_hat)

        # Stop if all errors corrected
        if np.all(decoded == c):
            break

    errors_after = np.sum(decoded != c)

    # Display results in popup
    show_results_popup(H, c, x, y, c_hat, syndrome, q, Q, R, Q, q_hat, decoded,
                       errors, errors_after)


if __name__ == "__main__":
    run_ldpc_min_sum()
