import numpy as np
from display import show_results_popup

# Function to safely compute inverse hyperbolic tangent
def safe_atanh(x):
    x = np.clip(x, -0.999999, 0.999999)
    return np.arctanh(x)

# Make a hard decision based on the sign of LLRs
def hard_decision(llr):
    return (llr < 0).astype(int)

# Compute the syndrome vector s  H * c mod 2
def compute_syndrome(H, c):
    return H.dot(c) % 2

# Horizontal step: compute check-to-variable messages R
def horizontal_step(H, Q):
    rows, cols = H.shape
    R = np.zeros_like(Q, dtype=float)
    for i in range(rows):
        for j in range(cols):
            if H[i, j] == 1:
                # Multiply tanh(Q/2) of all other variable nodes connected to this check node
                others = [np.tanh(Q[i, jj]/2) for jj in range(cols) if H[i, jj] == 1 and jj != j]
                R[i, j] = 2 * safe_atanh(np.prod(others))
    return R

# Vertical step: update variable-to-check messages Q_new
def vertical_step(H, R, q):
    rows, cols = H.shape
    Q_new = np.zeros((rows, cols))
    for j in range(cols):
        for i in range(rows):
            if H[i, j] == 1:
                # Sum all R values from other check nodes connected to this variable node
                others = [R[ii, j] for ii in range(rows) if H[ii, j] == 1 and ii != i]
                Q_new[i, j] = q[j] + np.sum(others)
    return Q_new

# Update the final LLR values q_hat for all variable nodes
def update_llr(H, R, q):
    cols = H.shape[1]
    q_hat = np.zeros_like(q)
    for j in range(cols):
        # Sum all R messages from connected check nodes and add the initial LLR q
        q_hat[j] = q[j] + np.sum([R[i, j] for i in range(H.shape[0]) if H[i, j] == 1])
    return q_hat

# Main function to run the LDPC SPA decoding
def run_ldpc_spa():
    # Define the parity-check matrix H
    H = np.array([
        [1, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 0]
    ])

    # Define the transmitted codeword c
    c = np.array([0, 0, 0, 0, 0, 0])

    # BPSK modulation: 0 -> +1, 1 -> -1
    x = 1 - 2*c

    # Channel noise parameters
    sigma2 = 1.0  # Noise variance
    sigma = np.sqrt(sigma2)

    # Received signal y
    # Option 1: Fixed received signal (for reproducibility)
    y = np.array([0.339, 1.492, 0.6115, 2.4295, 1.5465, -0.368])
    # Option 2: Add random Gaussian noise to the transmitted signal
    # y  x + np.random.normal(0, sigma, sizex.shape)

    # Step 1: Hard decision
    c_hat = hard_decision(y)
    errors = np.sum(c_hat!= c)  # Count errors before decoding

    # Step 2: Compute syndrome for the received hard decision
    syndrome = compute_syndrome(H, c_hat)

    # Step 3: Compute initial LLRs from received signal
    q = (2 * y) / sigma2

    # Step 4: Initialize variable-to-check message matrix Q
    Q = H * q  # Only consider LLRs for positions where H1

    # Step 5: Horizontal step - check-to-variable messages
    R = horizontal_step(H, Q)

    # Step 6: Vertical step - update variable-to-check messages
    Q_new = vertical_step(H, R, q)

    # Step 7: Compute final LLRs for each variable node
    q_hat = update_llr(H, R, q)

    # Step 8: Hard decision after decoding
    decoded = hard_decision(q_hat)
    errors_after = np.sum(decoded != c)  # Count errors after decoding

    # Step 9: Display all results in a popup window
    show_results_popup(H, c, x, y, c_hat, syndrome, q, Q, R, Q_new, q_hat, decoded, errors, errors_after)

# Run the LDPC SPA decoding
run_ldpc_spa()