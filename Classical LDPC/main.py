# Purpose:
#   Demonstrates classical LDPC decoding with the sum-product algorithm
#   on a small parity-check matrix.
#
# Process:
#   1. Define a parity-check matrix, transmitted codeword, and noisy values.
#   2. Convert received values to log-likelihood ratios.
#   3. Run horizontal and vertical message-passing updates.
#   4. Display syndrome, messages, and decoded output in a popup.
#
# Theory link:
#   Classical LDPC BP decoding estimates a binary codeword from noisy
#   channel evidence. Check-node updates enforce parity constraints, and
#   variable-node updates combine channel LLRs with incoming messages.

import numpy as np
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

def safe_atanh(x):
    """
    Compute arctanh with clipping for numerical stability.

    Role in pipeline:
        Prevents check-node products from reaching +/-1, where the
        sum-product update would diverge.
    """
    x = np.clip(x, -0.999999, 0.999999)
    return np.arctanh(x)

def hard_decision(llr):
    """
    Convert LLRs to binary decisions.

    Role in pipeline:
        Produces the decoded bit estimate after message passing.
    """
    return (llr < 0).astype(int)

def compute_syndrome(H, c):
    """
    Compute the classical LDPC syndrome Hc over GF(2).

    Role in pipeline:
        Checks which parity constraints are violated by a hard decision.
    """
    return H.dot(c) % 2

def horizontal_step(H, Q):
    """
    Run the check-node update of the sum-product algorithm.

    Role in pipeline:
        Sends parity-constraint information from checks to variables.
    """
    rows, cols = H.shape
    R = np.zeros_like(Q, dtype=float)
    for i in range(rows):
        for j in range(cols):
            if H[i, j] == 1:
                others = [np.tanh(Q[i, jj]/2) for jj in range(cols) if H[i, jj]==1 and jj!=j]
                R[i, j] = 2 * safe_atanh(np.prod(others))
    return R

def vertical_step(H, R, q):
    """
    Run the variable-node update of the sum-product algorithm.

    Role in pipeline:
        Combines channel LLRs with incoming check messages to refresh
        variable-to-check beliefs.
    """
    rows, cols = H.shape
    Q_new = np.zeros((rows, cols))
    for j in range(cols):
        for i in range(rows):
            if H[i, j]==1:
                others = [R[ii, j] for ii in range(rows) if H[ii, j]==1 and ii!=i]
                Q_new[i, j] = q[j] + np.sum(others)
    return Q_new

def update_llr(H, R, q):
    """
    Compute final posterior LLR estimates for all variables.

    Role in pipeline:
        Aggregates channel evidence and check-node messages before the
        final hard decision.
    """
    cols = H.shape[1]
    q_hat = np.zeros_like(q)
    for j in range(cols):
        q_hat[j] = q[j] + np.sum([R[i,j] for i in range(H.shape[0]) if H[i,j]==1])
    return q_hat

def run_ldpc_spa_popup():
    """
    Execute the small LDPC demonstration and display all intermediate data.

    Role in pipeline:
        Shows the classical BP message-passing workflow that precedes the
        later quantum stabilizer and QLDPC simulations.
    """
    # Step 1: H matrix
    H = np.array([
        [1,0,1,0,1,1],
        [0,1,1,1,0,1],
        [1,1,0,1,1,0]
    ])
    # Step 2: Transmitted Codeword and BPSK
    c = np.array([0,0,0,0,0,0])
    x = 1 - 2*c
    # Step 3: Received signal
    sigma2 = 1.0
    sigma = np.sqrt(sigma2)
    y = np.array([0.339,1.492,0.6115,2.4295,1.5465,-0.368])
    # y  x + np.random.normal(0, sigma, sizex.shape)  # Uncomment for random noise

    # Hard decision before decoding
    c_hat = hard_decision(y)
    errors = np.sum(c_hat != c)
    syndrome = compute_syndrome(H, c_hat)

    # LLRs and messages
    q = (2*y)/sigma2
    Q = H * q
    R = horizontal_step(H, Q)
    Q_new = vertical_step(H, R, q)
    q_hat = update_llr(H, R, q)
    decoded = hard_decision(q_hat)
    errors_after = np.sum(decoded != c)

    # Build a single string for the popup
    results = f"""
Step 1: Parity-check matrix H
{H}

Step 2: Transmitted codeword and BPSK modulation
Transmitted codeword (c): {c}
BPSK-modulated signal (x): {x}

Step 3: Received noisy signal
Received signal (y): {y}

Step 4: Hard decision before decoding
Hard decision: {c_hat}
Number of bit errors before decoding: {errors}

Step 4a: Syndrome before decoding
{syndrome}

Step 5: Received LLRs (q)
{np.round(q,3)}

Step 6: Initial Q matrix (variable-to-check messages)
{np.round(Q,3)}

Step 7: R matrix after horizontal step (check-to-variable messages)
{np.round(R,3)}

Step 8: Q matrix after vertical update
{np.round(Q_new,3)}

Step 9: Updated q_hat values (final LLRs)
{np.round(q_hat,3)}

Step 10: Hard decision after decoding
Decoded codeword: {decoded}
Number of bit errors after decoding: {errors_after}
"""

    # Display in popup
    root = tk.Tk()
    root.title("LDPC SPA Decoding Results")
    text_area = ScrolledText(root, wrap=tk.WORD, width=80, height=40)
    text_area.pack(padx=10, pady=10)
    text_area.insert(tk.END, results)
    text_area.configure(state='disabled')
    root.mainloop()

# Run the LDPC SPA with popup
run_ldpc_spa_popup()
