import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import numpy as np

def show_results_popup(H, c, x, y, c_hat, syndrome, q, Q, R, Q_new, q_hat, decoded, errors, errors_after):
    """
    Display LDPC decoding results in a popup window.
    Works for SPA or Min-Sum algorithms.
    """

    # Build a formatted string with all steps
    results = f"""
Step 1: Parity-check matrix H
{H}

Step 2: Transmitted codeword and BPSK modulation
Transmitted codeword (c): {c}
BPSK-modulated signal (x): {x}

Step 3: Received noisy signal
Received signal (y): {np.round(y, 3)}

Step 4: Hard decision & Numer of Bit Errors
Hard decision: {c_hat}
Number of bit errors: {errors}

Step 4a: Syndrome
{syndrome}

Step 5: Received LLRs (q)
{np.round(q, 3)}

Step 6: Initial Q matrix (variable-to-check messages)
{np.round(Q, 3)}

Step 7: R matrix after horizontal step (check-to-variable messages)
{np.round(R, 3)}

Step 8: Q matrix after vertical update
{np.round(Q_new, 3)}

Step 9: Updated q_hat values (final LLRs)
{np.round(q_hat, 3)}

Step 10: Hard decision after decoding
Decoded codeword: {decoded}
Number of bit errors after decoding: {errors_after}
"""

    # Create popup window
    root = tk.Tk()
    root.title("LDPC Decoding Results")

    # Add a scrollable text area
    text_area = ScrolledText(root, wrap=tk.WORD, width=80, height=40)
    text_area.pack(padx=10, pady=10)

    # Insert results
    text_area.insert(tk.END, results)
    text_area.configure(state='disabled')  # make it read-only

    # Run the GUI loop
    root.mainloop()
