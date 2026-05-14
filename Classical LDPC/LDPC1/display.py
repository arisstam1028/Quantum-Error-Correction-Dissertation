# display.py
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import numpy as np

def show_results_popup(H, c, x, y, c_hat_before, syndrome_before, q, Q, R, Q_new, q_hat, decoded, errors_before, errors_after):
    results = f"""
Step 1: Parity-check matrix H
{H}

Step 2: Transmitted codeword and BPSK modulation
Transmitted codeword (c): {c}
BPSK-modulated signal (x): {x}

Step 3: Received noisy signal
Received signal (y): {y}

Step 4: Hard decision & Bit Errors
Hard decision: {c_hat_before}
Number of bit errors: {errors_before}

Step 4a: Syndrome
{syndrome_before}

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
    root = tk.Tk()
    root.title("LDPC SPA Decoding Results")
    text_area = ScrolledText(root, wrap=tk.WORD, width=80, height=40)
    text_area.pack(padx=10, pady=10)
    text_area.insert(tk.END, results)
    text_area.configure(state='disabled')
    root.mainloop()
