import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from ldpc_core import simulate_min_sum_ber


def main():
    # Run the LDPC simulation (no plotting inside)
    ebn0_dB, ber, avg_iters, ber_vs_iter, H, info = simulate_min_sum_ber()

    # Plot 1: BER vs Eb/N0
    ebn0_lin = 10.0 ** (ebn0_dB / 10.0)
    ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))

    plt.figure(figsize=(8, 5))
    plt.semilogy(ebn0_dB, ber_uncoded, "k--", label="Uncoded BPSK")
    plt.semilogy(
        ebn0_dB,
        ber,
        "bo-",
        label=f"LDPC Min-Sum (N={info['n']}, rate={info['Rc']:.2f})"
    )
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Eb/N0 - LDPC Min-Sum")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    # Plot 2: BER vs iteration (first frame at each SNR)
    max_iter = ber_vs_iter.shape[1]
    it_axis = np.arange(1, max_iter + 1)

    plt.figure(figsize=(8, 5))
    for idx, eb in enumerate(ebn0_dB):
        traj = ber_vs_iter[idx, :]
        if np.all(np.isnan(traj)):
            continue
        plt.semilogy(it_axis, traj, label=f"{eb:.1f} dB")

    plt.xlabel("Iteration")
    plt.ylabel("Bit error rate (first frame)")
    plt.title("LDPC Min-Sum: BER vs iteration")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
