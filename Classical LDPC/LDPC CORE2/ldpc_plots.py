import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from ldpc_core import simulate_min_sum_ber


def plot_ber_vs_iteration_all_snr(ebn0_dB, ber_vs_iter):
    """
    Plot BER vs iteration for ALL Eb/N0 values together.

    ebn0_dB      : array of Eb/N0 values (e.g. [0,1,...,10])
    ber_vs_iter  : shape  [len(ebn0_dB), max_iter]
                    Each row is the BER trajectory for one SNR point.
    """

    num_snr, max_iter = ber_vs_iter.shape
    it_axis = np.arange(1, max_iter + 1)

    plt.figure(figsize=(10, 8))

    # Plot each Eb/N0 curve
    for idx, eb in enumerate(ebn0_dB):
        traj = ber_vs_iter[idx, :]

        # Skip empty rows (NaN)
        if np.all(np.isnan(traj)):
            continue

        plt.semilogy(it_axis, traj, marker='o', label=f"{eb:.1f} dB")

    plt.xlabel("Iteration")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Min-Sum: BER vs Iteration for Each Eb/N0")

    # Beautiful grid like your example
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)

    # Legend outside the plot (exactly like your screenshot)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()


def main():
    # Run simulation (this generates all BER and trajectory data)
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
    plt.title("BER vs Eb/N0 - LDPC Normalized Min-Sum")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    # Plot 2: BER vs iteration (ALL Eb/N0 curves together)
    plot_ber_vs_iteration_all_snr(ebn0_dB, ber_vs_iter)


if __name__ == "__main__":
    main()
