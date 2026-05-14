import matplotlib.pyplot as plt


class StabilizerCircuitPlotter:
    @staticmethod
    def show_bp_comparison_curves(
        probabilities,
        bsc_average_qber,
        symmetric_average_qber,
    ):
        plt.figure(figsize=(7, 5))

        plt.plot(
            probabilities,
            bsc_average_qber,
            marker="o",
            linestyle="-",
            label="BP + Independent BSC approximation",
        )

        plt.plot(
            probabilities,
            symmetric_average_qber,
            marker="s",
            linestyle="-",
            label="BP + Symmetric depolarizing",
        )

        plt.yscale("log")
        plt.xlabel("Depolarizing probability (p)")
        plt.ylabel("Average QBER")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()