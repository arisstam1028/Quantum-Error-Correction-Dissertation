import matplotlib.pyplot as plt


class StabilizerCircuitPlotter:

    @staticmethod
    def show_metric_curves(
        probabilities,
        z_basis_qber,
        x_basis_qber,
        average_qber,
        logical_failure_rates,
    ):
        plt.figure(figsize=(7, 5))

        # ONLY plot average QBER
        plt.plot(
            probabilities,
            average_qber,
            marker="o",
            linestyle="-",
            label="5-qubit Laflamme code",
        )

        plt.yscale("log")

        plt.xlabel("Depolarizing probability (p)")
        plt.ylabel("QBER")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()

        plt.tight_layout()
        plt.show()