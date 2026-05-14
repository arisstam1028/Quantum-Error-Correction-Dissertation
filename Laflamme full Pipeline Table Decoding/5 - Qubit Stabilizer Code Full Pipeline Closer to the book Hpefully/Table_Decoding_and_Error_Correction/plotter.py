from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt


class StabilizerCircuitPlotter:

    @staticmethod
    def show_metric_curves(
        probabilities: Sequence[float],
        z_basis_qber: Sequence[float],
        x_basis_qber: Sequence[float],
        average_qber: Sequence[float],
        logical_failure_rates: Sequence[float] | None = None,
    ) -> None:

        plt.figure(figsize=(9, 6))

        # QBER curves (this is what the paper uses)
        plt.plot(probabilities, z_basis_qber, marker="s", label="QBER (Z basis)")
        plt.plot(probabilities, x_basis_qber, marker="^", label="QBER (X basis)")
        plt.plot(probabilities, average_qber, marker="d", label="Average QBER")

        # Optional: logical failure (for debugging / insight)
        if logical_failure_rates is not None:
            plt.plot(
                probabilities,
                logical_failure_rates,
                linestyle="--",
                marker="o",
                label="Logical failure rate",
            )

        plt.xlabel("Depolarizing probability p")
        plt.ylabel("Error rate")
        plt.title("5-Qubit Code: QBER vs Depolarizing Probability")

        plt.yscale("log")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.show()