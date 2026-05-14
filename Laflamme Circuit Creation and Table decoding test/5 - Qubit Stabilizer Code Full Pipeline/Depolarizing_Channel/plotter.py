from __future__ import annotations

from typing import Sequence
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


class ChannelPlotter:
    """
    Plotting utilities for the depolarizing-channel simulations.
    """

    @staticmethod
    def plot_error_count_graph(
        probabilities: Sequence[float],
        x_counts: Sequence[int],
        y_counts: Sequence[int],
        z_counts: Sequence[int],
        total_counts: Sequence[int],
        *,
        title: str = "Depolarizing_Channel Observed Error Counts",
    ) -> None:
        """
        Plot separate lines for observed X, Y, Z, and total error counts.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(probabilities, x_counts, marker="o", label="X errors")
        plt.plot(probabilities, y_counts, marker="^", label="Y errors")
        plt.plot(probabilities, z_counts, marker="s", label="Z errors")
        plt.plot(probabilities, total_counts, marker="d", label="Total errors")

        plt.xlabel("Physical error probability p")
        plt.ylabel("Observed error count")
        plt.title(title)
        plt.xlim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    @staticmethod
    def plot_encoder_circuit(
        qc: QuantumCircuit,
        *,
        title: str = "Encoder Circuit",
    ) -> None:
        """
        Plot the encoder circuit in a separate matplotlib figure.
        """
        fig = qc.draw(output="mpl", fold=-1, idle_wires=False)
        try:
            fig.suptitle(title)
            fig.tight_layout()
        except Exception:
            pass

    @classmethod
    def show_plots(
        cls,
        qc: QuantumCircuit,
        probabilities: Sequence[float],
        x_counts: Sequence[int],
        y_counts: Sequence[int],
        z_counts: Sequence[int],
        total_counts: Sequence[int],
        *,
        circuit_title: str = "Encoder Circuit",
        graph_title: str = "Depolarizing_Channel Observed Error Counts",
    ) -> None:
        """
        Create two separate figures:
        1) the encoder circuit
        2) the probability-vs-observed-error-count graph
        """
        cls.plot_encoder_circuit(qc, title=circuit_title)
        cls.plot_error_count_graph(
            probabilities,
            x_counts,
            y_counts,
            z_counts,
            total_counts,
            title=graph_title,
        )
        plt.show()