from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


class StabilizerCircuitPlotter:
    @staticmethod
    def plot_circuit(
        qc: QuantumCircuit,
        *,
        title: str = "Quantum Circuit",
    ) -> None:
        fig = qc.draw(output="mpl", fold=-1, idle_wires=False)
        try:
            fig.suptitle(title)
            fig.tight_layout()
        except Exception:
            pass

    @staticmethod
    def plot_error_rate_curve(
            probabilities,
            logical_error_rates,
            *,
            title: str = "Logical Error Rate vs Physical Error Probability",
    ):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(probabilities, logical_error_rates, marker="o")
        plt.xlabel("Physical error probability p")
        plt.ylabel("Logical error rate")
        plt.title(title)

        # Logarithmic BER axis
        plt.yscale("log")
        plt.grid(True, which="both")
        plt.tight_layout()

    @classmethod
    def show_measurement_circuits(
        cls,
        measurement_only: QuantumCircuit,
        encoded_plus_measurement: QuantumCircuit,
        sample_error_circuit: Optional[QuantumCircuit] = None,
        encoder_circuit: Optional[QuantumCircuit] = None,
        probabilities: Optional[Sequence[float]] = None,
        logical_error_rates: Optional[Sequence[float]] = None,
    ) -> None:
        if encoder_circuit is not None:
            cls.plot_circuit(
                encoder_circuit,
                title="Encoder Circuit",
            )

        cls.plot_circuit(
            measurement_only,
            title="General Stabilizer-Measurement Circuit",
        )

        cls.plot_circuit(
            encoded_plus_measurement,
            title="Encoded State + Stabilizer Measurement",
        )

        if sample_error_circuit is not None:
            cls.plot_circuit(
                sample_error_circuit,
                title="Encoded State + Error + Stabilizer Measurement",
            )

        if probabilities is not None and logical_error_rates is not None:
            cls.plot_error_rate_curve(
                probabilities,
                logical_error_rates,
            )

        plt.show()