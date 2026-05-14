from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


class StabilizerCircuitPlotter:
    """
    Plotting utilities for stabilizer-measurement circuits.
    """

    @staticmethod
    def plot_circuit(
        qc: QuantumCircuit,
        *,
        title: str = "Quantum Circuit",
    ) -> None:
        """
        Plot a single circuit in its own matplotlib figure.
        """
        fig = qc.draw(output="mpl", fold=-1, idle_wires=False)
        try:
            fig.suptitle(title)
            fig.tight_layout()
        except Exception:
            pass

    @classmethod
    def show_measurement_circuits(
        cls,
        measurement_only: QuantumCircuit,
        encoded_plus_measurement: QuantumCircuit,
        sample_error_circuit: Optional[QuantumCircuit] = None,
    ) -> None:
        """
        Show the available stabilizer-measurement circuits in separate windows.
        """
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

        plt.show()