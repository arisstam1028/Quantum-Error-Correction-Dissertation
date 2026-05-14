import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer


def shor_encode(alpha=1.0, beta=0.0):
    """
    Shor (9,1) quantum code encoder.
    Encodes |psi> = alpha|0> + beta|1> into 9 physical qubits.
    """

    # Normalize input state
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm

    qc = QuantumCircuit(9, name="Shor (9,1) Encoder")

    # --- Logical input state ---
    qc.initialize([alpha, beta], 0)

    # --- Phase-flip protection (3-block repetition in Hadamard basis) ---
    qc.h(0)
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h(0)
    qc.h(3)
    qc.h(6)

    # --- Bit-flip protection (within each block) ---
    # Block 1: qubits 0,1,2
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Block 2: qubits 3,4,5
    qc.cx(3, 4)
    qc.cx(3, 5)

    # Block 3: qubits 6,7,8
    qc.cx(6, 7)
    qc.cx(6, 8)

    return qc


# =========================
# Main execution
# =========================

if __name__ == "__main__":

    # Encode |+> = (|0> + |1>)/sqrt(2)
    qc = shor_encode(
        alpha=1 / np.sqrt(2),
        beta=1 / np.sqrt(2)
    )

    # Draw circuit in a NEW window (PyCharm)
    circuit_drawer(
        qc,
        output="mpl",
        fold=-1,
        scale=0.7
    )

    plt.show()
