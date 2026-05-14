# simplified_encoder.py
# Simplify an existing encoder circuit (without regenerating it) and plot it.
#
# Requires: qiskit, matplotlib

from __future__ import annotations

from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile


@dataclass(frozen=True)
class SimplifySpec:
    optimization_level: int = 3
    # Keep CY/CZ visible (avoid hardware-basis expansion)
    basis_gates: tuple[str, ...] = ("h", "s", "sdg", "cx", "cz", "cy")


class EncoderSimplifier:
    """
    Takes an existing QuantumCircuit (encoder) and returns a simplified version
    using Qiskit's compiler optimizations while preserving CY/CZ in the basis.
    """

    def __init__(
        self,
        *,
        optimization_level: int = 3,
        basis_gates: tuple[str, ...] = ("h", "s", "sdg", "cx", "cz", "cy"),
    ):
        self._spec = SimplifySpec(optimization_level=optimization_level, basis_gates=basis_gates)

    @property
    def spec(self) -> SimplifySpec:
        return self._spec

    def simplify(self, qc: QuantumCircuit, *, new_name: str | None = None) -> QuantumCircuit:
        """
        Simplify the given circuit and return a new circuit.
        """
        out = transpile(
            qc,
            basis_gates=list(self._spec.basis_gates),
            optimization_level=self._spec.optimization_level,
        )
        out.name = new_name or (qc.name + "_simplified" if qc.name else "simplified")
        return out


class SimplifiedCircuitPlotter:
    """
    Plot circuits in a separate window using matplotlib.
    Supports plotting one circuit or (optionally) two circuits in one window.
    """

    @staticmethod
    def show(qc: QuantumCircuit, *, fold: int = 80, block: bool = True) -> None:
        import matplotlib.pyplot as plt

        qc.draw(output="mpl", fold=fold)
        plt.show(block=block)

    @staticmethod
    def show_side_by_side(
        qc_left: QuantumCircuit,
        qc_right: QuantumCircuit,
        *,
        titles: tuple[str, str] = ("Original", "Simplified"),
        fold: int = 80,
        block: bool = True,
    ) -> None:
        """
        One window, two subplots. This is helpful for visual comparison.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title(titles[0])
        ax1.axis("off")
        qc_left.draw(output="mpl", fold=fold, ax=ax1)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(titles[1])
        ax2.axis("off")
        qc_right.draw(output="mpl", fold=fold, ax=ax2)

        plt.tight_layout()
        plt.show(block=block)