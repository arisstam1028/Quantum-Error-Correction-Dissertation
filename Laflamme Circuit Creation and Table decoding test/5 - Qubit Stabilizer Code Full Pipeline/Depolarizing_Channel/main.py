from __future__ import annotations

import sys
from pathlib import Path

from depolarizing_channel import DepolarizingChannel
from plotter import ChannelPlotter


def import_encoder_builder():
    """
    Make the 'Encoding_Circuit_Builder' folder importable, then import
    the reusable encoder-builder function bundle from its main.py.
    """
    current_dir = Path(__file__).resolve().parent
    five_qubit_dir = current_dir.parent
    encoder_dir = five_qubit_dir / "Encoding_Circuit_Builder"

    if str(encoder_dir) not in sys.path:
        sys.path.insert(0, str(encoder_dir))

    from main import build_five_qubit_encoder_bundle  # type: ignore
    return build_five_qubit_encoder_bundle


def simulate_observed_error_counts(
    qc,
    probabilities,
    num_trials: int = 1000,
    seed: int = 42,
):
    """
    For each physical error probability p, run many depolarizing-channel trials
    and count the actual observed X, Y, Z, and total errors.

    Returns raw counts, not normalized rates.
    """
    physical_qubits = list(range(qc.num_qubits))

    x_counts_all = []
    y_counts_all = []
    z_counts_all = []
    total_counts_all = []

    for p_index, p in enumerate(probabilities):
        x_count = 0
        y_count = 0
        z_count = 0
        total_count = 0

        channel = DepolarizingChannel(p=p, seed=seed + p_index)

        for _ in range(num_trials):
            _, pattern, _ = channel.apply_random(
                qc,
                qubits=physical_qubits,
                inplace=False,
                barrier=False,
            )

            for pauli in pattern:
                if pauli == "X":
                    x_count += 1
                    total_count += 1
                elif pauli == "Y":
                    y_count += 1
                    total_count += 1
                elif pauli == "Z":
                    z_count += 1
                    total_count += 1

        x_counts_all.append(x_count)
        y_counts_all.append(y_count)
        z_counts_all.append(z_count)
        total_counts_all.append(total_count)

        print(f'p{p:.1f} | X_count{x_count}, Y_count{y_count}, Z_count{z_count}, Total_count{total_count}')

    return x_counts_all, y_counts_all, z_counts_all, total_counts_all


def main() -> None:
    build_five_qubit_encoder_bundle = import_encoder_builder()

    bundle = build_five_qubit_encoder_bundle(name="five_qubit_encoder")

    qc_simplified = bundle["simplified_circuit"]
    qc_to_transmit = qc_simplified

    probabilities = [i / 10 for i in range(11)]
    num_trials = 1000

    print("Circuit being transmitted:")
    print(qc_to_transmit.draw(output="text"))

    print("\nRunning depolarizing-channel simulation...\n")
    x_counts, y_counts, z_counts, total_counts = simulate_observed_error_counts(
        qc=qc_to_transmit,
        probabilities=probabilities,
        num_trials=num_trials,
        seed=42,
    )

    print("\nPlotting circuit and probability-vs-observed-error-count graph...")
    ChannelPlotter.show_plots(
        qc=qc_to_transmit,
        probabilities=probabilities,
        x_counts=x_counts,
        y_counts=y_counts,
        z_counts=z_counts,
        total_counts=total_counts,
        circuit_title="Simplified 5-Qubit Encoder Circuit",
        graph_title="Depolarizing_Channel: Probability vs Observed Error Count",
    )


if __name__ == "__main__":
    main()