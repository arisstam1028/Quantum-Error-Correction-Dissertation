# test_depolarizing.py

from collections import Counter

from Depolarizing_Channel.depolarizing import DepolarizingChannel


def binary_pair_to_pauli(ex: int, ez: int) -> str:
    if ex == 0 and ez == 0:
        return "I"
    if ex == 1 and ez == 0:
        return "X"
    if ex == 0 and ez == 1:
        return "Z"
    if ex == 1 and ez == 1:
        return "Y"
    raise ValueError(f"Invalid bits ({ex}, {ez})")


def test_depolarizing_channel(p: float, n_qubits: int = 10000, seed: int = 42) -> None:
    channel = DepolarizingChannel(seed=seed)
    counts = Counter()

    for _ in range(n_qubits):
        ex, ez = channel.sample_error(1, p)
        pauli = binary_pair_to_pauli(int(ex[0]), int(ez[0]))
        counts[pauli] += 1

    i_count = counts.get("I", 0)
    x_count = counts.get("X", 0)
    y_count = counts.get("Y", 0)
    z_count = counts.get("Z", 0)

    print("\n=== Depolarizing Channel Test ===")
    print(f"Samples: {n_qubits}, p = {p}\n")

    print("Counts:")
    print(f"I: {i_count}")
    print(f"X: {x_count}")
    print(f"Y: {y_count}")
    print(f"Z: {z_count}\n")

    print("Observed probabilities:")
    print(f"I: {i_count / n_qubits:.6f}")
    print(f"X: {x_count / n_qubits:.6f}")
    print(f"Y: {y_count / n_qubits:.6f}")
    print(f"Z: {z_count / n_qubits:.6f}\n")

    print("Expected:")
    print(f"I: {1 - p:.6f}")
    print(f"X: {p / 3:.6f}")
    print(f"Y: {p / 3:.6f}")
    print(f"Z: {p / 3:.6f}")
    print("=================================\n")


if __name__ == "__main__":
    test_depolarizing_channel(p=0.1)