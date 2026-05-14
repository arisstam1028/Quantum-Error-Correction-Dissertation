from __future__ import annotations
from qiskit import QuantumCircuit
from basis_encoder import BasisEncoder, CircuitPlotter

def main() -> None:
    encoder = BasisEncoder()
    plotter = CircuitPlotter()

    # Choose input mode:
    mode = "bits"   # "bits" or "int"

    if mode == "bits":
        bits = [1, 0, 1, 1]
        n = len(bits)
        qc = QuantumCircuit(n, n)

        encoder.encode_bits(qc, list(range(n)), bits)
        encoder.print_encoding_summary_bits(bits)

        plotter.show(qc, title="Basis Encoding (state preparation)")

        qc.measure(range(n), range(n))
        plotter.show(qc, title="Basis Encoding + Measurements")

    elif mode == "int":
        n = 4
        value = 13
        little_endian = True

        qc = QuantumCircuit(n, n)
        bits_used = encoder.encode_int(qc, list(range(n)), value, little_endian=little_endian)
        encoder.print_encoding_summary_int(value, bits_used, little_endian=little_endian)

        plotter.show(qc, title="Basis Encoding (state preparation)")

        qc.measure(range(n), range(n))
        plotter.show(qc, title="Basis Encoding + Measurements")

    else:
        raise ValueError("mode must be 'bits' or 'int'")


if __name__ == "__main__":
    main()