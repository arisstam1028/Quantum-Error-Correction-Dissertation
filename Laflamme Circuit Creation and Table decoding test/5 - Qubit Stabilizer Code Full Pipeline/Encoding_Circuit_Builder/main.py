# Purpose:
#   Provide the five-qubit stabilizer data and build encoder circuits.
#
# Process:
#   1. Define the standard-form stabilizer matrix Hs = [Hx | Hz].
#   2. Define the logical X operator used for the encoded qubit.
#   3. Build raw and simplified encoder circuits for demonstrations.
#
# Theory link:
#   The stabilizer matrix and logical operator are the algebraic input
#   to Algorithm 1. The resulting circuit prepares the five-qubit
#   stabilizer codespace from binary symplectic data.
from __future__ import annotations

from qiskit import QuantumCircuit

from .Algorithm1 import StabilizerEncoder, HsPrinter
from .verify_encoder_v2 import verify_stabilizer_span_algorithm1
from .cz_audit import audit_cz
from .simplify_encoder_v3 import EncoderSimplifier, SimplifiedCircuitPlotter


def get_five_qubit_data() -> tuple[list[list[int]], list[int]]:
    """
    Return the standard-form Hs matrix and logical_X vector
    for the 5-qubit example.

    Role in pipeline:
        Supplies the binary symplectic input Hs = [Hx | Hz] and X_bar
        used by the encoder circuit builder.
    """
    Hs = [
        [1, 0, 0, 0, 1,   1, 1, 0, 1, 1],
        [0, 1, 0, 0, 1,   0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1,   1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1,   1, 0, 1, 1, 1],
    ]

    logical_X = [0, 0, 0, 0, 1,  1, 0, 0, 1, 0]
    return Hs, logical_X


def build_five_qubit_encoder(
    *,
    name: str = "encoder",
    cy_as_native: bool = True,
    cz_as_native: bool = True,
) -> QuantumCircuit:
    """
    Build the unsimplified five-qubit encoder circuit.

    Role in pipeline:
        Instantiates Algorithm 1 for the dissertation's small stabilizer
        code example.
    """
    Hs, logical_X = get_five_qubit_data()

    encoder = StabilizerEncoder(
        Hs,
        logical_X,
        name=name,
        cy_as_native=cy_as_native,
        cz_as_native=cz_as_native,
    )
    return encoder.build()


def build_five_qubit_simplified_encoder(
    *,
    name: str = "encoder",
    cy_as_native: bool = True,
    cz_as_native: bool = True,
    do_semantic_cz_prune: bool = True,
) -> QuantumCircuit:
    """
    Build and simplify the five-qubit encoder circuit.

    Role in pipeline:
        Removes gates that are redundant under the stabilizer-span
        verification, leaving a smaller circuit with unchanged encoding
        behaviour.
    """
    Hs, logical_X = get_five_qubit_data()

    encoder = StabilizerEncoder(
        Hs,
        logical_X,
        name=name,
        cy_as_native=cy_as_native,
        cz_as_native=cz_as_native,
    )
    qc = encoder.build()

    simplifier = EncoderSimplifier(Hs, do_semantic_cz_prune=do_semantic_cz_prune)
    qc_simpl = simplifier.simplify(qc)
    return qc_simpl


def build_five_qubit_encoder_bundle(
    *,
    name: str = "encoder",
    cy_as_native: bool = True,
    cz_as_native: bool = True,
    do_semantic_cz_prune: bool = True,
):
    """
    Return all useful objects for downstream code.

    Role in pipeline:
        Returns Hs, logical X, encoder metadata, and circuits so syndrome
        measurement and table decoding use the same stabilizer basis.
    """
    Hs, logical_X = get_five_qubit_data()

    encoder = StabilizerEncoder(
        Hs,
        logical_X,
        name=name,
        cy_as_native=cy_as_native,
        cz_as_native=cz_as_native,
    )
    qc = encoder.build()

    simplifier = EncoderSimplifier(Hs, do_semantic_cz_prune=do_semantic_cz_prune)
    qc_simpl = simplifier.simplify(qc)

    return {
        "Hs": Hs,
        "logical_X": logical_X,
        "encoder": encoder,
        "original_circuit": qc,
        "simplified_circuit": qc_simpl,
    }


def main() -> None:
    bundle = build_five_qubit_encoder_bundle()

    Hs = bundle["Hs"]
    logical_X = bundle["logical_X"]
    encoder = bundle["encoder"]
    qc = bundle["original_circuit"]
    qc_simpl = bundle["simplified_circuit"]

    HsPrinter.print_all(encoder)

    audit_cz(Hs, logical_X, qc)
    verify_stabilizer_span_algorithm1(Hs, qc)

    print("Ops original  :", qc.count_ops())
    print("Ops simplified:", qc_simpl.count_ops())

    print("\nOriginal circuit:\n")
    print(qc.draw(output="text"))

    print("\nSimplified circuit:\n")
    print(qc_simpl.draw(output="text"))

    SimplifiedCircuitPlotter.show_side_by_side(qc, qc_simpl)


if __name__ == "__main__":
    main()
