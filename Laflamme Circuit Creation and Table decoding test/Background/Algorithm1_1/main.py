# main.py
# Example entry point kept separate from Algorithm1.py

from Algorithm1 import StabilizerEncoder, HsPrinter, CircuitPlotter
from verify_encoder_v2 import verify_stabilizer_span_algorithm1
from cz_audit import audit_cz
from simplify_encoder_v3 import EncoderSimplifier, SimplifiedCircuitPlotter
from qiskit import transpile

Hs = [
    [1, 0, 0, 0, 1,   1, 1, 0, 1, 1],
    [0, 1, 0, 0, 1,   0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1,   1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1,   1, 0, 1, 1, 1],
]

logical_X = [0, 0, 0, 0, 1,  1, 0, 0, 1, 0]

encoder = StabilizerEncoder(Hs, logical_X, name="encoder", cy_as_native=True, cz_as_native=True)
HsPrinter.print_all(encoder)

qc = encoder.build()

audit_cz(Hs, logical_X, qc)
verify_stabilizer_span_algorithm1(Hs, qc)

simplifier = EncoderSimplifier(Hs, do_semantic_cz_prune=True)
qc_simpl = simplifier.simplify(qc)

print("Ops original  :", qc.count_ops())
print("Ops simplified:", qc_simpl.count_ops())

print("\nCircuit (text):\n")
print(qc.draw(output="text"))

# One window, side-by-side comparison
SimplifiedCircuitPlotter.show_side_by_side(qc, qc_simpl)