# main.py

from Algorithm1 import StabilizerEncoder, HsPrinter, CircuitPlotter

# ----- Example from paper -----
Hs = [
    [1, 0, 0, 0, 1,   1, 1, 0, 1, 1],
    [0, 1, 0, 0, 1,   0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1,   1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1,   1, 0, 1, 1, 1],
]

logical_X = [0, 0, 0, 0, 1,  1, 0, 0, 1, 0]

encoder = StabilizerEncoder(Hs, logical_X)

# Print matrices and parameters
HsPrinter.print_all(encoder)

# Build circuit
qc = encoder.build()

print("\nText Circuit:\n")
print(qc.draw(output="text"))

# Open circuit in separate window
CircuitPlotter.show(qc)