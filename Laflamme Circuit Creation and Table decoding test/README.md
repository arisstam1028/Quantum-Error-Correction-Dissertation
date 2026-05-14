# Laflamme Five-Qubit Stabilizer Project

This project builds and tests a small quantum error-correcting code pipeline based on the five-qubit stabilizer code and the encoder-construction method from the stabilizer-circuit tutorial paper.

The project has three layers:

- `Background`: exploratory scripts for Algorithm 1, basis encoders, logical operators, and small bicycle CSS codes.
- `5 - Qubit Stabilizer Code Split`: the same work divided into independent folders while the pipeline was being developed.
- `5 - Qubit Stabilizer Code Full Pipeline`: the integrated version that builds the encoder, derives stabilizers, constructs measurement circuits, builds a table decoder, simulates errors, and plots logical error rate.

## Theory

The code uses the binary symplectic representation of Pauli operators:

```text
I = (0,0)
X = (1,0)
Z = (0,1)
Y = (1,1)
```

A stabilizer row is written as `[x | z]`. Two Pauli rows commute when their symplectic product is zero:

```text
x . z' + z . x' = 0 mod 2
```

The encoder builder uses a standard-form stabilizer matrix `Hs = [X | Z]` and a logical-X vector. The decoder is a syndrome lookup table: every zero/single-qubit Pauli error is assigned to its syndrome.

## Main Result Path

Use the integrated project:

```text
5 - Qubit Stabilizer Code Full Pipeline/main.py
```

That entrypoint can run a single demonstration or a vectorized Monte Carlo sweep. The later `Laflamme full Pipeline Table Decoding` and `Laflamme full Pipeline BP Decoding` projects simplify this further by removing circuit simulation from the Monte Carlo loop.

