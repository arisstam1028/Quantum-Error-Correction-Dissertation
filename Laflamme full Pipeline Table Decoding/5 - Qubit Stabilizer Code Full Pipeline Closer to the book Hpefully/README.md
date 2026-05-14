# Five-Qubit Table Decoding: Symmetric Depolarizing

This folder is the exact symmetric-depolarizing version of the five-qubit table-decoding simulator.

## Difference From `QBER DepoS`

The core structure is the same: binary symplectic errors, algebraic syndrome computation, table decoding, and residual-error metrics. The important difference is the channel model:

```text
I with probability 1-p
X with probability p/3
Y with probability p/3
Z with probability p/3
```

This preserves the X/Z correlation introduced by Y errors, unlike the independent-BSC approximation.

## Files

### `main.py`

Defines `main()`, which builds and runs the simulation.

### `simulation_runner.py`

Contains:

- `BinarySymplectic`: Pauli/binary conversion and XOR arithmetic.
- `FiveQubitCode`: stabilizer definitions.
- `SimulationConfig`: probabilities, frames, seed, and failure threshold.
- `SimulationResults`: output metrics.
- `SimulationRunner`: Monte Carlo engine.
- `SimulationReport`: text and plot reporting.

The runner samples exact symmetric depolarizing errors, computes the syndrome, applies table correction, and counts a residual as failure if it is not the identity.

### `Depolarizing_Channel/depolarizing.py`

Defines `DepolarizingChannel.sample_error(n, p)`, which samples exact symmetric Pauli errors and returns `(ex, ez)`.

### `Table_Decoding_and_Error_Correction`

- `stabilizer_measurement.py`: builds `hx`, `hz`, and computes syndromes.
- `decoder.py`: builds the single-qubit syndrome table.
- `plotter.py`: plots error-rate metrics.

### `test_depolarizing.py`

Empirical channel-frequency check.

