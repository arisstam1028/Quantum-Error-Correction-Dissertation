# Laflamme Full Pipeline: Table Decoding

This project simulates the five-qubit stabilizer code without building or executing quantum circuits for each trial. It represents errors in binary symplectic form, computes syndromes algebraically, applies a syndrome-table decoder, and plots QBER/FER-style results.

There are two simulation folders:

- `5 - Qubit Stabilizer Code Full Pipeline QBER DepoS`: independent-BSC approximation, where X and Z binary components are sampled independently with probability `2p/3`.
- `5 - Qubit Stabilizer Code Full Pipeline Closer to the book Hpefully`: exact symmetric depolarizing channel, where `X`, `Y`, and `Z` each occur with probability `p/3`.

The top-level `compare_channels.py` imports both projects and plots their average QBER curves.

## Theory

The five-qubit code uses stabilizers:

```text
XZZXI
IXZZX
XIXZZ
ZXIXZ
```

Errors are represented by `(ex, ez)`. The syndrome is:

```text
syndrome_i = hx_i . ez + hz_i . ex mod 2
```

The table decoder stores corrections for the identity and all single-qubit `X`, `Y`, and `Z` errors. A residual error after correction is:

```text
rx = ex xor cx
rz = ez xor cz
```

The simulation counts a failure when the residual is not the identity.

## Top-Level File

### `compare_channels.py`

Compares the two table-decoding channel models.

Functions:

- `_clear_conflicting_modules()`: removes previously imported sibling modules from `sys.modules` so both similarly named projects can be loaded cleanly.
- `_load_simulation_runner(project_dir, alias)`: dynamically imports a selected `simulation_runner.py`.
- `_run_project(module, seed=7)`: builds the five-qubit code, runs the probability sweep, and returns results.
- `_print_summary(label, results)`: prints probability/QBER data.
- `main()`: runs both channel models, plots average QBER, and saves the comparison figure.

