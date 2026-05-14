# MacKay `[800,400]` Project

This directory contains the main dissertation QLDPC project.

## Project

### `Optimised_Bicycle_GB_LLR`

The active simulation framework. It loads the fixed MacKay bicycle matrix, samples depolarizing Pauli errors, computes CSS syndromes, decodes using BP or sequential BP variants, optionally applies random perturbation, and plots FER/iteration results.

See:

- `Optimised_Bicycle_GB_LLR/README.md` for the full project overview.
- `Optimised_Bicycle_GB_LLR/matrices/README.md` for the fixed MacKay matrix and smaller comparison matrices.
- `Optimised_Bicycle_GB_LLR/decoder/README.md` for BP and perturbation details.
- `Optimised_Bicycle_GB_LLR/simulation/README.md` for the Monte Carlo success/failure criterion.

## Theory Summary

The MacKay code is represented as a CSS stabilizer code. The simulation decodes X and Z error components separately:

```text
sX = Hx ez^T mod 2
sZ = Hz ex^T mod 2
```

The decoder succeeds when the residual error is stabilizer-equivalent to identity, checked through row-space membership.

