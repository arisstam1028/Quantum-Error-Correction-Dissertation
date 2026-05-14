# MacKay `[800,400]` Project

This directory contains the main dissertation QLDPC project plus supporting
scripts for generating and plotting MacKay bicycle-code results.

## Project

### `Optimised_Bicycle_GB_LLR`

The active simulation framework. It loads the fixed MacKay bicycle matrix,
samples either an independent BSC approximation or exact symmetric
depolarizing Pauli errors, computes CSS syndromes, decodes using BP or
sequential BP variants, and plots FER/iteration results.

The current active runner supports three channel modes:

- `bsc`: independent X/Z binary symmetric approximation with marginal probability `2p/3`.
- `depolarizing`: exact symmetric quantum depolarizing channel with `X`, `Y`, and `Z` each occurring with probability `p/3`.
- `both`: runs both channel models for direct plotting comparison.

The active decoder choices are:

- `bp`: optimized flooding-schedule binary BP.
- `svns`: sequential variable-node scheduling BP.
- `scns`: sequential check-node scheduling BP.
- `compare`: compare decoder types where supported.

See:

- `Optimised_Bicycle_GB_LLR/README.md` for the full project overview.
- `Optimised_Bicycle_GB_LLR/matrices/README.md` for the fixed MacKay matrix and smaller comparison matrices.
- `Optimised_Bicycle_GB_LLR/decoder/README.md` for BP and perturbation details.
- `Optimised_Bicycle_GB_LLR/simulation/README.md` for the Monte Carlo success/failure criterion.

### `Mackays Codes Generator`

Standalone scripts for producing MacKay-style bicycle matrices.

- `Mackays code and family generator.py` builds smaller construction-B bicycle examples and optional family-style candidates.
- `Mackays_half_rate_800_400_code_w30.py` searches for a half-rate `[800,400]` matrix with row weight 30, target rank 200, and `Hx = Hz = H`.

These scripts are generation tools rather than the main simulation entry point.
The generated fixed matrix is consumed by `Optimised_Bicycle_GB_LLR/matrices/mackay_800_400.py`.

### `Plots.py`

Replots stored comparison data for:

- random perturbation results,
- independent BSC approximation results,
- exact symmetric depolarizing-channel results.

## Theory Summary

The MacKay code is represented as a CSS stabilizer code. The simulation decodes X and Z error components separately:

```text
sX = Hx ez^T mod 2
sZ = Hz ex^T mod 2
```

The decoder succeeds when the residual error is stabilizer-equivalent to identity, checked through row-space membership.

The BSC approximation samples X and Z components independently with probability `2p/3`. The exact depolarizing model samples a single Pauli per qubit:

```text
I with 1-p
X, Y, Z with p/3 each
```

A `Y` error contributes to both binary components in symplectic form.
