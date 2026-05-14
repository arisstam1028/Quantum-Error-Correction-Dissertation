# Tests

This section contains lightweight correctness checks for the QLDPC framework.

## Files

### `test_bicycle_code.py`

Functions:

- `test_bicycle_commutation()`: verifies that constructed bicycle CSS matrices commute.
- `test_bicycle_dimensions()`: verifies expected matrix dimensions.

### `test_bp_decoder.py`

Functions:

- `test_bp_zero_syndrome()`: checks that BP succeeds on a zero-syndrome example.
- `test_bp_single_check()`: checks that BP recovers the only bit involved in a single unsatisfied check.

### `test_helpers.py`

Functions:

- `test_rowspace_checker_matches_in_rowspace()`: checks row-space membership for known combinations.

### `test_syndrome.py`

Functions:

- `test_css_syndrome_shapes()`: validates syndrome output shapes.
- `test_css_syndrome_values()`: checks known syndrome values.

## Purpose

These tests are not exhaustive performance validation. They protect the algebraic building blocks used by the Monte Carlo simulator: commutation, BP parity checks, row-space membership, and syndrome computation.

