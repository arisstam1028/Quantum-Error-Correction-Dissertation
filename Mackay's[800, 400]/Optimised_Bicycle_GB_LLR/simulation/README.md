# Simulation

This section runs Monte Carlo experiments over the selected QLDPC matrix and decoder.

## Files

### `metrics.py`

Functions:

- `frame_error_rate(failures, total_frames)`: returns `failures / total_frames`, with safe handling for zero frames.

### `runner.py`

Main simulation classes.

Functions/classes:

- `SingleRunResult`: stores one frame's success flag, iteration counts, true errors, estimated errors, residuals, and syndromes.
- `load_matrix_module(module_name)`: imports a matrix module and validates that it defines compatible `C`, `Hx`, and `Hz`.
- `QLDPCFamilyRunner`: discovers and runs a family of matrix modules.
- `QLDPCFamilyRunner.discover_modules()`: finds files matching `*_m<number>.py`.
- `QLDPCFamilyRunner._label_from_module(module_path)`: creates labels such as `m=1`.
- `QLDPCFamilyRunner.run_family()`: runs Monte Carlo for every discovered family member.
- `QLDPCRunner`: loads matrices, constructs channel and decoders, and runs individual frames.
- `QLDPCRunner.run_single_frame(p)`: samples an error, computes syndrome, decodes X and Z components separately, forms residuals, and checks row-space success.

### `monte_carlo.py`

Monte Carlo loop.

Functions:

- `run_monte_carlo(config)`: iterates over physical error probabilities, runs frames until frame or failure limits are reached, records failures, FER, and average iterations.

## Success Criterion

The decoder does not need to return the exact physical error. It succeeds when the residual error is in the stabilizer row space:

```text
ex_res in Row(Hx)
ez_res in Row(Hz)
```

That means the residual is stabilizer-equivalent to identity and causes no logical failure.

