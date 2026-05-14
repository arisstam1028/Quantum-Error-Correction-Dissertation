# MacKay Bicycle QLDPC Simulation Framework

This is the main QLDPC project. It constructs or loads bicycle CSS parity-check matrices, simulates either an independent BSC approximation or exact symmetric Pauli depolarizing noise, computes CSS syndromes, decodes with BP-style decoders, and plots FER and average iteration counts.

## Main Theory

The project uses CSS stabilizer codes. X-type checks detect Z components of an error, and Z-type checks detect X components:

```text
sX = Hx ez^T mod 2
sZ = Hz ex^T mod 2
```

The depolarizing channel samples Pauli errors:

```text
I with 1-p
X, Y, Z with p/3 each
```

The BSC approximation instead samples the X component and Z component independently with probability:

```text
p_component = 2p/3
```

For binary BP, each component is decoded with effective binary error probability:

```text
p_binary = 2p/3
```

The residual after decoding is:

```text
ex_res = ex_true xor ex_hat
ez_res = ez_true xor ez_hat
```

The run is counted as successful if the residual is in the stabilizer row space, meaning it is equivalent to a stabilizer and therefore not a logical failure.

## Main Files

### `main.py`

Top-level executable.

Functions:

- `main()`: chooses a matrix module, selects decoder type (`bp`, `svns`, `scns`, or `compare`), selects channel mode (`bsc`, `depolarizing`, or `both`), runs Monte Carlo simulation, prints results, and plots FER/iteration curves.
- `run_single_channel(config, channel_mode)`: applies the requested channel model before running the sweep.

Global controls:

- `DECODER_TYPE`: decoder selection.
- `RUN_FAMILY`: switches between one fixed matrix and a family sweep.
- `CHANNEL_MODE`: chooses the independent BSC approximation, exact depolarizing channel, or both for comparison.

### `config.py`

Configuration dataclass.

Classes/functions:

- `QLDPCConfig`: stores matrix module, BP parameters, channel toggle, probability sweep, frame counts, failure limits, and output options.
- `build_config(matrix_module="matrices.mackay_800_400")`: creates a config and validates that probability/frame/failure lists have matching length.

## Sections

- `channel`: BSC-approximation and exact depolarizing channel sampler.
- `code_construction`: bicycle/circulant construction and code statistics.
- `core`: GF(2), Pauli, CSS, and syndrome helpers.
- `decoder`: binary BP graph and decoder.
- `Sequential_BP_Based_Decoding`: SVNS and SCNS decoder variants.
- `matrices`: fixed matrix modules, including the MacKay `[800,400]` code.
- `simulation`: Monte Carlo runner.
- `plotting`: result plotting.
- `tests`: lightweight correctness tests.

## Main MacKay Matrix

`matrices.mackay_800_400` defines:

```text
C shape        = (400, 400)
Hx, Hz shape   = (200, 800)
N              = 800
K              = 400
rank           = 200
row weight     = 30
column weights = 5 to 10
Hx Hz^T mod 2  = 0
```

The module sets `Hx = H` and `Hz = H`, so the CSS code uses the same bicycle check matrix for X and Z checks.
