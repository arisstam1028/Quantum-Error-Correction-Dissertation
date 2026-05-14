# Five-Qubit BP Decoding

This folder simulates the five-qubit code with belief propagation decoding.

## Files

### `main.py`

Entrypoint.

Functions:

- `main()`: builds the code, creates a simulation config, runs both channel models, prints summaries, and plots comparison curves.

### `simulation_runner.py`

Main BP simulation harness.

Classes:

- `BinarySymplectic`: Pauli/binary conversion and XOR arithmetic.
- `FiveQubitCode`: stores the four five-qubit stabilizers.
- `FiveQubitBPDecoder`: wraps a binary BP decoder around the stabilizer measurement matrix.
- `SimulationConfig`: probabilities, frame counts, seed, failure threshold, and BP iteration count.
- `ChannelResults`: metrics for one channel model.
- `ComparisonResults`: stores BSC and symmetric-channel results.
- `SimulationRunner`: runs both channel models.
- `SimulationReport`: prints and plots results.

Important methods:

- `FiveQubitBPDecoder.__init__(measurement, max_iters=30)`: builds `H_bp = [hz | hx]`.
- `FiveQubitBPDecoder.decode(syndrome, physical_p)`: converts syndrome string to a vector, uses prior `2p/3`, and splits the BP estimate back into `(ex_hat, ez_hat)`.
- `SimulationRunner._run_single_channel(...)`: runs one channel model.
- `SimulationRunner.run(config)`: runs both BSC approximation and symmetric depolarizing comparisons.

### `Depolarizing_Channel/depolarizing.py`

Channel model.

Classes:

- `DepolarizingChannel`: samples either independent-BSC component errors or exact symmetric Pauli errors.

### `Table_Decoding_and_Error_Correction/bp_graph.py`

Sparse Tanner graph representation.

Classes/functions:

- `BPGraph`: stores edge variables, check edge pointers, and adjacency.
- `build_bp_graph(H)`: converts a binary parity-check matrix into the graph arrays needed by BP.

### `Table_Decoding_and_Error_Correction/bp_decoder.py`

Binary BP decoder.

Functions/classes:

- `_prefix_suffix_products(values)`: computes products excluding each edge.
- `_parity_from_check_edges(estimate, edge_var, check_edge_ptr)`: computes predicted syndrome from a hard estimate.
- `BinaryBPDecoder`: LLR-domain flooding BP decoder.
- `BinaryBPDecoder._channel_llr(p_error)`: converts error probability to LLR.
- `BinaryBPDecoder.decode(syndrome, p_error)`: runs check-node and variable-node BP updates until the syndrome matches or max iterations are reached.

### `Table_Decoding_and_Error_Correction/decoder_result.py`

Defines `DecoderResult`, which stores estimated error, success flag, iterations used, residual syndrome, and convergence flag.

### `Table_Decoding_and_Error_Correction/stabilizer_measurement.py`

Binary stabilizer measurement helper. It parses stabilizers into `hx` and `hz` and computes syndrome strings.

### `Table_Decoding_and_Error_Correction/plotter.py`

Plotting helper for BP comparison curves.

### `test_depolarizing.py`

Channel sanity check with empirical Pauli frequencies.

