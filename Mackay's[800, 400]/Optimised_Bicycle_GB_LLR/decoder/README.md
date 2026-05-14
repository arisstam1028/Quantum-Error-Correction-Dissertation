# Decoder

This section implements the binary belief-propagation decoder used by the MacKay QLDPC simulation.

## Files

### `bp_graph.py`

Builds a sparse Tanner graph representation.

Classes/functions:

- `BPGraph`: stores check-to-variable lists, variable-to-check lists, edge arrays, pointer arrays, and edge lookup tables.
- `build_bp_graph(H)`: converts a binary parity-check matrix into check-major and variable-major edge layouts.

### `bp_decoder.py`

Flooding binary BP decoder in the LLR domain, with optional random perturbation.

Functions/classes:

- `_prefix_suffix_products(values)`: returns products excluding each position, used for check-node updates.
- `_parity_from_check_edges(estimate, edge_var, check_edge_ptr)`: computes the predicted syndrome from a hard estimate.
- `BinaryBPDecoder`: dataclass storing `H`, BP parameters, graph arrays, and perturbation settings.
- `BinaryBPDecoder._channel_llr(p_error)`: converts a binary error probability into an LLR.
- `BinaryBPDecoder._base_prior_llr(p_error)`: builds the starting prior vector.
- `BinaryBPDecoder._perturb_prior_llr(...)`: applies random perturbation to variables attached to one frustrated check.
- `BinaryBPDecoder._run_bp_phase(...)`: runs one BP phase from a supplied prior.
- `BinaryBPDecoder.decode(syndrome, p_error)`: runs initial BP and, if enabled, repeated perturbation phases.

### `decoder_result.py`

Result container.

Classes:

- `DecoderResult`: stores estimated error, success, iterations used, residual syndrome, and convergence flag.

## Random Perturbation

When ordinary BP fails, random perturbation selects a frustrated check and slightly increases the prior probability of error on variables attached to that check. BP is then restarted with the perturbed prior. This is a binary analogue of perturbative decoding strategies used to help BP escape trapping sets.

