# Sequential BP Based Decoding

This section contains sequential scheduling variants of binary BP.

## Files

### `svns_decoder.py`

Sequential Variable Node Scheduling (SVNS).

Functions/classes:

- `_prefix_suffix_products(values)`: products excluding each edge.
- `_parity_from_check_edges(estimate, edge_var, check_edge_ptr)`: predicted syndrome helper.
- `SVNSBPDecoder`: BP decoder that updates variable nodes sequentially.
- `SVNSBPDecoder._channel_llr(p_error)`: probability-to-LLR conversion.
- `SVNSBPDecoder.decode(syndrome, p_error)`: for each iteration, visits variable nodes in order, refreshes incoming check messages for that variable, updates the bit LLR, and updates outgoing variable messages.

### `scns_decoder.py`

Sequential Check Node Scheduling (SCNS).

Functions/classes:

- `_prefix_suffix_products(values)`: products excluding each edge.
- `_parity_from_check_edges(estimate, edge_var, check_edge_ptr)`: predicted syndrome helper.
- `SCNSBPDecoder`: BP decoder that updates check nodes sequentially.
- `SCNSBPDecoder._channel_llr(p_error)`: probability-to-LLR conversion.
- `SCNSBPDecoder.decode(syndrome, p_error)`: updates check nodes in sequence and refreshes beliefs after each schedule step.

## Theory

Flooding BP updates all checks and variables in separate global phases. Sequential schedules update parts of the graph immediately, sometimes improving convergence on loopy LDPC/QLDPC Tanner graphs.

