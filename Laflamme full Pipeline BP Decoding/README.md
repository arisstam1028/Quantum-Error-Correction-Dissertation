# Laflamme Full Pipeline: BP Decoding

This project uses the same five-qubit binary symplectic simulation model as the table-decoding project, but replaces the table decoder with belief propagation.

## Theory

For a general stabilizer matrix split into `hx` and `hz`, the syndrome equation is:

```text
syndrome = hx * ez + hz * ex mod 2
```

The BP decoder creates one binary parity-check matrix over the combined error vector:

```text
e = [ex | ez]
H_bp = [hz | hx]
```

Then it runs binary sum-product BP in the LLR domain to estimate `e`.

## Main Folder

`5 - Qubit Stabilizer Code Full Pipeline QBER DepoS` compares:

- independent-BSC approximation
- exact symmetric depolarizing channel

using the same BP decoder.

