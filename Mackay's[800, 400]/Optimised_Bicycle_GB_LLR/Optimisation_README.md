# LLR-Based BP Decoder Optimization (QLDPC Project)

## Overview

This document describes the transition of the decoding framework from a **probability-domain belief propagation (BP)** implementation to an **LLR-domain (log-likelihood ratio) implementation**, along with structural optimizations to improve computational efficiency.

The goal of this refactor was to:

* Align the implementation with the algorithms presented in the reference papers
* Improve runtime performance for large-scale Monte Carlo simulations
* Preserve the original decoding functionality and structure

---

## 1. Key Conceptual Change: Probability Domain → LLR Domain

### Previous Implementation

The original decoders operated in the **probability domain**, maintaining:

* q0, q1 : variable-to-check messages
* r0, r1 : check-to-variable messages

Updates required:

* explicit multiplications over neighbourhoods
* normalization at every step
* repeated computation of exclusion products

---

### New Implementation

The updated decoders operate in the **LLR (log-likelihood ratio) domain**, defined as:

```
mu = log((1 - p) / p)
```

Messages are now:

* m_v→c : variable → check message
* m_c→v : check → variable message
* bitLLR_v : posterior log-likelihood ratio for variable v

---

### Flooding BP Equations

Check node update:

```
m_c→v = 2 * atanh( sigma_c * product over u≠v of tanh(m_u→c / 2) )
```

Variable node update:

```
bitLLR_v = mu + sum over c of m_c→v  
m_v→c    = bitLLR_v - m_c→v  
```

---

### Advantages of LLR Formulation

* Converts multiplications → additions
* Eliminates normalization steps
* Improves numerical stability
* Matches paper algorithms more closely
* Significantly faster for large graphs

---

## 2. Algorithm Mapping to Paper

The implementation now directly follows the structure of the reference algorithms.

---

### 2.1 Flooding BP

For each iteration:

1. Compute all check-to-variable messages (m_c→v)
2. Compute all variable beliefs (bitLLR_v)
3. Update all variable-to-check messages (m_v→c)
4. Perform hard decision and check syndrome

---

### 2.2 SVNS (Sequential Variable Node Scheduling)

For each variable node v:

1. Refresh all incoming messages m_c→v

2. Compute posterior:

   ```
   bitLLR_v = mu + sum over c of m_c→v  
   ```

3. Update outgoing messages:

   ```
   m_v→c = bitLLR_v - m_c→v  
   ```

---

### 2.3 SCNS (Sequential Check Node Scheduling)

For each check node c:

1. Update all outgoing messages m_c→v
2. For each neighbouring variable v:

   * recompute bitLLR_v
   * update only m_v→c for that edge

This matches the sequential structure described in the papers.

---

## 3. Structural Optimizations

---

### 3.1 Edge-Based Message Storage

#### Old

* Python dictionaries keyed by (v, c)

#### New

* Dense NumPy arrays indexed by edge ID

Example:

```
mv_to_c[e]
mc_to_v[e]
```

#### Benefit

* Eliminates dictionary lookup overhead
* Enables vectorized operations
* Improves memory locality

---

### 3.2 Flattened Graph Representation

The graph now stores:

* edge_var[e]   → variable index
* edge_check[e] → check index
* contiguous edge blocks for each node

#### Benefit

* Faster iteration over neighbours
* Removes Python list overhead in hot loops

---

### 3.3 Prefix/Suffix Products

Used in check updates to compute:

```
product over neighbours excluding one
```

#### Old

* Nested loops → O(d²)

#### New

* Prefix/suffix method → O(d)

#### Benefit

* Major speedup for higher-degree nodes

---

### 3.4 Removal of Probability Normalization

LLR domain removes:

* division operations
* explicit normalization
* repeated probability scaling

---

### 3.5 Clipping for Numerical Stability

To avoid overflow:

* LLR values clipped to range [-50, 50]
* tanh inputs implicitly bounded

---

## 4. Behaviour Preservation

The following aspects are unchanged:

* Decoder API: decode(syndrome, p_error)

* Output format: DecoderResult

* Hard decision rule:

  ```
  estimated_error[v] = 1 if bitLLR_v < 0 else 0  
  ```

* Syndrome-based stopping condition

* Maximum iteration count

* SVNS and SCNS scheduling order

* CSS decoding structure (X and Z separately)

* Monte Carlo simulation pipeline

---

## 5. Expected Performance Improvements

| Component           | Improvement                   |
| ------------------- | ----------------------------- |
| Message access      | High (no dictionaries)        |
| Check updates       | High (prefix/suffix products) |
| Variable updates    | Moderate                      |
| Sequential decoders | High                          |
| Large codes         | Very high                     |

---

## 6. Numerical Differences

Small differences may occur due to:

* LLR vs probability representation
* floating-point ordering differences

However:

* decoding behaviour remains equivalent
* convergence behaviour is consistent
* FER trends should match

---

## 7. Validation Checklist

After applying the update:

* Run existing unit tests
* Test small codes (e.g. bicycle_24)
* Compare:

  * convergence rate
  * FER vs p
  * runtime vs previous version

---

## 8. Runtime Measurement

A timer was added to main.py:

```python
import time

start = time.perf_counter()
main()
end = time.perf_counter()

print(f"Total runtime: {end - start:.2f} seconds")
```

---

## 9. Summary

This refactor:

* aligns the implementation with the literature
* significantly improves runtime performance
* preserves decoding structure and behaviour
* enables scalable QLDPC simulations

---
