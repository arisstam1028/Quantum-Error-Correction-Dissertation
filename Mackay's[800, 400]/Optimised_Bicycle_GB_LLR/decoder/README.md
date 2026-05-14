# decoder Package

This package implements a **Belief Propagation (BP) decoder** for quantum LDPC (QLDPC) codes.

The implementation is based on **Algorithm 1: Syndrome-Based BP** from the QLDPC literature, with practical adaptations for efficient simulation.

---

# Overview

The decoder operates on a **factor (Tanner) graph** derived from the parity-check matrices and performs **iterative message passing** between:

- **Variable nodes** → qubits (error variables)  
- **Check nodes** → stabilizers (parity constraints)

At each iteration, the algorithm:

1. Updates **check → variable messages**
2. Updates **variable → check messages**
3. Computes **posterior beliefs**
4. Produces a **hard decision**
5. Verifies **syndrome consistency**

This process repeats until convergence or a maximum number of iterations is reached.

---

# Relation to the Paper (Algorithm 1)

The original paper describes BP using:
H = (H_Z | H_X)


and performs decoding on a **single 2n-dimensional binary vector**.

---

## What is the same

This implementation follows Algorithm 1 in structure:

- ✔ Channel prior initialization  
- ✔ Iterative message passing  
- ✔ Check-node updates  
- ✔ Variable-node updates  
- ✔ Posterior computation  
- ✔ Hard decision via argmax  
- ✔ Syndrome consistency check for convergence  

Conceptually, this is the **same belief propagation algorithm** described in the paper.

---

## What is different

### 1. CSS decomposition (major difference)

Instead of decoding a single 2n-bit vector:

- **X errors are decoded using H_Z**  
- **Z errors are decoded using H_X**  

This results in **two independent binary BP decoders** rather than one joint decoder.

> This is standard in practice and significantly reduces complexity.

---

### 2. Efficient check-node update

The paper defines check-node updates as:
m_{c → v}^a = sum over valid configurations of product of incoming messages

This implementation instead uses the **closed-form binary parity update**:
m0 = 1/2 (1 + sign × product(q0 - q1))
m1 = 1/2 (1 - sign × product(q0 - q1))

> This is mathematically equivalent but computationally efficient.

---

### 3. Binary channel approximation

The depolarizing channel is approximated as:
P(1) = 2p / 3
P(0) = 1 - 2p / 3


> This simplifies decoding to a binary BP problem.

---

### 4. Loopy BP (not tree-based)

The algorithm is applied to **loopy Tanner graphs**, not trees.

> Therefore:
> - convergence is not guaranteed
> - performance depends on graph structure

---

# File Structure

## bp_decoder.py

### Role
Implements the **core BP algorithm (Algorithm 1)**.

### Correspondence to the paper

| Paper step | Code responsibility |
|-----------|-------------------|
| Step 1 | Initialize channel probabilities |
| Step 2 | Initialize variable → check messages |
| Step 3–10 | Iterative message passing |
| Step 5 | Check → variable update |
| Step 8 | Variable → check update |
| Step 11–13 | Posterior computation |
| Step 14 | Hard decision (argmax) |
| Step 16–18 | Syndrome check + convergence |

### Summary

This file is the **direct implementation of Algorithm 1**, adapted to:

- binary errors  
- CSS decoding  
- efficient message updates  

---

## bp_graph.py

### Role
Constructs the **Tanner graph** from a parity-check matrix.

### Functionality

- Converts matrix → adjacency lists  
- Stores:
  - variable → check connections  
  - check → variable connections  

### Correspondence to the paper

The paper assumes a factor graph implicitly.

This file makes that graph **explicit and computable**.

---

## decoder_result.py

### Role
Stores the output of the decoder.

### Fields

- estimated error vector  
- number of iterations  
- convergence status  

### Purpose

Separates **algorithm logic** from **result handling**.

---

# Summary of Implementation

| Feature | Paper | This implementation |
|--------|------|--------------------|
| Representation | full 2n vector | CSS split (X/Z separate) |
| Check update | summation | closed-form binary |
| Channel | depolarizing | binary approximation |
| Graph | implicit | explicit Tanner graph |
| Decoder type | BP | BP (loopy, iterative) |

---

# Key Takeaway

> This implementation is a **practical, efficient realization of Algorithm 1**, preserving its core message-passing structure while adapting it to:
>
> - CSS codes  
> - binary decoding  
> - computational efficiency  

---