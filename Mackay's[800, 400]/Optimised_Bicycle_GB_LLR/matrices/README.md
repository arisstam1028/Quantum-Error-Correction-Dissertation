# matrices Package

This package stores **fixed parity-check matrices used in simulations**.

The matrices define the stabilizer structure of the QLDPC code.

---

# Files

### bicycle_24_12.py

Generator script.

Purpose:

- Generates a **bicycle QLDPC code** with parameters:

n = 24 physical qubits  
m = 12 circulant size

Steps:

1. Construct circulant matrix C
2. Build CSS parity matrices

Hx = [ C | Cᵀ ]
Hz = [ C | Cᵀ ]

3. Check validity conditions:

- CSS commutation condition  
Hx Hzᵀ = 0

- Sparsity
- Rank estimation
- Estimated logical qubits

4. Export fixed matrices into:
bicycle_24.py


---

### bicycle_24.py

Contains **frozen matrices used in simulation**.

This ensures experiments are reproducible.

Example contents:
C
Hx
Hz


These matrices are imported by the simulation pipeline.

---

# Relation to the Paper

Bicycle codes were proposed as **cyclic LDPC constructions**.

Properties:

- sparse
- quasi-cyclic
- dual-containing

The matrices in this package follow the same construction but are much smaller than the large codes studied in the literature.

Typical research codes use:

n = 1000+

Here we use small examples for simulation feasibility.