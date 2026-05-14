# core Package

Contains fundamental utilities used throughout the simulation.

These implement the **mathematical representation of stabilizer codes**.

---

# Files

### css.py

Contains helper functions related to CSS codes.

Includes:

- commutation checks
- matrix structure utilities

---

### pauli.py

Represents Pauli errors using binary vectors.

Pauli errors are represented as:

X component  
Z component

Example:

X error vector  
Z error vector

This representation is standard in stabilizer code simulations.

---

### syndrome.py

Computes stabilizer syndromes.

For CSS codes:

sX = Hx * eZᵀ  
sZ = Hz * eXᵀ

These syndromes are the inputs to the BP decoder.

---

### helpers.py

Small helper utilities used throughout the framework.

Includes matrix operations and formatting helpers.