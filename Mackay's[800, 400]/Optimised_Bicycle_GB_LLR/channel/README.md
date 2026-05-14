# channel Package

This package models the **quantum noise channel**.

Currently only the **depolarizing channel** is implemented.

---

# Files

### depolarizing.py

Implements a depolarizing channel.

For each qubit:

No error with probability:

1 - p

Error with probability:

p

The error is uniformly chosen from:

X  
Y  
Z

---

# Binary Representation

For stabilizer simulation we convert Pauli errors into binary form.

Pauli error:

X → (1,0)  
Z → (0,1)  
Y → (1,1)

This allows the simulation to operate using binary matrices.

---

# Relation to the Papers

The depolarizing channel is approximated as a **Binary Symmetric Channel** with crossover probability:

2p / 3

This approximation is used in the BP decoder.