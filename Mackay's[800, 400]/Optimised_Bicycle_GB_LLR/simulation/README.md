# simulation Package

This package runs the **Monte Carlo experiments**.

It connects all components:

channel  
decoder  
matrices  
metrics

---

# Files

### runner.py

Runs a **single frame simulation**.

Steps:

1. Sample error from channel
2. Compute syndrome
3. Decode using BP
4. Check if decoding succeeded

Returns success/failure and decoder statistics.

---

### monte_carlo.py

Runs many frames for each channel probability.

Tracks:

frames simulated  
decoder failures  
average iterations

Stops early when a maximum number of failures is reached.

---

### metrics.py

Computes performance statistics including:

Frame Error Rate (FER)

FER = failures / frames

These metrics are used for plotting results.