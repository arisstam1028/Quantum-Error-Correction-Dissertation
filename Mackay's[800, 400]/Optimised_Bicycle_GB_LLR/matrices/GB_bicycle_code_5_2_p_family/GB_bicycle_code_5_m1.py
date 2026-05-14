import numpy as np

SUPPORT_A_BASE = [0, 4]
SUPPORT_B_BASE = [0, 1, 2, 4]
PM_SUPPORT = [0, 1]
SUPPORT_A = [1, 4]
SUPPORT_B = [3, 4]
M_FAMILY = 1
L_CURRENT = 5
ROWS_DROPPED_X = ()
ROWS_DROPPED_Z = ()

N = 10
MX = 5
MZ = 5
RANK_HX = 4
RANK_HZ = 4
K_ESTIMATE = 2
K_GCD = 2
CYCLES_4 = 0
GIRTH = 6
COMMUTES = True

C = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
], dtype=np.uint8)

A = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
], dtype=np.uint8)

B = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0]
], dtype=np.uint8)

Hx = np.array([
    [0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
], dtype=np.uint8)

Hz = np.array([
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
], dtype=np.uint8)
