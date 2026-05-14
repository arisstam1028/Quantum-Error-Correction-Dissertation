from typing import Tuple

import numpy as np


PAULI_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}

XZ_TO_PAULI = {
    (0, 0): "I",
    (1, 0): "X",
    (1, 1): "Y",
    (0, 1): "Z",
}


def pauli_string_to_binary(pauli: str) -> Tuple[np.ndarray, np.ndarray]:
    ex = []
    ez = []
    for ch in pauli:
        x, z = PAULI_TO_XZ[ch]
        ex.append(x)
        ez.append(z)
    return np.array(ex, dtype=np.uint8), np.array(ez, dtype=np.uint8)


def binary_to_pauli_string(ex: np.ndarray, ez: np.ndarray) -> str:
    if ex.shape != ez.shape:
        raise ValueError("ex and ez must have the same shape")
    return "".join(XZ_TO_PAULI[(int(x), int(z))] for x, z in zip(ex, ez))


def add_pauli_errors(ex1: np.ndarray, ez1: np.ndarray, ex2: np.ndarray, ez2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (ex1 ^ ex2).astype(np.uint8), (ez1 ^ ez2).astype(np.uint8)