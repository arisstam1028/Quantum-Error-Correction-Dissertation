# Purpose:
#   Implements Pauli error sampling for the five-qubit code under
#   a symmetric depolarizing channel.
#
# Process:
#   1. Draw one random number per physical qubit.
#   2. Map the depolarizing probabilities to I, X, Y, or Z errors.
#   3. Return the error in binary symplectic form as X and Z bits.
#
# Theory link:
#   The symmetric depolarizing channel applies I with probability
#   1-p and X, Y, Z with probability p/3 each. In binary symplectic
#   form, Y is represented as simultaneous X and Z components.

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DepolarizingChannel:
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample_error(self, n: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample one n-qubit Pauli error from the depolarizing channel.

        Role in pipeline:
            Provides the physical error pattern that is later converted
            into a stabilizer syndrome and decoded by the table decoder.

        Returns:
            ex, ez in GF(2)^n
        Pauli mapping:
            I -> (0,0)
            X -> (1,0)
            Z -> (0,1)
            Y -> (1,1)
        """
        r = self.rng.random(n)
        ex = np.zeros(n, dtype=np.uint8)
        ez = np.zeros(n, dtype=np.uint8)

        x_mask = (r >= 1.0 - p) & (r < 1.0 - 2.0 * p / 3.0)
        y_mask = (r >= 1.0 - 2.0 * p / 3.0) & (r < 1.0 - p / 3.0)
        z_mask = r >= 1.0 - p / 3.0

        ex[x_mask] = 1
        # A Y error anticommutes like both X and Z, so both bits are set.
        ex[y_mask] = 1
        ez[y_mask] = 1
        ez[z_mask] = 1

        return ex, ez

    def sample_error_batch(self, batch_size: int, n: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample several independent Pauli error patterns.

        Role in pipeline:
            Supports batched Monte Carlo experiments while preserving the
            same binary symplectic representation used by syndrome decoding.
        """
        ex_batch = np.zeros((batch_size, n), dtype=np.uint8)
        ez_batch = np.zeros((batch_size, n), dtype=np.uint8)
        for i in range(batch_size):
            ex, ez = self.sample_error(n, p)
            ex_batch[i] = ex
            ez_batch[i] = ez
        return ex_batch, ez_batch

    @staticmethod
    def channel_binary_prior(p: float) -> Tuple[float, float]:
        """
        Binary prior used by CSS BP.
        For one component (X-part or Z-part), effective error probability is 2p/3.

        Role in pipeline:
            Converts the Pauli depolarizing probability into the marginal
            binary prior used when X and Z components are decoded separately.
        """
        p1 = 2.0 * p / 3.0
        p0 = 1.0 - p1
        return p0, p1
