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
        ex[y_mask] = 1
        ez[y_mask] = 1
        ez[z_mask] = 1

        return ex, ez

    def sample_error_batch(self, batch_size: int, n: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        p1 = 2.0 * p / 3.0
        p0 = 1.0 - p1
        return p0, p1