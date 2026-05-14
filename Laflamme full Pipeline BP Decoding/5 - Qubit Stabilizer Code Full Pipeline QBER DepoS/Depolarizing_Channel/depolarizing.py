from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DepolarizingChannel:
    seed: Optional[int] = None
    use_independent_bsc_approx: bool = True

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample_error(self, n: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            ex, ez in GF(2)^n

        If use_independent_bsc_approx=True, use the model described in the book:
            px = pz = 2p/3
        and sample X-part and Z-part independently.

        This induces:
            pI = 1 - 4p/3 + 4p^2/9
            pX = 2p/3 - 4p^2/9
            pY = 4p^2/9
            pZ = 2p/3 - 4p^2/9

        If use_independent_bsc_approx=False, use exact symmetric depolarizing:
            I with 1-p
            X,Y,Z each with p/3
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must satisfy 0 <= p <= 1, got {p}")

        if self.use_independent_bsc_approx:
            pe = 2.0 * p / 3.0

            ex = (self.rng.random(n) < pe).astype(np.uint8)
            ez = (self.rng.random(n) < pe).astype(np.uint8)

            return ex, ez

        # Exact symmetric depolarizing channel
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
        Binary prior used in the independent-BSC approximation:
            pe = 2p/3
        """
        p1 = 2.0 * p / 3.0
        p0 = 1.0 - p1
        return p0, p1