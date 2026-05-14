"""
Purpose:
    Sample binary symplectic error vectors for MacKay QLDPC experiments.

Process:
    Either sample X and Z components independently under the BSC
    approximation, or sample exact Pauli errors from the symmetric
    depolarizing channel.

Theory link:
    A Y error carries both X and Z binary components, so it contributes to
    both CSS decoding problems.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DepolarizingChannel:
    seed: Optional[int] = None
    use_independent_bsc_approx: bool = True

    def __post_init__(self) -> None:
        """
        Create the random number generator used by all channel samples.
        """
        self.rng = np.random.default_rng(self.seed)

    def sample_error(self, n: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            ex, ez in GF(2)^n

        If use_independent_bsc_approx=True:
            X and Z sampled independently with p = 2p/3

        Else:
            Exact depolarizing:
                I: 1-p
                X,Y,Z: p/3

        Role in pipeline:
            Supplies the true physical error used for syndrome generation
            before BP estimates a correction.
        """

        if self.use_independent_bsc_approx:
            pe = 2.0 * p / 3.0

            ex = (self.rng.random(n) < pe).astype(np.uint8)
            ez = (self.rng.random(n) < pe).astype(np.uint8)

            return ex, ez

        # Exact depolarizing
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

    @staticmethod
    def channel_binary_prior(p: float) -> Tuple[float, float]:
        """
        Return the binary prior used by each CSS component decoder.

        Role in pipeline:
            Converts physical depolarizing probability p into the marginal
            probability that an X or Z component is present.
        """
        p1 = 2.0 * p / 3.0
        p0 = 1.0 - p1
        return p0, p1
