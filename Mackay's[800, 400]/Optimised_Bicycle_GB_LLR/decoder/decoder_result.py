from dataclasses import dataclass

import numpy as np


@dataclass
class DecoderResult:
    estimated_error: np.ndarray
    success: bool
    iterations_used: int
    residual_syndrome: np.ndarray
    converged: bool