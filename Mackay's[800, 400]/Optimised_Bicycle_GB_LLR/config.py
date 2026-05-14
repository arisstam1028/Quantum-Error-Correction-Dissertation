from dataclasses import dataclass, field
from typing import List


@dataclass
class QLDPCConfig:
    # Fixed matrix module to import at runtime
    matrix_module: str = "matrices.mackay_800_400"

    # Decoder selection
    decoder_type: str = "bp"

    # Decoder: base BP
    max_bp_iters: int = 90
    bp_epsilon: float = 1e-12

    # Decoder: random perturbation for BinaryBPDecoder
    enable_random_perturbation: bool = False
    perturb_iters: int = 40
    perturb_max_feedbacks: int = 40
    perturb_strength: float = 0.1
    perturb_seed: int | None = None

    # Simulation
    probabilities: List[float] = field(
        default_factory=lambda: [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03]
    )
    frames_per_p: List[int] = field(
        default_factory=lambda: [
            200000, 100000, 50000, 20000, 20000,
            20000, 10000, 10000, 10000
        ]
    )
    max_failures_per_p: List[int] = field(
        default_factory=lambda: [
            3000, 3000, 3000, 3000, 3000,
            3000, 3000, 3000, 3000
        ]
    )

    # Output
    verbose: bool = True
    print_matrices: bool = True


def build_config(matrix_module: str = "matrices.mackay_800_400") -> QLDPCConfig:
    cfg = QLDPCConfig(matrix_module=matrix_module)

    if len(cfg.frames_per_p) != len(cfg.probabilities):
        raise ValueError("frames_per_p must have the same length as probabilities")

    if len(cfg.max_failures_per_p) != len(cfg.probabilities):
        raise ValueError("max_failures_per_p must have the same length as probabilities")

    return cfg