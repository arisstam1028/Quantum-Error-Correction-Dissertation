from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from Depolarizing_Channel.depolarizing import DepolarizingChannel
from Table_Decoding_and_Error_Correction.decoder import SyndromeTableDecoder
from Table_Decoding_and_Error_Correction.plotter import StabilizerCircuitPlotter
from Table_Decoding_and_Error_Correction.stabilizer_measurement import StabilizerMeasurement


# =========================
# Binary symplectic helpers
# =========================
class BinarySymplectic:
    @staticmethod
    def pauli_char_to_bits(pauli: str) -> tuple[int, int]:
        if pauli == "I":
            return 0, 0
        if pauli == "X":
            return 1, 0
        if pauli == "Z":
            return 0, 1
        if pauli == "Y":
            return 1, 1
        raise ValueError(f"Invalid Pauli symbol: {pauli}")

    @staticmethod
    def bits_to_pauli_char(x: int, z: int) -> str:
        if x == 0 and z == 0:
            return "I"
        if x == 1 and z == 0:
            return "X"
        if x == 0 and z == 1:
            return "Z"
        if x == 1 and z == 1:
            return "Y"
        raise ValueError(f"Invalid bits ({x}, {z})")

    @classmethod
    def pauli_string_to_bsf(cls, pauli_string: str) -> tuple[np.ndarray, np.ndarray]:
        ex = np.zeros(len(pauli_string), dtype=np.uint8)
        ez = np.zeros(len(pauli_string), dtype=np.uint8)

        for i, p in enumerate(pauli_string):
            x, z = cls.pauli_char_to_bits(p)
            ex[i] = x
            ez[i] = z

        return ex, ez

    @classmethod
    def bsf_to_pauli_string(cls, ex: np.ndarray, ez: np.ndarray) -> str:
        if len(ex) != len(ez):
            raise ValueError("ex and ez must have the same length")

        return "".join(
            cls.bits_to_pauli_char(int(x), int(z))
            for x, z in zip(ex.tolist(), ez.tolist())
        )

    @staticmethod
    def add_errors(
        ex1: np.ndarray,
        ez1: np.ndarray,
        ex2: np.ndarray,
        ez2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return ((ex1 ^ ex2).astype(np.uint8), (ez1 ^ ez2).astype(np.uint8))

    @staticmethod
    def weight(ex: np.ndarray, ez: np.ndarray) -> int:
        return int(((ex | ez) != 0).sum())

    @staticmethod
    def key(ex: np.ndarray, ez: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(int(v) for v in ex.tolist()),
            tuple(int(v) for v in ez.tolist()),
        )


# =========================
# 5-qubit code
# =========================
@dataclass(frozen=True)
class FiveQubitCode:
    stabilizers: Sequence[str] = (
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubits", len(self.stabilizers[0]))


# =========================
# Simulation config/results
# =========================
@dataclass(frozen=True)
class SimulationConfig:
    probabilities: Iterable[float]
    frames_per_probability: Iterable[int]
    seed: int | None = None
    failure_threshold: int | None = None


@dataclass(frozen=True)
class SimulationResults:
    probabilities: list[float]
    frames_run: list[int]

    logical_failure_rates: list[float]
    z_basis_qber: list[float]
    x_basis_qber: list[float]
    average_qber: list[float]


# =========================
# Simulation runner
# =========================
class SimulationRunner:
    def __init__(self, code: FiveQubitCode | None = None):
        self.code = code or FiveQubitCode()
        self.measurement = StabilizerMeasurement(self.code.stabilizers)
        self.decoder = SyndromeTableDecoder(self.measurement)

    @staticmethod
    def _is_trivial_residual(ex: np.ndarray, ez: np.ndarray) -> bool:
        return not np.any(ex) and not np.any(ez)

    def run(self, config: SimulationConfig) -> SimulationResults:
        channel = DepolarizingChannel(
            seed=config.seed,
            use_independent_bsc_approx=True,   # <- book-style setting
        )

        probabilities = list(config.probabilities)
        frames_list = list(config.frames_per_probability)

        logical_failure_rates: list[float] = []
        z_qber_list: list[float] = []
        x_qber_list: list[float] = []
        avg_qber_list: list[float] = []
        frames_run: list[int] = []

        for p, max_frames in zip(probabilities, frames_list):
            failures = 0
            z_errors = 0
            x_errors = 0
            frames = 0

            for _ in range(max_frames):
                ex, ez = channel.sample_error(self.code.n_qubits, p)

                syndrome = self.measurement.compute_syndrome(ex, ez)
                cx, cz = self.decoder.decode(syndrome)

                # residual = error + correction mod 2
                rx, rz = BinarySymplectic.add_errors(ex, ez, cx, cz)

                frames += 1

                # Book-style logical/frame failure proxy:
                # count as failure if anything remains after correction.
                # This is closer to their classical-domain hard-decision view.
                if not self._is_trivial_residual(rx, rz):
                    failures += 1

                # Book-style effective QBER:
                # treat residual X-part and Z-part separately as two BSC-like channels.
                if np.any(rx):
                    z_errors += 1

                if np.any(rz):
                    x_errors += 1

                if (
                    config.failure_threshold is not None
                    and failures >= config.failure_threshold
                ):
                    break

            frames_run.append(frames)

            logical_failure_rates.append(failures / frames)
            z_qber = z_errors / frames
            x_qber = x_errors / frames
            avg_qber = 0.5 * (z_qber + x_qber)

            z_qber_list.append(z_qber)
            x_qber_list.append(x_qber)
            avg_qber_list.append(avg_qber)

        return SimulationResults(
            probabilities=probabilities,
            frames_run=frames_run,
            logical_failure_rates=logical_failure_rates,
            z_basis_qber=z_qber_list,
            x_basis_qber=x_qber_list,
            average_qber=avg_qber_list,
        )


# =========================
# Reporting
# =========================
class SimulationReport:
    def __init__(self, code: FiveQubitCode):
        self.code = code

    def print_summary(self, results: SimulationResults) -> None:
        print("\n=== Simulation Results ===\n")

        for p, frames, lf, z, x, avg in zip(
            results.probabilities,
            results.frames_run,
            results.logical_failure_rates,
            results.z_basis_qber,
            results.x_basis_qber,
            results.average_qber,
        ):
            print(
                f"p={p:.3f} | frames={frames} "
                f"| FER={lf:.6f} "
                f"| Z-QBER={z:.6f} "
                f"| X-QBER={x:.6f} "
                f"| AVG-QBER={avg:.6f}"
            )

    def show_plot(self, results: SimulationResults) -> None:
        StabilizerCircuitPlotter.show_metric_curves(
            probabilities=results.probabilities,
            z_basis_qber=results.z_basis_qber,
            x_basis_qber=results.x_basis_qber,
            average_qber=results.average_qber,
            logical_failure_rates=results.logical_failure_rates,
        )