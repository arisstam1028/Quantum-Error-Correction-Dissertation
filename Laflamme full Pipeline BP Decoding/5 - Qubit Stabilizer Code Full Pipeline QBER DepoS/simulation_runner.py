from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from Depolarizing_Channel.depolarizing import DepolarizingChannel
from Table_Decoding_and_Error_Correction.bp_decoder import BinaryBPDecoder
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
# BP wrapper for quantum code
# =========================
class FiveQubitBPDecoder:
    def __init__(self, measurement: StabilizerMeasurement, max_iters: int = 30):
        self.measurement = measurement
        self.n = measurement.n_qubits

        # syndrome = Hx*ez + Hz*ex
        # if e = [ex | ez], then H_bp = [Hz | Hx]
        H_bp = np.hstack([measurement.hz, measurement.hx]).astype(np.uint8)
        self.bp = BinaryBPDecoder(H_bp, max_iters=max_iters)

    def decode(self, syndrome: str, physical_p: float) -> tuple[np.ndarray, np.ndarray]:
        syndrome_vec = np.array([int(b) for b in syndrome], dtype=np.uint8)

        # matched BP prior for the independent-BSC approximation
        p_error = 2.0 * physical_p / 3.0
        p_error = min(max(p_error, 1e-12), 1.0 - 1e-12)

        result = self.bp.decode(syndrome_vec, p_error)
        est = result.estimated_error

        ex_hat = est[:self.n].astype(np.uint8)
        ez_hat = est[self.n:].astype(np.uint8)
        return ex_hat, ez_hat


# =========================
# Simulation config/results
# =========================
@dataclass(frozen=True)
class SimulationConfig:
    probabilities: Iterable[float]
    frames_per_probability: Iterable[int]
    seed: int | None = None
    failure_threshold: int | None = None
    bp_max_iters: int = 30


@dataclass(frozen=True)
class ChannelResults:
    logical_failure_rates: list[float]
    z_basis_qber: list[float]
    x_basis_qber: list[float]
    average_qber: list[float]
    frames_run: list[int]


@dataclass(frozen=True)
class ComparisonResults:
    probabilities: list[float]
    bsc: ChannelResults
    symmetric: ChannelResults


# =========================
# Simulation runner
# =========================
class SimulationRunner:
    def __init__(self, code: FiveQubitCode | None = None):
        self.code = code or FiveQubitCode()
        self.measurement = StabilizerMeasurement(self.code.stabilizers)

    @staticmethod
    def _is_trivial_residual(ex: np.ndarray, ez: np.ndarray) -> bool:
        return not np.any(ex) and not np.any(ez)

    def _run_single_channel(
        self,
        probabilities: list[float],
        frames_list: list[int],
        seed: int | None,
        failure_threshold: int | None,
        bp_max_iters: int,
        use_independent_bsc_approx: bool,
    ) -> ChannelResults:
        channel = DepolarizingChannel(
            seed=seed,
            use_independent_bsc_approx=use_independent_bsc_approx,
        )
        decoder = FiveQubitBPDecoder(self.measurement, max_iters=bp_max_iters)

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
                cx, cz = decoder.decode(syndrome, p)

                rx, rz = BinarySymplectic.add_errors(ex, ez, cx, cz)

                frames += 1

                if not self._is_trivial_residual(rx, rz):
                    failures += 1

                if np.any(rx):
                    z_errors += 1

                if np.any(rz):
                    x_errors += 1

                if failure_threshold is not None and failures >= failure_threshold:
                    break

            frames_run.append(frames)

            logical_failure_rates.append(failures / frames)
            z_qber = z_errors / frames
            x_qber = x_errors / frames
            avg_qber = 0.5 * (z_qber + x_qber)

            z_qber_list.append(z_qber)
            x_qber_list.append(x_qber)
            avg_qber_list.append(avg_qber)

        return ChannelResults(
            logical_failure_rates=logical_failure_rates,
            z_basis_qber=z_qber_list,
            x_basis_qber=x_qber_list,
            average_qber=avg_qber_list,
            frames_run=frames_run,
        )

    def run(self, config: SimulationConfig) -> ComparisonResults:
        probabilities = list(config.probabilities)
        frames_list = list(config.frames_per_probability)

        bsc_results = self._run_single_channel(
            probabilities=probabilities,
            frames_list=frames_list,
            seed=config.seed,
            failure_threshold=config.failure_threshold,
            bp_max_iters=config.bp_max_iters,
            use_independent_bsc_approx=True,
        )

        symmetric_results = self._run_single_channel(
            probabilities=probabilities,
            frames_list=frames_list,
            seed=None if config.seed is None else config.seed + 1,
            failure_threshold=config.failure_threshold,
            bp_max_iters=config.bp_max_iters,
            use_independent_bsc_approx=False,
        )

        return ComparisonResults(
            probabilities=probabilities,
            bsc=bsc_results,
            symmetric=symmetric_results,
        )


# =========================
# Reporting
# =========================
class SimulationReport:
    def __init__(self, code: FiveQubitCode):
        self.code = code

    @staticmethod
    def _print_channel_summary(name: str, probabilities: list[float], results: ChannelResults) -> None:
        print(f"\n=== {name} ===\n")
        for p, frames, lf, z, x, avg in zip(
            probabilities,
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

    def print_summary(self, results: ComparisonResults) -> None:
        self._print_channel_summary("BP + Independent-BSC Approximation", results.probabilities, results.bsc)
        self._print_channel_summary("BP + Symmetric Depolarizing", results.probabilities, results.symmetric)

    def show_plot(self, results: ComparisonResults) -> None:
        StabilizerCircuitPlotter.show_bp_comparison_curves(
            probabilities=results.probabilities,
            bsc_average_qber=results.bsc.average_qber,
            symmetric_average_qber=results.symmetric.average_qber,
        )