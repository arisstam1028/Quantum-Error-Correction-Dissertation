# Purpose:
#   Runs the five-qubit code table-decoding Monte Carlo simulation
#   and reports logical failure and basis-dependent QBER estimates.
#
# Process:
#   1. Build the five-qubit stabilizer code and its logical cosets.
#   2. Sample depolarizing-channel Pauli errors frame by frame.
#   3. Compute syndromes, decode with a fixed table, and classify residuals.
#   4. Accumulate logical failure and QBER statistics for plotting.
#
# Theory link:
#   Actual errors and estimated corrections are combined modulo two in
#   binary symplectic form. Residual errors in the stabilizer group are
#   harmless, while non-trivial logical cosets count as decoding failures.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from Depolarizing_Channel.depolarizing import DepolarizingChannel
from Table_Decoding_and_Error_Correction.decoder import SyndromeTableDecoder
from Table_Decoding_and_Error_Correction.plotter import StabilizerCircuitPlotter
from Table_Decoding_and_Error_Correction.stabilizer_measurement import StabilizerMeasurement


# Binary symplectic helpers
class BinarySymplectic:
    @staticmethod
    def pauli_char_to_bits(pauli: str) -> tuple[int, int]:
        """
        Convert one Pauli symbol into binary X and Z components.

        Role in pipeline:
            Establishes the binary symplectic representation shared by
            stabilizers, physical errors, corrections, and logical cosets.
        """
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
        """
        Add two Pauli errors modulo phase using GF(2) addition.

        Role in pipeline:
            Forms residual errors by combining the sampled physical error
            with the correction returned by the decoder.
        """
        return ((ex1 ^ ex2).astype(np.uint8), (ez1 ^ ez2).astype(np.uint8))

    @staticmethod
    def key(ex: np.ndarray, ez: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(int(v) for v in ex.tolist()),
            tuple(int(v) for v in ez.tolist()),
        )


# 5-qubit code
@dataclass(frozen=True)
class FiveQubitCode:
    stabilizers: Sequence[str] = (
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    )
    logical_x: str = "XXXXX"
    logical_z: str = "ZZZZZ"

    def __post_init__(self) -> None:
        n = len(self.stabilizers[0])
        object.__setattr__(self, "n_qubits", n)

        logical_x_ex, logical_x_ez = BinarySymplectic.pauli_string_to_bsf(self.logical_x)
        logical_z_ex, logical_z_ez = BinarySymplectic.pauli_string_to_bsf(self.logical_z)

        object.__setattr__(self, "logical_x_ex", logical_x_ex)
        object.__setattr__(self, "logical_x_ez", logical_x_ez)
        object.__setattr__(self, "logical_z_ex", logical_z_ex)
        object.__setattr__(self, "logical_z_ez", logical_z_ez)

        logical_y_ex, logical_y_ez = BinarySymplectic.add_errors(
            logical_x_ex, logical_x_ez, logical_z_ex, logical_z_ez
        )
        object.__setattr__(self, "logical_y_ex", logical_y_ex)
        object.__setattr__(self, "logical_y_ez", logical_y_ez)

        stabilizer_group = self._generate_stabilizer_group()
        object.__setattr__(self, "stabilizer_group", stabilizer_group)

        identity_coset = set(stabilizer_group)
        logical_x_coset = {
            BinarySymplectic.key(
                *BinarySymplectic.add_errors(self.logical_x_ex, self.logical_x_ez, ex, ez)
            )
            for ex, ez in stabilizer_group
        }
        logical_y_coset = {
            BinarySymplectic.key(
                *BinarySymplectic.add_errors(self.logical_y_ex, self.logical_y_ez, ex, ez)
            )
            for ex, ez in stabilizer_group
        }
        logical_z_coset = {
            BinarySymplectic.key(
                *BinarySymplectic.add_errors(self.logical_z_ex, self.logical_z_ez, ex, ez)
            )
            for ex, ez in stabilizer_group
        }

        object.__setattr__(self, "identity_coset", identity_coset)
        object.__setattr__(self, "logical_x_coset", logical_x_coset)
        object.__setattr__(self, "logical_y_coset", logical_y_coset)
        object.__setattr__(self, "logical_z_coset", logical_z_coset)

    def _generate_stabilizer_group(
        self,
    ) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
        """
        Generate the stabilizer group from the four independent generators.

        Role in pipeline:
            Creates the identity coset used to decide whether a residual
            Pauli error is stabilizer-equivalent and therefore harmless.
        """
        zero_ex = np.zeros(self.n_qubits, dtype=np.uint8)
        zero_ez = np.zeros(self.n_qubits, dtype=np.uint8)

        group_arrays: list[tuple[np.ndarray, np.ndarray]] = [(zero_ex, zero_ez)]

        for stabilizer in self.stabilizers:
            gx, gz = BinarySymplectic.pauli_string_to_bsf(stabilizer)

            new_group: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
            for ex, ez in group_arrays:
                k1 = BinarySymplectic.key(ex, ez)
                new_group[k1] = (ex.copy(), ez.copy())

                ex2, ez2 = BinarySymplectic.add_errors(ex, ez, gx, gz)
                k2 = BinarySymplectic.key(ex2, ez2)
                new_group[k2] = (ex2, ez2)

            group_arrays = list(new_group.values())

        return {BinarySymplectic.key(ex, ez) for ex, ez in group_arrays}

    def classify_residual(self, ex: np.ndarray, ez: np.ndarray) -> str:
        """
        Classify a residual error as stabilizer-equivalent or logical.

        Role in pipeline:
            Converts the residual correction-plus-error pattern into the
            logical failure label used by the Monte Carlo statistics.
        """
        residual_key = BinarySymplectic.key(ex, ez)

        if residual_key in self.identity_coset:
            return "I"
        if residual_key in self.logical_x_coset:
            return "X"
        if residual_key in self.logical_y_coset:
            return "Y"
        if residual_key in self.logical_z_coset:
            return "Z"

        residual_string = BinarySymplectic.bsf_to_pauli_string(ex, ez)
        raise ValueError(f"Invalid residual: {residual_string}")


# Simulation config/results
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


# Simulation runner
class SimulationRunner:
    def __init__(self, code: FiveQubitCode | None = None):
        self.code = code or FiveQubitCode()
        self.measurement = StabilizerMeasurement(self.code.stabilizers)
        self.decoder = SyndromeTableDecoder(self.measurement)

    def run(self, config: SimulationConfig) -> SimulationResults:
        """
        Execute the table-decoding Monte Carlo experiment.

        Role in pipeline:
            Repeats channel sampling, syndrome calculation, decoding, and
            residual classification for each physical error probability.
        """
        rng = random.Random(config.seed)
        channel = DepolarizingChannel(seed=config.seed)

        probabilities = list(config.probabilities)
        frames_list = list(config.frames_per_probability)

        logical_failure_rates = []
        z_qber_list = []
        x_qber_list = []
        avg_qber_list = []
        frames_run = []

        for p, max_frames in zip(probabilities, frames_list):
            failures = 0
            z_errors = 0
            x_errors = 0
            frames = 0

            for _ in range(max_frames):
                ex, ez = channel.sample_error(self.code.n_qubits, p)
                syndrome = self.measurement.compute_syndrome(ex, ez)
                cx, cz = self.decoder.decode(syndrome)

                # residual = correction + error  (mod 2)
                rx, rz = BinarySymplectic.add_errors(cx, cz, ex, ez)
                logical = self.code.classify_residual(rx, rz)

                frames += 1

                if logical != "I":
                    failures += 1

                # Z basis
                if logical == "X":
                    z_errors += 1
                elif logical == "Y":
                    if rng.random() < 0.5:
                        z_errors += 1

                # X basis
                if logical == "Z":
                    x_errors += 1
                elif logical == "Y":
                    if rng.random() < 0.5:
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


# Reporting
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
                f"| logical={lf:.6f} "
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
