from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from Encoding_Circuit_Builder.main import build_five_qubit_encoder_bundle
from Table_Decoding_and_Error_Correction.decoder import SyndromeTableDecoder
from Table_Decoding_and_Error_Correction.stabilizer_measurement import StabilizerParser
from Table_Decoding_and_Error_Correction.plotter import StabilizerCircuitPlotter


@dataclass(frozen=True)
class SweepResult:
    probabilities: list[float]
    frames_requested: list[int]
    frames_used: list[int]
    failures: list[int]
    logical_error_rates: list[float]


class SimulationRunner:
    @staticmethod
    def print_section(title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def print_symplectic_matrix(title: str, rows: list[list[int]]) -> None:
        if not rows:
            print(f"{title}: <empty>")
            return

        n = len(rows[0]) // 2
        print(title)
        for row in rows:
            x = "".join(str(b) for b in row[:n])
            z = "".join(str(b) for b in row[n:])
            print(f"  {x} | {z}")

    @staticmethod
    def pauli_strings_to_xz(stabilizers: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert stabilizer Pauli strings to binary X/Z matrices.

        Returns:
            sx: shape (r, n)
            sz: shape (r, n)
        """
        r = len(stabilizers)
        n = len(stabilizers[0])

        sx = np.zeros((r, n), dtype=np.uint8)
        sz = np.zeros((r, n), dtype=np.uint8)

        for i, stab in enumerate(stabilizers):
            for j, ch in enumerate(stab):
                if ch == "I":
                    pass
                elif ch == "X":
                    sx[i, j] = 1
                elif ch == "Z":
                    sz[i, j] = 1
                elif ch == "Y":
                    sx[i, j] = 1
                    sz[i, j] = 1
                else:
                    raise ValueError(f"Invalid Pauli '{ch}' in stabilizer '{stab}'")

        return sx, sz

    @staticmethod
    def build_decoder_lookup_arrays(
        decoder: SyndromeTableDecoder,
        n: int,
        r: int,
    ) -> np.ndarray:
        """
        Build a correction lookup table indexed by syndrome integer.

        correction_codes[idx, q] ∈ {0,1,2,3}
            0 -> I
            1 -> X
            2 -> Y
            3 -> Z
        """
        num_syndromes = 2 ** r
        correction_codes = np.zeros((num_syndromes, n), dtype=np.uint8)

        pauli_to_code = {
            "I": 0,
            "X": 1,
            "Y": 2,
            "Z": 3,
        }

        for syndrome_str, decoded in decoder.table.items():
            syndrome_index = int(syndrome_str, 2)

            if decoded.pauli == "I" or decoded.qubit is None:
                continue

            correction_codes[syndrome_index, decoded.qubit] = pauli_to_code[decoded.pauli]

        return correction_codes

    @staticmethod
    def sample_error_codes(batch_size: int, n: int, p: float) -> np.ndarray:
        """
        Vectorized depolarizing sampling.

        Returns array of shape (batch_size, n) with values:
            0 -> I
            1 -> X
            2 -> Y
            3 -> Z
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must satisfy 0 <= p <= 1, got {p}")

        u = np.random.random((batch_size, n))

        t0 = 1.0 - p
        t1 = t0 + p / 3.0
        t2 = t0 + 2.0 * p / 3.0

        # Produces:
        #   0 if u < t0
        #   1 if t0 <= u < t1
        #   2 if t1 <= u < t2
        #   3 if u >= t2
        codes = (
            (u >= t0).astype(np.uint8)
            + (u >= t1).astype(np.uint8)
            + (u >= t2).astype(np.uint8)
        )

        return codes

    @staticmethod
    def codes_to_xz(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert Pauli codes to X/Z binary representation.

        codes:
            0 -> I = (0,0)
            1 -> X = (1,0)
            2 -> Y = (1,1)
            3 -> Z = (0,1)
        """
        x = ((codes == 1) | (codes == 2)).astype(np.uint8)
        z = ((codes == 2) | (codes == 3)).astype(np.uint8)
        return x, z

    @staticmethod
    def syndrome_bits_from_errors(
        ex: np.ndarray,
        ez: np.ndarray,
        sx: np.ndarray,
        sz: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized symplectic syndrome computation.

        For each error E and stabilizer S:
            syndrome bit = ex·sz + ez·sx (mod 2)
        """
        bits = ((ex @ sz.T) ^ (ez @ sx.T)) & 1
        return bits.astype(np.uint8)

    @staticmethod
    def syndrome_bits_to_indices(bits: np.ndarray) -> np.ndarray:
        """
        Convert syndrome bits [b0,b1,...,b_{r-1}] to integer index
        corresponding to bitstring "b0b1...b_{r-1}".
        """
        r = bits.shape[1]
        weights = (2 ** np.arange(r - 1, -1, -1, dtype=np.int64))
        return (bits.astype(np.int64) @ weights).astype(np.int64)

    @staticmethod
    def residual_identity_mask(
        error_codes: np.ndarray,
        correction_codes: np.ndarray,
    ) -> np.ndarray:
        """
        Check whether correction * error = identity, ignoring global phase.

        In binary symplectic form, Pauli multiplication modulo phase is XOR
        on the (x|z) bits.
        """
        ex, ez = SimulationRunner.codes_to_xz(error_codes)
        cx, cz = SimulationRunner.codes_to_xz(correction_codes)

        rx = ex ^ cx
        rz = ez ^ cz

        is_identity = (~rx.any(axis=1)) & (~rz.any(axis=1))
        return is_identity

    @classmethod
    def run_error_rate_sweep_vectorized(
        cls,
        *,
        n: int,
        sx: np.ndarray,
        sz: np.ndarray,
        correction_lookup: np.ndarray,
        probabilities: Sequence[float],
        frames_per_p: Sequence[int],
        max_failures_per_p: Sequence[int],
        chunk_size: int = 20000,
    ) -> SweepResult:
        if not (
            len(probabilities) == len(frames_per_p) == len(max_failures_per_p)
        ):
            raise ValueError(
                "probabilities, frames_per_p, and max_failures_per_p must have the same length"
            )

        probabilities_list = [float(p) for p in probabilities]
        frames_requested_list = [int(x) for x in frames_per_p]
        max_failures_list = [int(x) for x in max_failures_per_p]

        frames_used_all: list[int] = []
        failures_all: list[int] = []
        logical_error_rates: list[float] = []

        for p, frames_target, max_failures in zip(
            probabilities_list,
            frames_requested_list,
            max_failures_list,
        ):
            if frames_target <= 0:
                raise ValueError(f"frames_per_p entries must be positive, got {frames_target}")
            if max_failures <= 0:
                raise ValueError(
                    f"max_failures_per_p entries must be positive, got {max_failures}"
                )

            frames_done = 0
            failures = 0

            while frames_done < frames_target and failures < max_failures:
                batch = min(chunk_size, frames_target - frames_done)

                error_codes = cls.sample_error_codes(batch, n, p)
                ex, ez = cls.codes_to_xz(error_codes)

                syndrome_bits = cls.syndrome_bits_from_errors(ex, ez, sx, sz)
                syndrome_indices = cls.syndrome_bits_to_indices(syndrome_bits)

                correction_codes = correction_lookup[syndrome_indices]

                success_mask = cls.residual_identity_mask(error_codes, correction_codes)
                batch_failures = int((~success_mask).sum())

                frames_done += batch
                failures += batch_failures

            logical_error_rate = failures / frames_done
            frames_used_all.append(frames_done)
            failures_all.append(failures)
            logical_error_rates.append(logical_error_rate)

            print(
                f"p = {p:.3f} | "
                f"frames used = {frames_done}/{frames_target} | "
                f"failures = {failures} | "
                f"logical error rate = {logical_error_rate:.8f}"
            )

        return SweepResult(
            probabilities=probabilities_list,
            frames_requested=frames_requested_list,
            frames_used=frames_used_all,
            failures=failures_all,
            logical_error_rates=logical_error_rates,
        )

    @classmethod
    def run(cls) -> None:
        cls.print_section("MONTE CARLO SIMULATION MODE")

        bundle = build_five_qubit_encoder_bundle(name="five_qubit_encoder")

        encoder_object = bundle["encoder"]
        Hs = bundle["Hs"]

        spec = encoder_object.spec
        n = spec.n
        k = spec.k
        r = spec.r

        print(f"n = {n}")
        print(f"k = {k}")
        print(f"r = {r}")

        cls.print_section("Hs MATRIX")
        cls.print_symplectic_matrix("Hs (X | Z):", Hs)

        cls.print_section("INITIAL STABILIZERS")
        stabilizers = StabilizerParser.from_symplectic_rows(Hs)
        for i, stab in enumerate(stabilizers, start=1):
            print(f"M{i} = {stab}")

        decoder = SyndromeTableDecoder(stabilizers)

        sx, sz = cls.pauli_strings_to_xz(stabilizers)
        correction_lookup = cls.build_decoder_lookup_arrays(decoder, n=n, r=r)

        cls.print_section("SIMULATION SETTINGS")

        probabilities = [i / 200 for i in range(21)]

        frames_per_p = [
            300000, 200000, 200000, 180000, 160000,
            140000, 120000, 100000,  90000,  80000,
             70000,  60000,  50000,  40000,  30000,
             25000,  20000,  15000,  12000,  10000,
              8000,
        ]

        max_failures_per_p = [
              200,   200,   200,   200,   200,
              200,   200,   200,   200,   200,
              200,   200,   200,   200,   200,
              200,   200,   200,   200,   200,
              200,
        ]

        print(f"probabilities      = {probabilities}")
        print(f"frames_per_p       = {frames_per_p}")
        print(f"max_failures_per_p = {max_failures_per_p}")

        cls.print_section("RUNNING VECTORIZED SWEEP")
        result = cls.run_error_rate_sweep_vectorized(
            n=n,
            sx=sx,
            sz=sz,
            correction_lookup=correction_lookup,
            probabilities=probabilities,
            frames_per_p=frames_per_p,
            max_failures_per_p=max_failures_per_p,
            chunk_size=20000,
        )

        cls.print_section("PLOTTING")
        StabilizerCircuitPlotter.plot_error_rate_curve(
            result.probabilities,
            result.logical_error_rates,
            title="Logical Error Rate vs Physical Error Probability",
        )
        plt.show()