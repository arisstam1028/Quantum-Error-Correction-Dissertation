from dataclasses import dataclass
from copy import deepcopy
import importlib
import os
import pkgutil
import re

import numpy as np

from channel.depolarizing import DepolarizingChannel
from code_construction.code_analysis import print_bicycle_matrices, print_code_stats
from core.helpers import GF2RowSpaceChecker
from core.syndrome import compute_css_syndrome


@dataclass
class SingleRunResult:
    success: bool
    iterations_used_x: int
    iterations_used_z: int
    ex_true: np.ndarray
    ez_true: np.ndarray
    ex_hat: np.ndarray
    ez_hat: np.ndarray
    ex_res: np.ndarray
    ez_res: np.ndarray
    sX: np.ndarray
    sZ: np.ndarray


def load_matrix_module(module_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dynamically import a fixed matrix module, for example:
        matrices.bicycle_24

    The module must define:
        C, Hx, Hz
    """
    module = importlib.import_module(module_name)

    required_names = ["C", "Hx", "Hz"]
    missing = [name for name in required_names if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Matrix module '{module_name}' is missing required variables: {missing}"
        )

    C = np.asarray(module.C, dtype=np.uint8)
    Hx = np.asarray(module.Hx, dtype=np.uint8)
    Hz = np.asarray(module.Hz, dtype=np.uint8)

    if C.ndim != 2:
        raise ValueError(f"{module_name}.C must be a 2D matrix")
    if Hx.ndim != 2 or Hz.ndim != 2:
        raise ValueError(f"{module_name}.Hx and {module_name}.Hz must be 2D matrices")
    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError(f"{module_name}.Hx and {module_name}.Hz must have the same number of columns")
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"{module_name}.C must be square")
    if Hx.shape[1] != 2 * C.shape[1]:
        raise ValueError(
            f"{module_name} has inconsistent dimensions: expected Hx/Hz to have "
            f"{2 * C.shape[1]} columns from C, got {Hx.shape[1]}"
        )

    return C, Hx, Hz


class QLDPCFamilyRunner:
    def __init__(self, base_module_path: str, decoder_type: str, config_template) -> None:
        self.base_module_path = base_module_path
        self.decoder_type = decoder_type
        self.config_template = config_template

    def discover_modules(self) -> list[tuple[int, str]]:
        package = importlib.import_module(self.base_module_path)
        discovered: list[tuple[int, str]] = []
        pattern = re.compile(r"^(?P<name>.+)_m(?P<m>\d+)$")

        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.ispkg:
                continue

            match = pattern.match(module_info.name)
            if match is None:
                continue

            discovered.append(
                (int(match.group("m")), f"{self.base_module_path}.{module_info.name}")
            )

        if not discovered:
            family_dir = os.path.dirname(package.__file__) if getattr(package, "__file__", None) else None
            raise ValueError(
                f"No family members matching '*_m<number>.py' were found in '{self.base_module_path}'"
                if family_dir is None
                else f"No family members matching '*_m<number>.py' were found in '{family_dir}'"
            )

        discovered.sort(key=lambda item: item[0])
        return discovered

    @staticmethod
    def _label_from_module(module_path: str) -> str:
        module_name = module_path.rsplit(".", 1)[-1]
        match = re.search(r"_m(\d+)$", module_name)
        if match is None:
            return module_name
        return f"m={match.group(1)}"

    def run_family(self) -> dict[str, list[dict]]:
        from simulation.monte_carlo import run_monte_carlo

        family_results: dict[str, list[dict]] = {}

        for _, module_path in self.discover_modules():
            config = deepcopy(self.config_template)
            config.matrix_module = module_path
            config.decoder_type = self.decoder_type

            results = run_monte_carlo(config)
            label = self._label_from_module(module_path)
            family_results[label] = results

        return family_results


class QLDPCRunner:
    def __init__(self, config) -> None:
        self.config = config

        self.C, self.Hx, self.Hz = load_matrix_module(config.matrix_module)
        self.n = self.Hx.shape[1]

        if config.verbose:
            print("\n=== Loaded Matrix Module ===")
            print(config.matrix_module)
            print_code_stats(self.Hx, self.Hz)

            if config.print_matrices:
                print_bicycle_matrices(self.C, self.Hx, self.Hz)

        self.channel = DepolarizingChannel(seed=None)

        # CSS decoding:
        #   Z-check syndrome sZ decodes the X-part using Hz
        #   X-check syndrome sX decodes the Z-part using Hx
        if config.decoder_type == "bp":
            from decoder.bp_decoder import BinaryBPDecoder as Decoder

            decoder_kwargs = {
                "max_iters": config.max_bp_iters,
                "epsilon": config.bp_epsilon,
                "enable_random_perturbation": config.enable_random_perturbation,
                "perturb_iters": config.perturb_iters,
                "perturb_max_feedbacks": config.perturb_max_feedbacks,
                "perturb_strength": config.perturb_strength,
                "random_seed": config.perturb_seed,
            }

        elif config.decoder_type == "svns":
            from Sequential_BP_Based_Decoding import SVNSBPDecoder as Decoder

            decoder_kwargs = {
                "max_iters": config.max_bp_iters,
                "epsilon": config.bp_epsilon,
            }

        elif config.decoder_type == "scns":
            from Sequential_BP_Based_Decoding import SCNSBPDecoder as Decoder

            decoder_kwargs = {
                "max_iters": config.max_bp_iters,
                "epsilon": config.bp_epsilon,
            }

        else:
            raise ValueError(f"Unknown decoder_type: {config.decoder_type}")

        self.decoder_x = Decoder(self.Hz, **decoder_kwargs)
        self.decoder_z = Decoder(self.Hx, **decoder_kwargs)

        self.hx_rowspace = GF2RowSpaceChecker(self.Hx)
        self.hz_rowspace = GF2RowSpaceChecker(self.Hz)

    def run_single_frame(self, p: float) -> SingleRunResult:
        ex_true, ez_true = self.channel.sample_error(self.n, p)
        sX, sZ = compute_css_syndrome(self.Hx, self.Hz, ex_true, ez_true)

        p_binary = 2.0 * p / 3.0

        # Decode X component from Z-check syndrome
        result_x = self.decoder_x.decode(sZ, p_binary)

        # Decode Z component from X-check syndrome
        result_z = self.decoder_z.decode(sX, p_binary)

        ex_hat = result_x.estimated_error.astype(np.uint8)
        ez_hat = result_z.estimated_error.astype(np.uint8)

        ex_res = (ex_true ^ ex_hat).astype(np.uint8)
        ez_res = (ez_true ^ ez_hat).astype(np.uint8)

        x_ok = self.hx_rowspace.contains(ex_res)
        z_ok = self.hz_rowspace.contains(ez_res)
        success = bool(x_ok and z_ok)

        return SingleRunResult(
            success=success,
            iterations_used_x=result_x.iterations_used,
            iterations_used_z=result_z.iterations_used,
            ex_true=ex_true,
            ez_true=ez_true,
            ex_hat=ex_hat,
            ez_hat=ez_hat,
            ex_res=ex_res,
            ez_res=ez_res,
            sX=sX,
            sZ=sZ,
        )