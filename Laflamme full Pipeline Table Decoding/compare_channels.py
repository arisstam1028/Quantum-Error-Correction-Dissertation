# Purpose:
#   Compare two five-qubit table-decoding channel models.
#
# Process:
#   1. Dynamically load the independent-BSC simulation project.
#   2. Dynamically load the exact symmetric-depolarizing project.
#   3. Run both sweeps and plot their average QBER curves.
#
# Theory link:
#   The comparison shows the effect of replacing the full depolarizing
#   channel with independent X/Z binary symmetric channels.
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

# Corrected mapping based on your note:
# "Closer to the book Hpefully"  symmetric depolarizing
# "QBER DepoS"  BSC approximation
SYMM_PROJECT = ROOT / "5 - Qubit Stabilizer Code Full Pipeline Closer to the book Hpefully"
BSC_PROJECT = ROOT / "5 - Qubit Stabilizer Code Full Pipeline QBER DepoS"


def _clear_conflicting_modules() -> None:
    to_remove = []

    for name in list(sys.modules.keys()):
        if (
            name == "simulation_runner"
            or name.startswith("Depolarizing_Channel")
            or name.startswith("Table_Decoding_and_Error_Correction")
            or name == "simulation_runner_bsc"
            or name == "simulation_runner_symm"
        ):
            to_remove.append(name)

    for name in to_remove:
        sys.modules.pop(name, None)


def _load_simulation_runner(project_dir: Path, alias: str):
    sim_path = project_dir / "simulation_runner.py"
    if not sim_path.exists():
        raise FileNotFoundError(f"Could not find: {sim_path}")

    _clear_conflicting_modules()

    sys.path.insert(0, str(project_dir))
    try:
        spec = importlib.util.spec_from_file_location(alias, sim_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {sim_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _run_project(module, seed: int = 7):
    code = module.FiveQubitCode()
    runner = module.SimulationRunner(code)

    probabilities = [i / 200 for i in range(1, 21)]

    config = module.SimulationConfig(
        probabilities=probabilities,
        frames_per_probability=[200000 for _ in probabilities],
        seed=seed,
        failure_threshold=None,
    )

    return runner.run(config)


def _print_summary(label: str, results) -> None:
    print(f'\n {label} ')
    print("p        avg_qber")
    print("-" * 24)
    for p, q in zip(results.probabilities, results.average_qber):
        print(f"{p:0.3f}    {q:0.8f}")


def main() -> None:
    if not BSC_PROJECT.exists():
        raise FileNotFoundError(f"BSC project folder not found:\n{BSC_PROJECT}")

    if not SYMM_PROJECT.exists():
        raise FileNotFoundError(f"Symmetric project folder not found:\n{SYMM_PROJECT}")

    # Load BSC approximation project
    bsc_module = _load_simulation_runner(
        BSC_PROJECT,
        alias="simulation_runner_bsc",
    )
    bsc_results = _run_project(bsc_module, seed=7)

    # Load exact symmetric depolarizing project
    symm_module = _load_simulation_runner(
        SYMM_PROJECT,
        alias="simulation_runner_symm",
    )
    symm_results = _run_project(symm_module, seed=7)

    _print_summary("BSC approximation", bsc_results)
    _print_summary("Exact symmetric depolarizing", symm_results)

    plt.figure(figsize=(9, 6))

    plt.plot(
        bsc_results.probabilities,
        bsc_results.average_qber,
        marker="o",
        linestyle="--",
        label="BSC approximation",
    )

    plt.plot(
        symm_results.probabilities,
        symm_results.average_qber,
        marker="s",
        linestyle="-",
        label="Exact symmetric depolarizing",
    )

    plt.yscale("log")
    plt.xlabel("Depolarizing probability p")
    plt.ylabel("Average QBER")
    plt.title("QBER Comparison: BSC vs Symmetric Quantum Depolarizing")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = ROOT / "bsc_vs_symmetric_qber_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved plot to:\n{output_path}")

    plt.show()


if __name__ == "__main__":
    main()
