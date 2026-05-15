"""
Purpose:
    Entry point for the MacKay bicycle QLDPC simulations.

Process:
    Select one matrix family, one decoder mode, and one channel mode. Run
    the Monte Carlo sweep, then plot FER and average decoder effort.

Theory link:
    The same CSS decoder can be tested against an independent BSC
    approximation or the exact symmetric depolarizing channel.
"""

from copy import deepcopy
import time

from config import build_config
from plotting.plotter import (
    plot_avg_iterations,
    plot_avg_iterations_compare,
    plot_fer,
    plot_fer_compare,
)
from simulation.monte_carlo import run_monte_carlo
from simulation.runner import QLDPCFamilyRunner


DECODER_TYPE = "bp"   # "bp", "svns", "scns", "compare"
RUN_FAMILY = False

# "bsc", "depolarizing", or "both"
CHANNEL_MODE = "both"

CHANNEL_OPTIONS = {
    "bsc": (True, "BSC Approx"),
    "depolarizing": (False, "Depolarizing"),
}


def run_single_channel(config, channel_mode: str) -> list[dict]:
    """
    Run one Monte Carlo sweep after applying the selected channel model.

    Role in pipeline:
        Keeps channel comparison outside the decoder so BP, SVNS, and SCNS
        share the same matrix and probability schedule.
    """
    use_bsc_channel, label = CHANNEL_OPTIONS[channel_mode]
    config.use_bsc_channel = use_bsc_channel

    print(f'\n Running Channel: {label} ')
    return run_monte_carlo(config)


def main() -> None:
    """
    Configure and run fixed-code or family experiments.

    Role in pipeline:
        Routes between a single MacKay matrix run, a BSC-vs-depolarizing
        comparison, and a family sweep over generated bicycle-code modules.
    """
    matrix_module = "matrices.mackay_800_400"
    family_module = "matrices.GB_bicycle_code_5_2_p_family"

    if CHANNEL_MODE not in CHANNEL_OPTIONS and CHANNEL_MODE != "both":
        raise ValueError("CHANNEL_MODE must be 'bsc', 'depolarizing', or 'both'")

    if RUN_FAMILY:
        if CHANNEL_MODE == "both":
            raise ValueError("CHANNEL_MODE='both' is only supported when RUN_FAMILY=False")

        decoder_types = ["bp", "svns", "scns"] if DECODER_TYPE == "compare" else [DECODER_TYPE]
        base_config = build_config(matrix_module=matrix_module)
        base_config.use_bsc_channel = CHANNEL_OPTIONS[CHANNEL_MODE][0]

        all_results = {}

        for dec in decoder_types:
            family_runner = QLDPCFamilyRunner(
                base_module_path=family_module,
                decoder_type=dec,
                config_template=deepcopy(base_config),
            )
            family_results = family_runner.run_family()
            all_results[dec] = family_results

        return

    if DECODER_TYPE != "compare":
        config = build_config(matrix_module=matrix_module)
        config.decoder_type = DECODER_TYPE

        if CHANNEL_MODE == "both":
            all_results = {}

            for channel_mode, (_, label) in CHANNEL_OPTIONS.items():
                channel_config = deepcopy(config)
                all_results[label] = run_single_channel(channel_config, channel_mode)

            print('\n Final Results ')
            print(f"Matrix module used: {config.matrix_module}")
            print(f"Decoder used: {config.decoder_type}")
            print("Channel: BSC Approx and Depolarizing")

            plot_fer_compare(all_results, title="FER vs p: Channel Comparison")
            plot_avg_iterations_compare(all_results, title="Average Iterations vs p: Channel Comparison")
            return

        results = run_single_channel(config, CHANNEL_MODE)
        channel_label = CHANNEL_OPTIONS[CHANNEL_MODE][1]

        print('\n Final Results ')
        print(f"Matrix module used: {config.matrix_module}")
        print(f"Decoder used: {config.decoder_type}")
        print(f"Channel: {channel_label}")

        plot_fer(results, title=f"FER vs p: {channel_label}")
        plot_avg_iterations(results, title=f"Average Iterations vs p: {channel_label}")
        return


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    print(f'\n Total Runtime ')
    print(f"{end_time - start_time:.2f} seconds")
