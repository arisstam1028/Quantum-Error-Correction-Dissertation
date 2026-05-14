from copy import deepcopy
import time

from config import build_config
from plotting.plotter import (
    plot_fer,
    plot_fer_compare,
    plot_iterations,
    plot_iterations_compare,
)
from simulation.monte_carlo import run_monte_carlo
from simulation.runner import QLDPCFamilyRunner

DECODER_TYPE = "bp"   # "bp", "svns", "scns", "compare"
RUN_FAMILY = False


def main() -> None:
    matrix_module = "matrices.mackay_800_400"
    family_module = "matrices.GB_bicycle_code_5_2_p_family"

    if RUN_FAMILY:
        decoder_types = ["bp", "svns", "scns"] if DECODER_TYPE == "compare" else [DECODER_TYPE]
        base_config = build_config(matrix_module=matrix_module)
        all_results = {}

        for dec in decoder_types:
            config_template = deepcopy(base_config)
            config_template.decoder_type = dec

            family_runner = QLDPCFamilyRunner(
                base_module_path=family_module,
                decoder_type=dec,
                config_template=config_template,
            )
            family_results = family_runner.run_family()
            all_results[dec] = family_results

            print(f"\n=== Family Results ({dec}) ===")
            for label, results in family_results.items():
                print(f"\n--- {label} ---")
                for row in results:
                    print(
                        f"p={row['p']:.4f} | frames={row['frames']} | "
                        f"failures={row['failures']} | FER={row['fer']:.6e} | "
                        f"avg_iter={row['avg_iterations']:.3f}"
                    )

            plot_fer_compare(
                family_results,
                title=f"QLDPC FER Family Comparison ({family_module}, {dec})"
            )
            plot_iterations_compare(
                family_results,
                title=f"QLDPC Iterations Family Comparison ({family_module}, {dec})"
            )
        return

    if DECODER_TYPE != "compare":
        config = build_config(matrix_module=matrix_module)
        config.decoder_type = DECODER_TYPE

        # Enable perturbation here when using the binary BP decoder
        if config.decoder_type == "bp":
            config.enable_random_perturbation = True
            config.perturb_iters = 40
            config.perturb_max_feedbacks = 40
            config.perturb_strength = 0.1

        results = run_monte_carlo(config)

        print("\n=== Final Results ===")
        print(f"Matrix module used: {config.matrix_module}")
        print(f"Decoder used: {config.decoder_type}")
        print(f"Random perturbation enabled: {config.enable_random_perturbation}")

        for row in results:
            print(
                f"p={row['p']:.4f} | frames={row['frames']} | "
                f"failures={row['failures']} | FER={row['fer']:.6e} | "
                f"avg_iter={row['avg_iterations']:.3f}"
            )

        plot_fer(
            results,
            title=f"QLDPC FER vs Depolarizing Probability ({config.matrix_module}, {config.decoder_type})"
        )
        plot_iterations(
            results,
            title=f"QLDPC Average Iterations vs Depolarizing Probability ({config.matrix_module}, {config.decoder_type})"
        )
        return

    decoder_types = ["bp", "svns", "scns"]
    all_results = {}

    for dec in decoder_types:
        config = build_config(matrix_module=matrix_module)
        config.decoder_type = dec

        if dec == "bp":
            config.enable_random_perturbation = True
            config.perturb_iters = 40
            config.perturb_max_feedbacks = 40
            config.perturb_strength = 0.1

        results = run_monte_carlo(config)
        all_results[dec] = results

        print(f"\n=== Final Results ({dec}) ===")
        for row in results:
            print(
                f"p={row['p']:.4f} | frames={row['frames']} | "
                f"failures={row['failures']} | FER={row['fer']:.6e} | "
                f"avg_iter={row['avg_iterations']:.3f}"
            )

    plot_fer_compare(
        all_results,
        title=f"QLDPC FER Comparison ({matrix_module})"
    )
    plot_iterations_compare(
        all_results,
        title=f"QLDPC Average Iterations Comparison ({matrix_module})"
    )


if __name__ == "__main__":
    start_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"\n=== Total Runtime ===")
    print(f"{total_time:.2f} seconds ({total_time / 60:.2f} minutes)")