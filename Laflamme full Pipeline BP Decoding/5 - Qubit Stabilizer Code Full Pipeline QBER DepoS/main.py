from simulation_runner import (
    FiveQubitCode,
    SimulationConfig,
    SimulationReport,
    SimulationRunner,
)


def main() -> None:
    code = FiveQubitCode()
    runner = SimulationRunner(code)

    probabilities = [i / 200 for i in range(1, 21)]

    config = SimulationConfig(
        probabilities=probabilities,
        frames_per_probability=[200000 for _ in probabilities],
        seed=7,
        failure_threshold=None,
        bp_max_iters=30,
    )

    results = runner.run(config)

    report = SimulationReport(code)
    report.print_summary(results)
    report.show_plot(results)


if __name__ == "__main__":
    main()