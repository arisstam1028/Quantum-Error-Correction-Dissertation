"""
Purpose:
    Run the MacKay QLDPC Monte Carlo sweep over physical error rates.

Process:
    Repeatedly call the single-frame runner until each probability reaches
    its frame limit or failure limit, then report FER and average iterations.

Theory link:
    Early stopping by failure count saves time at high physical error rates
    while still estimating the frame error rate from observed frames.
"""

from simulation.metrics import frame_error_rate
from simulation.runner import QLDPCRunner


def run_monte_carlo(config) -> list[dict]:
    """
    Execute the configured probability sweep.

    Role in pipeline:
        Aggregates frame-level success and decoder effort into rows consumed
        by reporting and plotting functions.
    """
    runner = QLDPCRunner(config)
    results: list[dict] = []

    for idx, p in enumerate(config.probabilities):
        target_frames = config.frames_per_p[idx]
        max_failures = config.max_failures_per_p[idx]

        failures = 0
        total_frames = 0
        total_iters = 0

        if config.verbose:
            print(f'\n Running p  {p:.4f} ')

        while total_frames < target_frames and failures < max_failures:
            single = runner.run_single_frame(p)
            total_frames += 1
            total_iters += single.iterations_used_x + single.iterations_used_z

            if not single.success:
                failures += 1

        fer = frame_error_rate(failures, total_frames)
        avg_iterations = total_iters / max(total_frames, 1)

        row = {
            "p": p,
            "frames": total_frames,
            "failures": failures,
            "fer": fer,
            "avg_iterations": avg_iterations,
        }
        results.append(row)

        if config.verbose:
            print(f'p{p:.4f} | frames{total_frames} | failures{failures} | FER{fer:.6e} | avg_iter{avg_iterations:.3f}')

    return results
