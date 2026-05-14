"""
Purpose:
    Plot FER and average iteration curves for MacKay QLDPC runs.

Process:
    Extract probability, FER, and iteration columns from result rows and draw
    either single-run or comparison figures.

Theory link:
    FER tracks logical failure probability, while average iterations measures
    decoder effort as physical noise increases.
"""

import matplotlib.pyplot as plt


def plot_fer(results: list[dict], title: str = "FER vs p") -> None:
    """
    Plot frame error rate against physical error probability.
    """
    ps = [r["p"] for r in results]
    fers = [r["fer"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.semilogy(ps, fers, marker="o")
    plt.xlabel("Depolarizing probability p")
    plt.ylabel("Frame Error Rate (FER)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_avg_iterations(results: list[dict], title: str = "Average Iterations vs p") -> None:
    """
    Plot average decoder iterations against physical error probability.
    """
    ps = [r["p"] for r in results]
    avg_iterations = [r["avg_iterations"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.semilogy(ps, avg_iterations, marker="o")
    plt.xlabel("Depolarizing probability p")
    plt.ylabel("Average iterations")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_fer_compare(all_results, title="FER Comparison"):
    """
    Plot multiple FER curves on one axis.
    """
    plt.figure(figsize=(8, 6))

    for label, results in all_results.items():
        ps = [row["p"] for row in results]
        fers = [row["fer"] for row in results]
        plt.semilogy(ps, fers, marker="o", label=label)

    plt.xlabel("Physical error rate p")
    plt.ylabel("Frame Error Rate (FER)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_avg_iterations_compare(all_results, title="Average Iterations Comparison"):
    """
    Plot multiple average-iteration curves on one axis.
    """
    plt.figure(figsize=(8, 6))

    for label, results in all_results.items():
        ps = [row["p"] for row in results]
        avg_iterations = [row["avg_iterations"] for row in results]
        plt.semilogy(ps, avg_iterations, marker="o", label=label)

    plt.xlabel("Physical error rate p")
    plt.ylabel("Average iterations")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
