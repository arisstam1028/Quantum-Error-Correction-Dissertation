import matplotlib.pyplot as plt


def plot_fer(results: list[dict], title: str = "FER vs p") -> None:
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


def plot_iterations(results: list[dict], title: str = "Average iterations vs p") -> None:
    ps = [r["p"] for r in results]
    avg_iters = [r["avg_iterations"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ps, avg_iters, marker="o")
    plt.xlabel("Depolarizing probability p")
    plt.ylabel("Average decoding iterations")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_fer_compare(all_results, title="FER Comparison"):
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


def plot_iterations_compare(all_results, title="Average Iterations Comparison"):
    plt.figure(figsize=(8, 6))

    for label, results in all_results.items():
        ps = [row["p"] for row in results]
        avg_iters = [row["avg_iterations"] for row in results]
        plt.plot(ps, avg_iters, marker="o", label=label)

    plt.xlabel("Physical error rate p")
    plt.ylabel("Average decoding iterations")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()