import matplotlib.pyplot as plt

def depolarizing_branch_diagram(save_path=None, dpi=600):

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    #  Coordinates (DATA coordinates, not normalized) 
    x_start = 0
    x_end   = 10

    y_start = 0
    y_I = 0
    y_X = -1.2
    y_Z = -2.4
    y_Y = -3.6

    #  Start node 
    ax.plot(x_start, y_start, "ko", markersize=8)

    # Left input state
    ax.text(
        x_start - 0.4, y_start,
        r"$a|0\rangle + b|1\rangle$",
        fontsize=22, ha="right", va="center"
    )

    #  Arrow helper 
    def draw_branch(y, prob, op, out_state):
        ax.annotate(
            "",
            xy=(x_end, y),
            xytext=(x_start, y_start),
            arrowprops=dict(arrowstyle="-|>", lw=1.8)
        )

        # Probability label
        ax.text(
            (x_start + x_end) / 2,
            (y_start + y) / 2 + 0.25,
            prob,
            fontsize=18,
            ha="center"
        )

        # Operator label
        ax.text(
            x_end - 2.0,
            y + 0.2,
            op,
            fontsize=22,
            ha="left",
            weight="bold"
        )

        # Output state
        ax.text(
            x_end + 0.2,
            y,
            out_state,
            fontsize=22,
            ha="left",
            va="center"
        )

    #  Branches 
    draw_branch(
        y_I,
        r"$(1-p)$",
        r"$\mathbf{I}$",
        r"$a|0\rangle + b|1\rangle$"
    )

    draw_branch(
        y_X,
        r"$\frac{p}{3}$",
        r"$\mathbf{X}$",
        r"$a|1\rangle + b|0\rangle$"
    )

    draw_branch(
        y_Z,
        r"$\frac{p}{3}$",
        r"$\mathbf{Z}$",
        r"$a|0\rangle - b|1\rangle$"
    )

    draw_branch(
        y_Y,
        r"$\frac{p}{3}$",
        r"$\mathbf{Y}$",
        r"$i a|1\rangle - i b|0\rangle$"
    )

    #  Axis limits (CRITICAL) 
    ax.set_xlim(-2, 13)
    ax.set_ylim(-4.5, 1)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    depolarizing_branch_diagram()
