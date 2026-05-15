import matplotlib.pyplot as plt

def depolarizing_right_angle_clean(save_path=None, dpi=600):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Layout
    x_trunk = 0.0

    # Short arrows
    x_arrow_end = 4.2

    # Text kept close
    x_branch = 4.7

    y_top = 5.2
    y_I = 4.1
    y_X = 3.0
    y_Z = 1.9
    y_Y = 0.8

    # Text positions
    x_op = x_branch + 0.2
    x_state = x_branch + 1.0

    # Font size (uniform)
    FS = 22

    # Helpers
    def arrow(a, b, lw=2.2):
        ax.annotate(
            "",
            xy=b, xytext=a,
            arrowprops=dict(arrowstyle="-|>", lw=lw, color="black")
        )

    def line(a, b, lw=2.2):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            lw=lw,
            color="black"   # < force black
        )

    # Root
    ax.plot(x_trunk, y_top, "ko", markersize=8)
    ax.text(
        x_trunk, y_top + 0.45,
        r"$a|0\rangle + b|1\rangle$",
        fontsize=FS, ha="center", va="bottom"
    )

    # Vertical trunk (connector, not arrow)
    line((x_trunk, y_top), (x_trunk, y_Y))

    # Branches
    def branch(y, op_tex, out_tex, prob_tex):
        arrow((x_trunk, y), (x_arrow_end, y))

        ax.text(
            x_trunk - 0.55, y,
            prob_tex, fontsize=FS, ha="right", va="center"
        )

        ax.text(
            x_op, y, op_tex,
            fontsize=FS, ha="left", va="center", weight="bold"
        )
        ax.text(
            x_state, y, out_tex,
            fontsize=FS, ha="left", va="center"
        )

    branch(y_I, r"$\mathbf{I}$", r"$a|0\rangle + b|1\rangle$", r"$(1-p)$")
    branch(y_X, r"$\mathbf{X}$", r"$a|1\rangle + b|0\rangle$", r"$\frac{p}{3}$")
    branch(y_Z, r"$\mathbf{Z}$", r"$a|0\rangle - b|1\rangle$", r"$\frac{p}{3}$")
    branch(y_Y, r"$\mathbf{Y}$", r"$i a|1\rangle - i b|0\rangle$", r"$\frac{p}{3}$")

    # Single p label
    ax.text(
        x_trunk - 0.55, (y_I + y_X) / 2,
        r"$p$", fontsize=FS, ha="right", va="center"
    )

    # Limits
    ax.set_xlim(-2.0, x_state + 2.6)
    ax.set_ylim(0.0, 6.0)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    depolarizing_right_angle_clean()
