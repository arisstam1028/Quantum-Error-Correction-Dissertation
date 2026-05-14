import matplotlib.pyplot as plt

def depolarizing_channel_final(save_path=None, dpi=600):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")

    # -------------------------
    # Layout
    # -------------------------
    x_trunk = 0.0
    x_branch = 2.6

    x_op = x_branch + 0.18
    x_state = x_branch + 0.75
    x_mat = x_branch + 3.25

    y_top = 5.8
    y_I = 4.5
    y_X = 3.3
    y_Z = 2.1
    y_Y = 0.9

    # -------------------------
    # Unified text size
    # -------------------------
    FS = 22

    # -------------------------
    # Helpers
    # -------------------------
    def arrow(a, b, lw=2.2):
        ax.annotate(
            "", xy=b, xytext=a,
            arrowprops=dict(arrowstyle="-|>", lw=lw, color="black")
        )

    def line(a, b, lw=2.2):
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=lw, color="black")

    def branch(y, op_tex, out_tex, prob_tex):
        arrow((x_trunk, y), (x_branch, y))
        ax.text(x_trunk - 0.55, y, prob_tex,
                fontsize=FS, ha="right", va="center")
        ax.text(x_op, y, op_tex,
                fontsize=FS, ha="left", va="center", weight="bold")
        ax.text(x_state, y, out_tex,
                fontsize=FS, ha="left", va="center")

    def draw_2x2_matrix(x, y, a11, a12, a21, a22):
        w = 1.35
        h = 1.05
        tick = 0.18
        lw = 2.2

        left = x
        right = x + w
        top = y + h / 2
        bottom = y - h / 2

        ax.plot([left, left], [bottom, top], lw=lw, color="black")
        ax.plot([left, left + tick], [top, top], lw=lw, color="black")
        ax.plot([left, left + tick], [bottom, bottom], lw=lw, color="black")

        ax.plot([right, right], [bottom, top], lw=lw, color="black")
        ax.plot([right - tick, right], [top, top], lw=lw, color="black")
        ax.plot([right - tick, right], [bottom, bottom], lw=lw, color="black")

        cx1 = left + 0.35 * w
        cx2 = left + 0.78 * w
        cy1 = y + 0.24 * h
        cy2 = y - 0.24 * h

        ax.text(cx1, cy1, a11, fontsize=FS, ha="center", va="center")
        ax.text(cx2, cy1, a12, fontsize=FS, ha="center", va="center")
        ax.text(cx1, cy2, a21, fontsize=FS, ha="center", va="center")
        ax.text(cx2, cy2, a22, fontsize=FS, ha="center", va="center")

    # -------------------------
    # Diagram title
    # -------------------------
    ax.text(
        x_trunk + 2.5, y_top + 1.2,
        r"$\mathrm{Depolarisation\ Channel}$",
        fontsize=FS, ha="center", va="bottom"
    )

    # -------------------------
    # Root
    # -------------------------
    ax.plot(x_trunk, y_top, "ko", markersize=8)
    ax.text(x_trunk, y_top + 0.55,
            r"$a|0\rangle + b|1\rangle$",
            fontsize=FS, ha="center", va="bottom")

    line((x_trunk, y_top), (x_trunk, y_Y))

    ax.text(x_trunk - 0.55, (y_I + y_X) / 2,
            r"$p$", fontsize=FS, ha="right", va="center")

    # -------------------------
    # Branches
    # -------------------------
    branch(y_I, r"$\mathbf{I}$", r"$a|0\rangle + b|1\rangle$", r"$(1-p)$")
    branch(y_X, r"$\mathbf{X}$", r"$a|1\rangle + b|0\rangle$", r"$\frac{p}{3}$")
    branch(y_Z, r"$\mathbf{Z}$", r"$a|0\rangle - b|1\rangle$", r"$\frac{p}{3}$")
    branch(y_Y, r"$\mathbf{Y}$", r"$i a|1\rangle - i b|0\rangle$", r"$\frac{p}{3}$")

    # -------------------------
    # Pauli operators title
    # -------------------------
    ax.text(x_mat + 0.70, y_top - 0.25,
            r"$\mathrm{Pauli\ operators}$",
            fontsize=FS, ha="center", va="center")

    draw_2x2_matrix(x_mat, y_I, "1", "0", "0", "1")
    draw_2x2_matrix(x_mat, y_X, "0", "1", "1", "0")
    draw_2x2_matrix(x_mat, y_Z, "1", "0", "0", "-1")
    draw_2x2_matrix(x_mat, y_Y, "0", r"$-i$", r"$i$", "0")

    ax.set_xlim(-1.8, x_mat + 3.2)
    ax.set_ylim(0.3, 7.6)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    depolarizing_channel_final()
