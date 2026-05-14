import matplotlib.pyplot as plt

def draw_shor_stabilizer_graph(save_path=None, dpi=600):

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Data qubit positions
    qubits = {
        1:(0,2), 2:(0,1), 3:(0,0),
        2.5:(1.5,2), 3.5:(1.5,1), 4.5:(1.5,0),  # spacing helper
        4:(3,2), 5:(3,1), 6:(3,0),
        7:(6,2), 8:(6,1), 9:(6,0)
    }

    # Only real qubits
    qubit_nodes = [1,2,3,4,5,6,7,8,9]

    # Stabilizer positions
    z_stabs = {
        "Z12": (1.5, 1.5),
        "Z23": (1.5, 0.5),
        "Z45": (4.5, 1.5),
        "Z56": (4.5, 0.5),
        "Z78": (7.5, 1.5),
        "Z89": (7.5, 0.5),
    }

    x_stabs = {
        "X123456": (3, 3.1),
        "X456789": (6, 3.1),
    }

    # Draw qubits
    for q in qubit_nodes:
        x, y = qubits[q]
        ax.scatter(x, y, s=700)
        ax.text(x, y, f"$q_{q}$", ha="center", va="center",
                color="white", fontsize=11, weight="bold")

    # Draw stabilizers
    def draw_stabilizer(x, y, label, kind):
        ax.scatter(x, y, s=900, marker="s")
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, color="white")
        if kind == "X":
            ax.text(x, y+0.4, "X", ha="center", fontsize=12, weight="bold")
        else:
            ax.text(x, y+0.4, "Z", ha="center", fontsize=12, weight="bold")

    for k,(x,y) in z_stabs.items():
        draw_stabilizer(x,y,k,"Z")

    for k,(x,y) in x_stabs.items():
        draw_stabilizer(x,y,k,"X")

    # Edges
    edges = [
        ("Z12",1),("Z12",2),
        ("Z23",2),("Z23",3),
        ("Z45",4),("Z45",5),
        ("Z56",5),("Z56",6),
        ("Z78",7),("Z78",8),
        ("Z89",8),("Z89",9),
        ("X123456",1),("X123456",2),("X123456",3),
        ("X123456",4),("X123456",5),("X123456",6),
        ("X456789",4),("X456789",5),("X456789",6),
        ("X456789",7),("X456789",8),("X456789",9),
    ]

    def pos(node):
        if isinstance(node, int):
            return qubits[node]
        if node in z_stabs:
            return z_stabs[node]
        return x_stabs[node]

    for s,q in edges:
        xs,ys = pos(s)
        xq,yq = pos(q)
        ax.plot([xs,xq],[ys,yq],linewidth=1.5)

    ax.set_xlim(-1,9)
    ax.set_ylim(-1,4)
    plt.title("Shor (9,1) Code Stabilizer Graph", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    draw_shor_stabilizer_graph()
    