import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

# Add project root so the steinhardt package can be imported
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_DIR)

from steinhardt import calculate_steinhardt_descriptors


STRUCTURE_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "..", "structures")
)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Create results folder automatically
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# REFERENCE STRUCTURES
# ---------------------------------------------------------

CASES = [
    ("fcc_perfect_crystal.lmp", 3.2, "FCC"),
    ("bcc_perfect_crystal.lmp", 3.0, "BCC"),
    ("hcp_perfect_crystal.lmp", 3.5, "HCP"),
    ("sc_perfect_crystal.lmp", 3.3, "SC"),
]

# Calculate l = 2, 3, 4, 5, 6
L_VALUES = (2, 3, 4, 5, 6)


def main():

    labels = []

    Q_values = []
    W_hat_values = []

    n_l = len(L_VALUES)

    # -----------------------------------------------------
    # CALCULATE LOCAL DESCRIPTORS
    # -----------------------------------------------------

    for filename, cutoff, label in CASES:

        path = os.path.join(STRUCTURE_DIR, filename)

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        features = calculate_steinhardt_descriptors(
            path,
            file_format="lammps-data",
            cutoff=cutoff,
            l_values=L_VALUES,
            pbc=(True, True, True),
            include_q=True,
            include_w=False,
            include_w_hat=True,
        )

        # Output layout:
        #
        # [Q2 Q3 Q4 Q5 Q6 | What2 What3 What4 What5 What6]

        Q = features[:, :n_l]
        W_hat = features[:, n_l:]

        # Perfect crystals have equivalent local environments,
        # therefore the mean represents the crystal fingerprint.
        Q_values.append(Q.mean(axis=0))
        W_hat_values.append(W_hat.mean(axis=0))

        labels.append(label)

    Q_values = np.asarray(Q_values)
    W_hat_values = np.asarray(W_hat_values)

    # =====================================================
    # FIGURE 1
    # STEINHARDT Q_l FINGERPRINT HEATMAP
    # =====================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    image = ax.imshow(
        Q_values,
        aspect="auto"
    )

    ax.set_xticks(np.arange(len(L_VALUES)))
    ax.set_xticklabels(
        [rf"$Q_{{{l}}}$" for l in L_VALUES]
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_title("Steinhardt bond-order fingerprint")

    # Write numerical value inside every cell
    for i in range(Q_values.shape[0]):
        for j in range(Q_values.shape[1]):

            ax.text(
                j,
                i,
                f"{Q_values[i, j]:.3f}",
                ha="center",
                va="center"
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$Q_l$")

    fig.tight_layout()

    output = os.path.join(
        RESULTS_DIR,
        "steinhardt_Q_fingerprint_heatmap.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # FIGURE 2
    # Q4 AND Q6 CRYSTAL STRUCTURE FINGERPRINT
    # =====================================================

    # Find positions of l = 4 and l = 6
    i4 = L_VALUES.index(4)
    i6 = L_VALUES.index(6)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        Q_values[:, i4],
        width,
        label=r"$Q_4$"
    )

    ax.bar(
        x + width / 2,
        Q_values[:, i6],
        width,
        label=r"$Q_6$"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Steinhardt invariant")
    ax.set_title(
        r"Crystal structure fingerprint using $Q_4$ and $Q_6$"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output = os.path.join(
        RESULTS_DIR,
        "crystal_fingerprint_Q4_Q6.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # FIGURE 3
    # NORMALIZED THIRD-ORDER INVARIANTS
    # =====================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        W_hat_values[:, i4],
        width,
        label=r"$\hat{W}_4$"
    )

    ax.bar(
        x + width / 2,
        W_hat_values[:, i6],
        width,
        label=r"$\hat{W}_6$"
    )

    ax.axhline(
        0.0,
        linewidth=1.0
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Normalized third-order invariant")

    ax.set_title(
        r"Crystal symmetry fingerprint using "
        r"$\hat{W}_4$ and $\hat{W}_6$"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output = os.path.join(
        RESULTS_DIR,
        "crystal_fingerprint_What4_What6.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # -----------------------------------------------------
    # PRINT OUTPUT LOCATION
    # -----------------------------------------------------

    print("\nResults saved in:")
    print(RESULTS_DIR)

    print("\nGenerated figures:")
    print("1. steinhardt_Q_fingerprint_heatmap.png")
    print("2. crystal_fingerprint_Q4_Q6.png")
    print("3. crystal_fingerprint_What4_What6.png")


if __name__ == "__main__":
    main()