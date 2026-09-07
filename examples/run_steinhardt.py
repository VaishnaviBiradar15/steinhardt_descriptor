import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from steinhardt import (calculate_global_steinhardt,
                        calculate_steinhardt_descriptors, feature_names)

STRUCTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "structures")

CASES = [
    ("fcc_perfect_crystal.lmp", 3.2),
    ("bcc_perfect_crystal.lmp", 3.0),
    ("hcp_perfect_crystal.lmp", 3.5),
    ("sc_perfect_crystal.lmp", 3.3),
]

L_VALUES = (4, 6)


def main():
    names = feature_names(L_VALUES, include_q=True, include_w=True,
                          include_w_hat=True)
    header = "{:<28}".format("structure") + "".join(
        "{:>12}".format(n) for n in names)
    print(header)
    print("-" * len(header))

    for filename, cutoff in CASES:
        path = os.path.join(STRUCTURE_DIR, filename)
        if not os.path.exists(path):
            print("{:<28}{}".format(filename, "missing"))
            continue

        features = calculate_steinhardt_descriptors(
            path, file_format="lammps-data", cutoff=cutoff,
            l_values=L_VALUES, pbc=(True, True, True),
            include_q=True, include_w=True, include_w_hat=True)

        row = np.round(features.mean(axis=0), 6)
        print("{:<28}".format(filename) + "".join(
            "{:>12.6f}".format(v) for v in row))

    print()
    path = os.path.join(STRUCTURE_DIR, CASES[0][0])
    if os.path.exists(path):
        glob = calculate_global_steinhardt(
            path, file_format="lammps-data", cutoff=CASES[0][1],
            l_values=L_VALUES, pbc=(True, True, True))
        print("global bond average over {} bonds".format(glob["n_bonds"]))
        print("Q  =", np.round(glob["Q"], 6))
        print("W  =", np.round(glob["W"], 6))
        print("Wh =", np.round(glob["What"], 6))


if __name__ == "__main__":
    main()
