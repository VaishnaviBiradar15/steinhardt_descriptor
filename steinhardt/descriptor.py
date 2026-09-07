import sys

sys.path.insert(
    0,
    r"C:\Users\vaish\Compute_descriptors\steinhardt"
)

import numpy as np

from .bond_order import (bond_order_from_qlm, global_qlm_kernel,
                         steinhardt_batch_kernel)
from .neighbours import build_neighbours, neighbour_csr
from .wigner import build_wigner_tables

DEFAULT_L_VALUES = (4, 6)

#for single l we will have 3 invariant , Ql, Wl, W hat l (normalized one)
def feature_names(l_values=DEFAULT_L_VALUES, include_q=True, include_w=False,
                  include_w_hat=False):
    names = []
    if include_q:
        names += ["Q%d" % l for l in l_values]
    if include_w:
        names += ["W%d" % l for l in l_values]
    if include_w_hat:
        names += ["What%d" % l for l in l_values]
    return names

#local per-atom descriptor function
def bond_order_parameters(disp_flat, indptr, l_values=DEFAULT_L_VALUES):
    l_values = np.ascontiguousarray([int(l) for l in l_values], dtype=np.int64)
    if l_values.size == 0:
        raise ValueError("l_values must contain at least one degree l")
    if np.any(l_values < 0):
        raise ValueError("l_values must be non-negative")

    l_max = int(l_values.max())
    tables = build_wigner_tables(l_values)

    disp_flat = np.ascontiguousarray(disp_flat, dtype=np.float64)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    n_atoms = indptr.shape[0] - 1

    q_all = np.zeros((n_atoms, l_values.size), dtype=np.float64)
    w_all = np.zeros((n_atoms, l_values.size), dtype=np.float64)
    w_hat_all = np.zeros((n_atoms, l_values.size), dtype=np.float64)
    counts = np.zeros(n_atoms, dtype=np.int64)

    if n_atoms > 0:
        steinhardt_batch_kernel(disp_flat, indptr, l_max, l_values, tables,
                                q_all, w_all, w_hat_all, counts)

    return {"Q": q_all, "W": w_all, "What": w_hat_all, "n_bonds": counts,
            "l_values": l_values}


def global_bond_order_parameters(disp_flat, l_values=DEFAULT_L_VALUES):
    l_values = np.ascontiguousarray([int(l) for l in l_values], dtype=np.int64)
    l_max = int(l_values.max())
    tables = build_wigner_tables(l_values)

    disp_flat = np.ascontiguousarray(disp_flat, dtype=np.float64)
    qlm, used = global_qlm_kernel(disp_flat, l_max)

    q_out = np.zeros(l_values.size, dtype=np.float64)
    w_out = np.zeros(l_values.size, dtype=np.float64)
    w_hat_out = np.zeros(l_values.size, dtype=np.float64)
    bond_order_from_qlm(qlm, l_values, tables, q_out, w_out, w_hat_out)

    return {"Q": q_out, "W": w_out, "What": w_hat_out, "n_bonds": int(used),
            "l_values": l_values}


def calculate_steinhardt_descriptors(filename, file_format="lammps-data",
                                     cutoff=3.5, l_values=DEFAULT_L_VALUES,
                                     pbc=None, include_q=True, include_w=False,
                                     include_w_hat=False, neighbours=None):
    if not (include_q or include_w or include_w_hat):
        raise ValueError("at least one of include_q, include_w, include_w_hat "
                         "must be True")

    if neighbours is None:
        if pbc is None and file_format == "lammps-data":
            raise ValueError(
                "pbc must be given explicitly for 'lammps-data' files, e.g. "
                "pbc=(True, True, True); the format carries no boundary flags")
        atoms, disp_flat, indptr, _ = build_neighbours(
            filename, file_format=file_format, cutoff=cutoff, pbc=pbc)
    else:
        atoms, disp_flat, indptr = neighbours[0], neighbours[1], neighbours[2]

    result = bond_order_parameters(disp_flat, indptr, l_values=l_values)

    blocks = []
    if include_q:
        blocks.append(result["Q"])
    if include_w:
        blocks.append(result["W"])
    if include_w_hat:
        blocks.append(result["What"])

    return np.concatenate(blocks, axis=1)


def calculate_global_steinhardt(filename, file_format="lammps-data", cutoff=3.5,
                                l_values=DEFAULT_L_VALUES, pbc=None,
                                neighbours=None):
    if neighbours is None:
        if pbc is None and file_format == "lammps-data":
            raise ValueError(
                "pbc must be given explicitly for 'lammps-data' files, e.g. "
                "pbc=(True, True, True); the format carries no boundary flags")
        _, disp_flat, _, _ = build_neighbours(
            filename, file_format=file_format, cutoff=cutoff, pbc=pbc)
    else:
        disp_flat = neighbours[1]

    return global_bond_order_parameters(disp_flat, l_values=l_values)


def steinhardt_from_atoms(atoms, cutoff=3.5, l_values=DEFAULT_L_VALUES):
    disp_flat, indptr, _ = neighbour_csr(atoms, cutoff)
    return bond_order_parameters(disp_flat, indptr, l_values=l_values)
