import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def read_structure(filename, file_format=None, pbc=None):
    atoms = read(filename, format=file_format)
    if pbc is not None:
        atoms.set_pbc(pbc)
    return atoms


def neighbour_csr(atoms, cutoff):
    cutoff = float(cutoff)
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive, received %r" % cutoff)

    i_index, j_index, displacement = neighbor_list(
        "ijD", atoms, cutoff=cutoff, self_interaction=False)

    n_atoms = len(atoms)
    counts = np.bincount(i_index, minlength=n_atoms)
    indptr = np.zeros(n_atoms + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    order = np.argsort(i_index, kind="stable")
    disp_flat = np.ascontiguousarray(displacement[order], dtype=np.float64)
    neighbour_index = np.ascontiguousarray(j_index[order], dtype=np.int64)

    if disp_flat.shape[0] == 0:
        disp_flat = np.zeros((0, 3), dtype=np.float64)

    return disp_flat, indptr, neighbour_index


def build_neighbours(filename, file_format=None, cutoff=3.5, pbc=None):
    atoms = read_structure(filename, file_format=file_format, pbc=pbc)
    disp_flat, indptr, neighbour_index = neighbour_csr(atoms, cutoff)
    return atoms, disp_flat, indptr, neighbour_index
