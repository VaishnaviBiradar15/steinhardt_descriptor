import sys

sys.path.insert(
    0,
    r"C:\Users\vaish\Compute_descriptors\steinhardt"
)
import numpy as np
import pytest
from ase.build import bulk

from steinhardt import (bond_order_parameters, global_bond_order_parameters,
                        neighbour_csr, steinhardt_from_atoms, wigner_3j)

TOL = 1.0e-6


def fcc():
    return bulk("Al", "fcc", a=4.05, cubic=True) * (3, 3, 3), 3.2


def bcc8():
    return bulk("Fe", "bcc", a=2.87, cubic=True) * (3, 3, 3), 2.7


def bcc14():
    return bulk("Fe", "bcc", a=2.87, cubic=True) * (3, 3, 3), 3.0


def simple_cubic():
    return bulk("Po", "sc", a=3.0, cubic=True) * (3, 3, 3), 3.3


def hcp():
    return bulk("Mg", "hcp", a=3.21, c=3.21 * np.sqrt(8.0 / 3.0)) * (4, 4, 4), 3.5


@pytest.mark.parametrize(
    "builder,n_bonds,q4,q6,w4hat,w6hat",
    [
        (fcc, 12, 0.190941, 0.574524, -0.159317, -0.013161),
        (hcp, 12, 0.097222, 0.484762, 0.134097, -0.012442),
        (bcc8, 8, 0.509175, 0.628539, -0.159317, 0.013161),
        (bcc14, 14, 0.036370, 0.510688, 0.159317, 0.013161),
        (simple_cubic, 6, 0.763763, 0.353553, 0.159317, 0.013161),
    ],
)
def test_reference_values(builder, n_bonds, q4, q6, w4hat, w6hat):
    atoms, cutoff = builder()
    result = steinhardt_from_atoms(atoms, cutoff=cutoff, l_values=(4, 6))
    assert np.all(result["n_bonds"] == n_bonds)
    assert result["Q"][:, 0] == pytest.approx(q4, abs=1e-5)
    assert result["Q"][:, 1] == pytest.approx(q6, abs=1e-5)
    assert result["What"][:, 0] == pytest.approx(w4hat, abs=1e-5)
    assert result["What"][:, 1] == pytest.approx(w6hat, abs=1e-5)


def test_odd_l_vanishes_in_centrosymmetric_crystal():
    atoms, cutoff = fcc()
    result = steinhardt_from_atoms(atoms, cutoff=cutoff, l_values=(1, 2, 3, 5))
    assert np.max(np.abs(result["Q"])) < 1e-12


def test_translation_invariance():
    atoms, cutoff = fcc()
    reference = steinhardt_from_atoms(atoms, cutoff=cutoff, l_values=(4, 6))["Q"]
    shifted = atoms.copy()
    shifted.translate([0.37, -1.21, 2.03])
    shifted.wrap()
    moved = steinhardt_from_atoms(shifted, cutoff=cutoff, l_values=(4, 6))["Q"]
    assert np.allclose(np.sort(reference, axis=0), np.sort(moved, axis=0), atol=TOL)


def test_rotation_invariance():
    atoms, cutoff = fcc()
    reference = steinhardt_from_atoms(atoms, cutoff=cutoff, l_values=(4, 6, 8))
    rotated = atoms.copy()
    rotated.rotate(37.0, (0.3, 0.5, 0.81), rotate_cell=True)
    moved = steinhardt_from_atoms(rotated, cutoff=cutoff, l_values=(4, 6, 8))
    assert np.allclose(reference["Q"], moved["Q"], atol=TOL)
    assert np.allclose(reference["What"], moved["What"], atol=TOL)


def test_permutation_invariance():
    atoms, cutoff = fcc()
    reference = steinhardt_from_atoms(atoms, cutoff=cutoff, l_values=(4, 6))["Q"]
    order = np.random.default_rng(0).permutation(len(atoms))
    shuffled = atoms[order]
    moved = steinhardt_from_atoms(shuffled, cutoff=cutoff, l_values=(4, 6))["Q"]
    assert np.allclose(reference[order], moved, atol=TOL)


def test_global_matches_local_for_perfect_crystal():
    atoms, cutoff = fcc()
    disp_flat, indptr, _ = neighbour_csr(atoms, cutoff)
    local = bond_order_parameters(disp_flat, indptr, l_values=(4, 6))
    glob = global_bond_order_parameters(disp_flat, l_values=(4, 6))
    assert glob["n_bonds"] == int(local["n_bonds"].sum())
    assert np.allclose(glob["Q"], local["Q"][0], atol=TOL)


def closed_form_3j_zero_m(j1, j2, j3):
    from math import factorial

    total = j1 + j2 + j3
    if total % 2 != 0:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    half = total // 2
    sign = 1.0 if half % 2 == 0 else -1.0
    radicand = (factorial(total - 2 * j1) * factorial(total - 2 * j2)
                * factorial(total - 2 * j3) / factorial(total + 1))
    ratio = factorial(half) / (factorial(half - j1) * factorial(half - j2)
                               * factorial(half - j3))
    return sign * np.sqrt(radicand) * ratio


def test_wigner_3j_known_values():
    assert wigner_3j(0, 0, 0, 0, 0, 0) == pytest.approx(1.0)
    assert wigner_3j(1, 1, 0, 0, 0, 0) == pytest.approx(-1.0 / np.sqrt(3.0))
    assert wigner_3j(2, 2, 2, 0, 0, 0) == pytest.approx(-np.sqrt(2.0 / 35.0))
    assert wigner_3j(1, 1, 1, 0, 0, 0) == pytest.approx(0.0)
    assert wigner_3j(2, 2, 2, 1, 1, 1) == pytest.approx(0.0)
    for l in range(0, 11):
        assert wigner_3j(l, l, l, 0, 0, 0) == pytest.approx(
            closed_form_3j_zero_m(l, l, l), abs=1e-12)


def test_wigner_3j_permutation_symmetry():
    rng = np.random.default_rng(3)
    for _ in range(50):
        l = int(rng.integers(1, 7))
        m1 = int(rng.integers(-l, l + 1))
        m2 = int(rng.integers(-l, l + 1))
        m3 = -m1 - m2
        if abs(m3) > l:
            continue
        a = wigner_3j(l, l, l, m1, m2, m3)
        b = wigner_3j(l, l, l, m2, m3, m1)
        assert a == pytest.approx(b, abs=1e-12)
