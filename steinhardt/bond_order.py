import numpy as np
from numba import njit, prange
from .spherical_harmonics import spherical_harmonics_point

# second-order rotational invariant:
# Q_l = sqrt(4*pi/(2l+1) * sum_m |q_lm|^2)

FOUR_PI = 4.0 * np.pi


#calc of Ylm
@njit(cache=True)
def accumulate_qlm(disp, l_max, qlm, P, Y):
    n_bonds = disp.shape[0]
    used = 0
    for b in range(n_bonds):
        x = disp[b, 0]
        y = disp[b, 1]
        z = disp[b, 2]
        if x * x + y * y + z * z < 1.0e-24:
            continue
        spherical_harmonics_point(x, y, z, l_max, P, Y)
        for l in range(l_max + 1):
            for mi in range(2 * l + 1):
                qlm[l, mi] += Y[l, mi]
        used += 1
    return used

#normalize Ylm/Nb = qlm
@njit(cache=True)
def normalise_qlm(qlm, l_max, count):
    if count == 0:
        return
    scale = 1.0 / count
    for l in range(l_max + 1):
        for mi in range(2 * l + 1):
            qlm[l, mi] = qlm[l, mi] * scale

#calculate the squared norm of qml for later calculation 
#of second order invariant
@njit(cache=True)
def qlm_norm_squared(qlm, l):
    total = 0.0
    for mi in range(2 * l + 1):
        c = qlm[l, mi]
        total += c.real * c.real + c.imag * c.imag
    return total

#second order invariant
@njit(cache=True)
def order_parameter_q(qlm, l):
    return np.sqrt(FOUR_PI / (2.0 * l + 1.0) * qlm_norm_squared(qlm, l))

#Third order invariant
@njit(cache=True)
def order_parameter_w(qlm, l, table):
    total = complex(0.0, 0.0)
    size = 2 * l + 1
    for i1 in range(size):
        m1 = i1 - l
        for i2 in range(size):
            m2 = i2 - l
            m3 = -m1 - m2
            if abs(m3) > l:
                continue
            coefficient = table[i1, i2]
            if coefficient == 0.0:
                continue
            total += coefficient * qlm[l, i1] * qlm[l, i2] * qlm[l, m3 + l]
    return total.real

#for every l calculate normalized third invariant
# returns nothing, but fills q_out, w_out, w_hat_out in place.
@njit(cache=True)
def bond_order_from_qlm(qlm, l_values, tables, q_out, w_out, w_hat_out):
    for k in range(l_values.shape[0]):
        l = l_values[k]
        norm2 = qlm_norm_squared(qlm, l)
        q_out[k] = np.sqrt(FOUR_PI / (2.0 * l + 1.0) * norm2)
        w = order_parameter_w(qlm, l, tables[k])
        w_out[k] = w
        if norm2 > 1.0e-30:
            w_hat_out[k] = w / (norm2 ** 1.5)
        else:
            w_hat_out[k] = 0.0

#per atom loop for every atom in single structure, get its neighbours, 
#calc qml, normalize it, calculate Ql, Wl, 
# returns nothing, but fills q_all, w_all, w_hat_all, counts.
@njit(cache=True, parallel=True)
def steinhardt_batch_kernel(disp_flat, indptr, l_max, l_values, tables,
                            q_all, w_all, w_hat_all, counts):
    n_atoms = indptr.shape[0] - 1
    for a in prange(n_atoms):
        P = np.zeros((l_max + 1, l_max + 1), dtype=np.float64)
        Y = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
        qlm = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
        disp = disp_flat[indptr[a]:indptr[a + 1]]
        #calc of  Ylm (add all neighbouring atoms)
        used = accumulate_qlm(disp, l_max, qlm, P, Y)
        counts[a] = used
        # Normalize qlm before calculating Ql, Wl and W_hat_l
        normalise_qlm(qlm, l_max, used)
        bond_order_from_qlm(qlm, l_values, tables,
                            q_all[a], w_all[a], w_hat_all[a])

#qml for entire structure not for every atom
#global_qlm_kernel() → explicitly returns qlm, used
@njit(cache=True)
def global_qlm_kernel(disp_flat, l_max):
    P = np.zeros((l_max + 1, l_max + 1), dtype=np.float64)
    Y = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
    qlm = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
    #calc of  Ylm (add all neighbouring atoms)
    used = accumulate_qlm(disp_flat, l_max, qlm, P, Y)
    normalise_qlm(qlm, l_max, used)
    return qlm, used