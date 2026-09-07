import numpy as np
from numba import njit


@njit(cache=True)
def normalised_legendre(costheta, sintheta, l_max, P):
    P[0, 0] = 0.28209479177387814
    for m in range(0, l_max + 1):
        if m > 0:
            P[m, m] = -np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sintheta * P[m - 1, m - 1]
        if m + 1 <= l_max:
            P[m + 1, m] = np.sqrt(2.0 * m + 3.0) * costheta * P[m, m]
        for l in range(m + 2, l_max + 1):
            fl = float(l)
            fm = float(m)
            a = np.sqrt((4.0 * fl * fl - 1.0) / (fl * fl - fm * fm))
            b = np.sqrt((((fl - 1.0) ** 2) - fm * fm) / (4.0 * ((fl - 1.0) ** 2) - 1.0))
            P[l, m] = a * (costheta * P[l - 1, m] - b * P[l - 2, m])
    return P


@njit(cache=True)
def spherical_harmonics_point(x, y, z, l_max, P, Y):
    r = np.sqrt(x * x + y * y + z * z)
    costheta = z / r
    if costheta > 1.0:
        costheta = 1.0
    elif costheta < -1.0:
        costheta = -1.0
    sintheta = np.sqrt(1.0 - costheta * costheta)

    rho = np.sqrt(x * x + y * y)
    if rho > 1.0e-14:
        cosphi = x / rho
        sinphi = y / rho
    else:
        cosphi = 1.0
        sinphi = 0.0

    normalised_legendre(costheta, sintheta, l_max, P)

    for l in range(l_max + 1):
        Y[l, l] = complex(P[l, 0], 0.0)

    emr = 1.0
    emi = 0.0
    for m in range(1, l_max + 1):
        nr = emr * cosphi - emi * sinphi
        ni = emr * sinphi + emi * cosphi
        emr = nr
        emi = ni
        sign = -1.0 if (m % 2 == 1) else 1.0
        for l in range(m, l_max + 1):
            p = P[l, m]
            yr = p * emr
            yi = p * emi
            Y[l, l + m] = complex(yr, yi)
            Y[l, l - m] = complex(sign * yr, -sign * yi)
    return Y


def spherical_harmonics(x, y, z, l_max):
    P = np.zeros((l_max + 1, l_max + 1), dtype=np.float64)
    Y = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
    spherical_harmonics_point(float(x), float(y), float(z), int(l_max), P, Y)
    return Y
