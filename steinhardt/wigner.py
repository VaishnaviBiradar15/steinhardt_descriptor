import math

import numpy as np
from numba import njit


@njit(cache=True)
def log_factorial(n):
    return math.lgamma(n + 1.0)


@njit(cache=True)
def wigner_3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    prefactor = 0.5 * (
        log_factorial(j1 + j2 - j3)
        + log_factorial(j1 - j2 + j3)
        + log_factorial(-j1 + j2 + j3)
        - log_factorial(j1 + j2 + j3 + 1)
    )
    prefactor += 0.5 * (
        log_factorial(j1 + m1)
        + log_factorial(j1 - m1)
        + log_factorial(j2 + m2)
        + log_factorial(j2 - m2)
        + log_factorial(j3 + m3)
        + log_factorial(j3 - m3)
    )

    k_min = 0
    if j2 - j3 - m1 > k_min:
        k_min = j2 - j3 - m1
    if j1 - j3 + m2 > k_min:
        k_min = j1 - j3 + m2

    k_max = j1 + j2 - j3
    if j1 - m1 < k_max:
        k_max = j1 - m1
    if j2 + m2 < k_max:
        k_max = j2 + m2

    total = 0.0
    for k in range(k_min, k_max + 1):
        denominator = (
            log_factorial(k)
            + log_factorial(j1 + j2 - j3 - k)
            + log_factorial(j1 - m1 - k)
            + log_factorial(j2 + m2 - k)
            + log_factorial(j3 - j2 + m1 + k)
            + log_factorial(j3 - j1 - m2 + k)
        )
        term = np.exp(prefactor - denominator)
        if k % 2 == 1:
            total -= term
        else:
            total += term

    sign = 1.0 if ((j1 - j2 - m3) % 2 == 0) else -1.0
    return sign * total


@njit(cache=True)
def wigner_3j_lll_table(l):
    size = 2 * l + 1
    table = np.zeros((size, size), dtype=np.float64)
    for i1 in range(size):
        m1 = i1 - l
        for i2 in range(size):
            m2 = i2 - l
            m3 = -m1 - m2
            if abs(m3) <= l:
                table[i1, i2] = wigner_3j(l, l, l, m1, m2, m3)
    return table


def build_wigner_tables(l_values):
    l_values = [int(l) for l in l_values]
    l_max = max(l_values) if len(l_values) > 0 else 0
    tables = np.zeros((len(l_values), 2 * l_max + 1, 2 * l_max + 1), dtype=np.float64)
    for i, l in enumerate(l_values):
        tables[i, : 2 * l + 1, : 2 * l + 1] = wigner_3j_lll_table(l)
    return tables
