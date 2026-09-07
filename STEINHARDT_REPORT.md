# Steinhardt bond-orientational order parameters

Implementation notes, test results, and an explanation of why the numbers come out the way
they do.

Package: `steinhardt/` (Python, Numba, ASE)
Test script: `tests/test_steinhardt_check_l_values.py`
Figures: `results/`
Date of this report: September 2026

---

## 1. Theory and formulas

### 1.1 The neighbour shell

Every quantity below is built from one object: the set of bond vectors from a central atom
`i` to the neighbours that fall inside a cutoff radius `r_cut`.

```
bonds(i) = { r_ij  :  |r_ij| < r_cut ,  j != i }
N_b(i)   = number of such bonds
```

The cutoff is sharp. Every neighbour inside `r_cut` counts with weight 1 and everything
outside counts 0, exactly as in the original paper [1], where `r_cut` is placed at the first
minimum of the radial distribution function. There is no smooth cutoff function and no
distance weighting. Only the direction of each bond matters; its length is discarded. That
is what makes these parameters a pure measure of the *angular* arrangement of the shell.

In the code this step is `neighbours.py`, which wraps
`ase.neighborlist.neighbor_list("ijD", ...)` and returns the bond vectors in CSR form
(one flat `(n_bonds, 3)` array plus an `indptr` index). Displacement vectors coming out of
ASE already carry the periodic image shift, so periodicity needs no further handling.

### 1.2 The bond-orientational coefficients q_lm

Each bond direction is projected onto the complex spherical harmonics and averaged over the
shell. This is Eq. (2.6) of [1] in its per-atom form:

```
                1        N_b(i)
q_lm(i)  =  ---------  *   sum    Y_lm( theta_ij , phi_ij )
              N_b(i)       j = 1
```

with `theta` and `phi` the polar and azimuthal angles of `r_ij`, and `m = -l ... +l`, so
`q_lm` is a complex vector of length `2l+1` for each degree `l`.

`q_lm` on its own is not usable as a descriptor, because rotating the crystal mixes the
`m` components among themselves through the Wigner D matrices. The two invariants below
remove that dependence.

The harmonics are evaluated in `spherical_harmonics.py` by the standard stable recursion for
the fully normalised associated Legendre functions, carrying the Condon-Shortley phase, so
the convention matches `scipy.special.sph_harm_y`. The `m < 0` half is filled from
`Y_l,-m = (-1)^m conj(Y_lm)` rather than recomputed.

### 1.3 Second-order invariant Q_l

The rotationally invariant norm of `q_lm`, Eq. (2.7) of [1]:

```
                  /      4 pi        l                  \ 1/2
Q_l(i)  =        |  ------------ *  sum   | q_lm(i) |^2  |
                  \    2 l + 1     m = -l                /
```

The prefactor `4 pi / (2l+1)` is a normalisation choice that makes `Q_0 = 1` and keeps the
values of different `l` comparable. `Q_l` is invariant under rotation because
`sum_m |q_lm|^2` is the squared length of a vector that rotations only rotate, and it is
invariant under permutation and translation because the bond set itself is.

### 1.4 Third-order invariant W_l

`Q_l` throws away all phase information, so two shells with the same `|q_lm|` magnitudes but
different relative phases are indistinguishable. The third-order invariant recovers part of
that. It couples three `q_lm` components to total angular momentum zero using the Wigner 3-j
symbol, Eq. (2.8) of [1]:

```
                                  ( l    l    l  )
W_l(i)  =        sum              (              )  q_l,m1(i) q_l,m2(i) q_l,m3(i)
             m1 + m2 + m3 = 0     ( m1   m2   m3 )
```

`W_l` is real for any set of real bond directions, because `q_l,-m = (-1)^m conj(q_lm)`
holds for a real density. The code accumulates in complex arithmetic and keeps the real
part; the imaginary part sits at the 1e-16 level.

`W_l` scales as the cube of `q_lm`, so it mixes shape information with magnitude. The
normalised form, Eq. (2.9) of [1], removes the magnitude:

```
                        W_l(i)
W_hat_l(i)  =  ----------------------------
                /    l                \ 3/2
                |   sum   |q_lm(i)|^2  |
                \  m = -l              /
```

`W_hat_l` depends only on the *direction* of the `q_lm` vector in its `(2l+1)`-dimensional
space, not on its length. That is what makes it a symmetry indicator rather than an
amplitude, and it is why section 4 can predict its value exactly for the cubic structures.

### 1.5 Wigner 3-j symbols

`wigner.py` evaluates the 3-j symbols from Racah's closed formula [2], Eq. (3.7.3):

```
( j1 j2 j3 )                  j1-j2-m3    ______________
(          ) = delta         (-1)        V  Delta * F     *   sum   (-1)^k / D(k)
( m1 m2 m3 )      m1+m2+m3,0                                   k
```

with the triangle coefficient `Delta = (j1+j2-j3)! (j1-j2+j3)! (-j1+j2+j3)! / (j1+j2+j3+1)!`,
the factor `F = (j1+m1)!(j1-m1)!(j2+m2)!(j2-m2)!(j3+m3)!(j3-m3)!`, and the alternating sum
running over all `k` for which every factorial argument in

```
D(k) = k! (j1+j2-j3-k)! (j1-m1-k)! (j2+m2-k)! (j3-j2+m1+k)! (j3-j1-m2+k)!
```

stays non-negative. The whole thing is computed in log-gamma space, so nothing overflows at
large `l` and no symbolic algebra package is needed. Because `W_l` only ever needs the
`(l l l)` symbols, the code precomputes the full `(2l+1) x (2l+1)` table once per requested
`l` (`wigner_3j_lll_table`) and the hot loop just reads from it.

### 1.6 Local and global forms

The original paper defines both a per-particle average over one atom's bonds and a global
average over every bond in the configuration. The package provides both:
`bond_order_parameters` (per atom, used everywhere in this report) and
`global_bond_order_parameters` (one number per `l` for the whole cell). For a perfect
crystal, where every atom sits in an identical environment, the two agree, and one of the
unit tests checks that.

### 1.7 References

[1] P. J. Steinhardt, D. R. Nelson, M. Ronchetti, "Bond-orientational order in liquids and
glasses", Physical Review B **28**, 784 (1983). DOI: 10.1103/PhysRevB.28.784

[2] A. R. Edmonds, *Angular Momentum in Quantum Mechanics*, Princeton University Press
(1957), Eq. (3.7.3), for the closed form of the Wigner 3-j symbol.

[3] P. R. ten Wolde, M. J. Ruiz-Montero, D. Frenkel, "Numerical calculation of the rate of
crystal nucleation in a Lennard-Jones system at moderate undercooling", Journal of Chemical
Physics **104**, 9932 (1996), for the per-particle `q_lm` convention and the practice of
placing `r_cut` at the first `g(r)` minimum.

[4] W. Lechner, C. Dellago, "Accurate determination of crystal structures based on averaged
local bond order parameters", Journal of Chemical Physics **129**, 114707 (2008), for the
averaged variant `q_bar_lm` (not implemented here; see section 6).

[5] W. Mickel, S. C. Kapfer, G. E. Schroder-Turk, K. Mecke, "Shortcomings of the bond
orientational order parameters for the analysis of disordered particulate matter", Journal of
Chemical Physics **138**, 044501 (2013), on how strongly `Q_l` depends on the neighbour
definition. Relevant to section 5.

---

## 2. What the code does, file by file

| File | Contents |
|---|---|
| `spherical_harmonics.py` | Normalised associated Legendre recursion and complex `Y_lm` for one direction. Numba kernels, no allocation inside the loop. |
| `wigner.py` | Racah-formula `wigner_3j`, the `(l l l)` lookup table, and `build_wigner_tables` which stacks one table per requested `l`. |
| `bond_order.py` | The maths: accumulate `q_lm` over a shell, normalise by `N_b`, then `Q_l`, `W_l`, `W_hat_l`. `steinhardt_batch_kernel` is `@njit(parallel=True)` and walks the whole structure in one compiled pass. |
| `neighbours.py` | ASE structure reading and the CSR neighbour list. |
| `descriptor.py` | User-facing entry points, feature naming, and the local/global split. |

The batch kernel allocates its scratch arrays (`P`, `Y`, `qlm`) inside the `prange` loop, so
each thread gets its own copy and there is no race. The first call to any kernel pays a JIT
compilation cost of a second or two; `cache=True` writes the compiled object into
`__pycache__` so later runs start immediately.

---

## 3. Test case and results

### 3.1 Setup

Four ideal crystals, generated with ASE and written as LAMMPS data files, periodic in all
three directions. Cutoffs chosen to sit in the gap after the shell of interest.

| Structure | File | Atoms | `r_cut` (A) | Neighbours found |
|---|---|---:|---:|---:|
| FCC (Al, a = 4.05 A) | `fcc_perfect_crystal.lmp` | 108 | 3.2 | 12 |
| BCC (Fe, a = 2.87 A) | `bcc_perfect_crystal.lmp` | 128 | 3.0 | 14 |
| HCP (Mg, ideal c/a) | `hcp_perfect_crystal.lmp` | 96 | 3.5 | 12 |
| Simple cubic (a = 3.00 A) | `sc_perfect_crystal.lmp` | 64 | 3.3 | 6 |

The BCC cutoff of 3.0 A pulls in both the 8 first neighbours at 2.486 A and the 6 second
neighbours at 2.87 A, so the BCC row is a 14-neighbour result. This matters a lot and is
picked apart in section 4.6.

Degrees computed: `l = 2, 3, 4, 5, 6`. Every atom in each cell gives the same answer to
within 1e-15, so the per-structure numbers below are the mean over all atoms.

### 3.2 Numbers

`Q_l`:

| Structure | N_b | Q_2 | Q_3 | Q_4 | Q_5 | Q_6 |
|---|---:|---:|---:|---:|---:|---:|
| FCC | 12 | 0.0000000 | 0.0000000 | 0.1909407 | 0.0000000 | 0.5745243 |
| BCC | 14 | 0.0000000 | 0.0000000 | 0.0363696 | 0.0000000 | 0.5106882 |
| HCP | 12 | 0.0000000 | 0.0760726 | 0.0972222 | 0.2515864 | 0.4847617 |
| SC | 6 | 0.0000000 | 0.0000000 | 0.7637626 | 0.0000000 | 0.3535534 |

`W_hat_l`:

| Structure | W_hat_2 | W_hat_3 | W_hat_4 | W_hat_5 | W_hat_6 |
|---|---:|---:|---:|---:|---:|
| FCC | 0.0000000 | 0.0000000 | -0.1593174 | 0.0000000 | -0.0131606 |
| BCC | 0.0000000 | 0.0000000 | +0.1593174 | 0.0000000 | +0.0131606 |
| HCP | 0.0000000 | 0.0000000 | +0.1340970 | 0.0000000 | -0.0124420 |
| SC | 0.0000000 | 0.0000000 | +0.1593174 | 0.0000000 | +0.0131606 |

The entries printed as `0.0000000` are not all the same kind of zero. Some are exact
consequences of symmetry, some are round-off at the 1e-16 level, and one whole column is
zero by an algebraic identity that has nothing to do with the structure. Section 4 separates
them.

### 3.3 Agreement with published values

Every non-zero entry matches the literature to six decimal places.

| Structure | N_b | Q_4 | Q_6 | W_hat_4 | W_hat_6 |
|---|---:|---:|---:|---:|---:|
| FCC | 12 | 0.190941 | 0.574524 | -0.159317 | -0.013161 |
| HCP | 12 | 0.097222 | 0.484762 | +0.134097 | -0.012442 |
| BCC, first shell only | 8 | 0.509175 | 0.628539 | -0.159317 | +0.013161 |
| BCC, first + second shell | 14 | 0.036370 | 0.510688 | +0.159317 | +0.013161 |
| Simple cubic | 6 | 0.763763 | 0.353553 | +0.159317 | +0.013161 |

These are the standard reference values quoted throughout the bond-order literature, and
`tests/test_steinhardt.py` asserts all of them to 1e-5.

### 3.4 Figures

**Figure 1: Q_l fingerprint across the four structures.**

![Steinhardt bond-order fingerprint](results/steinhardt_Q_fingerprint_heatmap.png)

Read this one column by column. `Q_2` is zero everywhere. `Q_3` and `Q_5` are zero for the
three cubic structures and non-zero only for HCP. `Q_4` and `Q_6` are non-zero for all four
and carry most of the discriminating power.

**Figure 2: Q_4 and Q_6 as a two-number fingerprint.**

![Crystal fingerprint using Q4 and Q6](results/crystal_fingerprint_Q4_Q6.png)

The four structures land in four distinct places on the `(Q_4, Q_6)` plane. Simple cubic is
the outlier with the largest `Q_4` and the smallest `Q_6`; FCC has the largest `Q_6`; BCC has
a `Q_4` near zero; HCP sits between FCC and BCC on `Q_6` with a small `Q_4`.

**Figure 3: normalised third-order invariants.**

![Crystal symmetry fingerprint](results/crystal_fingerprint_What4_What6.png)

Three of the four bars for `W_hat_4` have identical height, 0.1593174, and FCC differs from
BCC and SC only in sign. Same story for `W_hat_6` at 0.0131606. HCP is the only structure
with a different magnitude. That is not a coincidence and section 4.5 derives it.

---

## 4. Why the results look like this

### 4.1 Q_2 is zero for the three cubic structures, and it has to be

`q_2m` is a set of five numbers attached to the neighbour shell. Rotating the shell rotates
that five-vector through the `l = 2` representation of SO(3). If the shell is left unchanged
by every operation of the site point group, then `q_2m` must be left unchanged too, so it has
to lie in the subspace of the `l = 2` representation that transforms as the identity
representation of that group.

For FCC, BCC and simple cubic the site point group is `O_h`, and the `l = 2` representation
restricted to `O_h` decomposes as

```
l = 2   ->   E_g  +  T_2g
```

Neither piece is the identity representation `A_1g`. There is no cubic-invariant combination
of the five `l = 2` harmonics at all, so the only vector `q_2m` can be is the zero vector.
`Q_2 = 0` exactly, for any cubic crystal, at any lattice parameter, at any cutoff.

The same argument tells you which `l` *do* survive. Under `O_h`, `A_1g` first appears in
`l = 0`, then `l = 4`, then `l = 6`, and after that in 8, 10, 12 and so on. It never appears
for `l = 1, 2, 3, 5, 7, 9, 11`. Computing `l = 1` to `12` for these structures reproduces
that pattern exactly:

| Structure | `l` with non-zero Q_l, 1 to 12 |
|---|---|
| FCC | 4, 6, 8, 10, 12 |
| BCC (14 neighbours) | 4, 6, 8, 10, 12 |
| SC | 4, 6, 8, 10, 12 |
| HCP | 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |

So `Q_4` and `Q_6` are not an arbitrary choice. They are the two lowest degrees at which a
cubic environment can produce any signal at all, which is why the whole field settled on them.

The measured values sit at 1e-16, which is float64 round-off on a sum of 12 to 14 terms of
order 0.1. They are zero.

### 4.2 Q_2 is zero for HCP too, but for a different reason

The HCP site point group is `D_3h`, and there the `l = 2` representation decomposes as

```
l = 2   ->   A_1'  +  E'  +  E''
```

`A_1'` *is* the identity representation. Symmetry permits a non-zero `q_2m` for HCP. It comes
out zero anyway because of the ideal axial ratio.

The `A_1'` component is the `m = 0` one, and up to a constant

```
q_20   ~   sum_j  ( 3 cos^2(theta_j) - 1 )
```

For the ideal HCP shell the twelve neighbours split into six in the basal plane at
`cos(theta) = 0` and six out of plane at `cos(theta) = +/- sqrt(2/3)`, which is what
`c/a = sqrt(8/3)` means. Then

```
6 * (3*0 - 1)  +  6 * (3*(2/3) - 1)  =  -6 + 6  =  0
```

The computed cosines confirm it: six values of `0.000000`, three of `+0.816497`, three of
`-0.816497`, and `sum(3cos^2 - 1) = -4.9e-15`.

This is an exact cancellation at one specific axial ratio, not a symmetry protection. Move
off ideal `c/a` and `Q_2` reappears immediately:

| c/a | 1.50 | 1.55 | 1.60 | 1.632993 | 1.65 | 1.70 | 1.80 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q_2 | 0.029070 | 0.017678 | 0.006849 | 0.000000 | 0.003442 | 0.013220 | 0.031341 |

That is the left panel of Figure 4. The right panel does the same for FCC under tetragonal
strain, where `Q_2` grows at about `0.48` per unit strain and reaches 0.0244 at 5% strain.

The practical reading for grain-boundary work: `Q_2` is dead weight in a perfect crystal but
it is a sensitive and cheap detector of uniaxial distortion. Real Mg and Zn are not at ideal
`c/a`, and atoms near a boundary are strained, so `Q_2` will not be zero there. It is worth
keeping in the feature set for that reason, not for classifying perfect phases.

### 4.3 Q_1 is zero everywhere, including HCP

`q_1m` is proportional to `sum_j r_hat_j`, the vector sum of the bond directions. All four
shells have their centroid on the central atom, so that sum is zero and `Q_1 = 0`. This holds
for HCP as well, despite HCP failing the inversion test in the next section, because the
three-fold axis plus the basal mirror already force the vector sum to vanish.

### 4.4 Odd l survive for HCP but not for FCC, BCC or SC

Under inversion, `Y_lm(-r_hat) = (-1)^l Y_lm(r_hat)`. If the neighbour shell is
centrosymmetric, meaning every bond `r` is matched by a bond `-r`, then for odd `l` the two
partners contribute equal and opposite terms and the whole sum cancels. The FCC
cuboctahedron, the BCC cube-plus-octahedron and the simple-cubic octahedron are all
centrosymmetric, so all odd `l` vanish for them.

The HCP twelve-neighbour shell is an *anti*-cuboctahedron: the three neighbours above the
basal plane are rotated 60 degrees relative to the three below, instead of sitting directly
opposite them. That shell has no inversion centre, so odd `l` survive:

```
Q_3(HCP) = 0.0760726        Q_5(HCP) = 0.2515864        Q_7(HCP) = 0.3108
```

This is the cleanest FCC-versus-HCP discriminator the method has. `Q_4` and `Q_6` separate
them too (0.191/0.575 against 0.097/0.485), but `Q_3` separates them by an exact zero against
a finite number, which is far more robust once thermal noise is added. Worth remembering when
picking features for stacking faults or HCP-like layers at a boundary.

### 4.5 The W_3 and W_5 columns are zero by algebra, for every structure

This one has nothing to do with crystal symmetry. The 3-j symbol picks up a factor
`(-1)^(j1+j2+j3)` under an odd permutation of its columns. With all three `j` equal to `l`
that factor is `(-1)^(3l)`, which is `-1` for odd `l`. The product `q_l,m1 q_l,m2 q_l,m3` in
the definition of `W_l` is symmetric under exchanging the `m` labels, so the sum reproduces
itself times `(-1)^(3l)`:

```
W_l  =  (-1)^(3l) * W_l       =>       W_l = 0   for every odd l
```

Direct evaluation confirms it:

| l | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3j(l,l,l;0,0,0) | 0 | -0.239046 | 0 | +0.134097 | 0 | -0.093060 | 0 | +0.071230 |

The whole `(l l l)` table vanishes identically for odd `l`. HCP is the proof case: it has
`Q_3 = 0.076` and `Q_5 = 0.252`, so there is real `l = 3` and `l = 5` structure in the shell,
yet `W_3 = 0.0` and `W_5 = 3.1e-36`.

Consequence for the descriptor: `W_l` and `W_hat_l` at odd `l` are structurally zero columns.
They are not measuring anything and should not go into a feature vector. Only even `l` are
worth requesting for the third-order invariants.

### 4.6 Why W_hat_4 is the same number for FCC, BCC and SC, up to sign

`W_hat_l` is a ratio of a cubic form to the 3/2 power of a quadratic form, so scaling
`q_lm` by any factor leaves it unchanged. It depends only on the direction of the `q_lm`
vector. For a cubic environment section 4.1 already fixed that direction: `A_1g` appears once
in `l = 4`, so there is exactly one cubic-invariant combination, the cubic harmonic `K_4`,
and `q_4m` must be a multiple of it. `K_4` is non-zero only at `m = 0` and `m = +/- 4`, in
the fixed ratio `q_44 / q_40 = sqrt(5/14) = 0.5976143`.

The computed `q_4m` vectors:

| Structure | q_4,-4 | q_4,0 | q_4,+4 | q_44 / q_40 | W_hat_4 |
|---|---:|---:|---:|---:|---:|
| FCC | -0.0738 | -0.1234 | -0.0738 | 0.5976 | -0.1593174 |
| SC | +0.2950 | +0.4937 | +0.2950 | 0.5976 | +0.1593174 |
| BCC (14) | +0.0140 | +0.0235 | +0.0140 | 0.5976 | +0.1593174 |
| HCP | 0 | +0.0823 | 0 | n/a | +0.1340970 |

All three cubic structures have exactly the same direction, so all three give the same
`|W_hat_4|`. Feeding the pure `K_4` vector into the code directly returns `+0.1593173731`,
and feeding its negative returns `-0.1593173731`. FCC has `q_40 < 0` while SC and BCC have
`q_40 > 0`, and `W_hat` is odd under `q -> -q`, which is the entire content of the sign
difference in Figure 3.

HCP has only the `m = 0` component at `l = 4`. For a pure `m = 0` vector the sum in `W_l`
collapses to the single term `3j(4,4,4;0,0,0) * q_40^3`, and the normalisation cancels `q_40`
completely, so

```
W_hat_4(HCP)  =  3j(4,4,4;0,0,0)  =  0.13409704688030277
```

which is the measured value to every digit. The same reasoning at `l = 6` gives 0.0131606 for
the cubic direction, and HCP differs there (-0.0124420) because its `l = 6` vector has
`m = 0, +/-3, +/-6` components rather than only `m = 0`.

So `W_hat` is a symmetry-class label, not a magnitude. It tells you *which* invariant
direction the environment sits on, and it will not distinguish two structures that share one.
`W_hat_4` alone cannot separate BCC from simple cubic; `Q_4` separates them instantly
(0.036 against 0.764). Use the two together.

### 4.7 Why Q_4 is nearly zero for BCC

This is the most interesting number in the table. The 14-neighbour BCC shell is a
superposition of two sub-shells, and their `l = 4` contributions almost cancel:

| Shell | N_b | q_40 | Q_4 | W_hat_4 |
|---|---:|---:|---:|---:|
| Cube alone (BCC first shell) | 8 | -0.329111 | 0.509175 | -0.159317 |
| Octahedron alone (BCC second shell, and SC) | 6 | +0.493666 | 0.763763 | +0.159317 |
| Combined, weighted by 8/14 and 6/14 | 14 | +0.023508 | 0.036370 | +0.159317 |

`(8 x -0.329111 + 6 x +0.493666) / 14 = +0.023508`, which is the measured 14-shell value
exactly. The cube and the octahedron are the two cubic shells with opposite-sign `K_4`
amplitude, and at the 8-to-6 population ratio of BCC they nearly annihilate each other.
`Q_4` falls from 0.509 to 0.036, a factor of 14, and `W_hat_4` flips sign because the residual
`q_40` is now positive.

Two things follow. First, the small `Q_4` of BCC is a genuine physical signature and is
exactly why BCC sits so distinctively in Figure 2. Second, it is fragile: the result depends
entirely on the cutoff capturing both sub-shells with the right weights, and it will move as
soon as the lattice is strained or the temperature is finite.

### 4.8 Why simple cubic has the largest Q_4 and the smallest Q_6

Six neighbours on the coordinate axes is the most `l = 4`-like arrangement of the four, and
with only six bonds there is less angular averaging to wash the signal out. `Q_4 = 0.7638` is
the largest number in the whole table. Conversely `Q_6 = 0.3536` is the smallest, because the
octahedron is a poor match to the `l = 6` cubic harmonic.

The `Q_6` ordering across the four, 0.5745 for FCC, 0.5107 for BCC, 0.4848 for HCP and 0.3536
for SC, is the reason `Q_6` became the standard crystalline-versus-liquid order parameter.
All the densely packed structures give something around 0.5, while a disordered shell averages
`q_6m` toward zero.

---

## 5. Cutoff sensitivity

The cutoff is the one genuinely free parameter and it is not a minor one. Scanning `r_cut`
for FCC with `a = 4.05 A`:

| r_cut (A) | N_b | Q_4 | Q_6 | W_hat_4 |
|---:|---:|---:|---:|---:|
| 2.90 | 12 | 0.190941 | 0.574524 | -0.159317 |
| 3.20 | 12 | 0.190941 | 0.574524 | -0.159317 |
| 3.50 | 12 | 0.190941 | 0.574524 | -0.159317 |
| 4.06 | 18 | 0.127294 | 0.265165 | +0.159317 |
| 5.00 | 42 | 0.054554 | 0.009821 | -0.159317 |
| 5.80 | 54 | 0.084863 | 0.135311 | -0.159317 |

Inside the gap between the first and second shell, 2.86 A to 4.05 A, the answer is completely
flat. Cross into the second shell and `Q_6` collapses from 0.575 to 0.265 and then to 0.010.
The second FCC shell sits at exactly `a = 4.05 A`, so a cutoff of 4.05 or 4.06 is on a knife
edge, and thermal displacements in a real structure will flip some atoms across it.

`W_hat_4` stays at `+/- 0.159317` throughout, because every one of those shells is still
cubic. Only the sign moves. That is section 4.6 again: `W_hat` reports symmetry class and is
almost blind to the cutoff, while `Q_l` is very sensitive to it.

For grain-boundary work this is the parameter to be careful about. Mickel et al. [5] make the
same point at length: a fixed radial cutoff is unstable exactly where the structure is
disordered, which is precisely at the boundary. Two options if it becomes a problem: place
`r_cut` from the actual `g(r)` minimum of the relaxed structure rather than the ideal lattice,
or switch the neighbour definition to a Voronoi or solid-angle-weighted one, which is what [5]
recommends.

---

## 6. Code review

Everything numerical is correct. All twelve unit tests in `tests/test_steinhardt.py` pass,
covering the five reference structures, the vanishing of odd `l` in centrosymmetric crystals,
translation, rotation and permutation invariance, agreement between the local and global
paths, and the 3-j symbols against a closed form for `l = 0` to `10`.

The problems below are about paths, robustness and packaging, not about the physics.

### 6.1 Hard-coded Windows path in descriptor.py

`steinhardt/descriptor.py` opens with

```python
import sys
sys.path.insert(0, r"C:\Users\vaish\Compute_descriptors\steinhardt")
```

That directory does not exist any more. `C:\Users\vaish\Compute_descriptors` now contains only
`ace`, `smooth_radial` and `structures`; the `steinhardt` folder was moved out to
`C:\Users\vaish\steinhardt_descriptor`. So the line currently inserts a dead path at the front
of `sys.path`.

Two reasons to delete it. It makes the package unusable on any other machine and unusable as
an installed package, and if that folder is ever recreated the line will put a second copy of
the `steinhardt` package ahead of the real one on the import path, so edits made in
`steinhardt_descriptor` would silently not be the ones running. A library module should never
modify `sys.path`.

Fix: delete lines 1 to 6 of `descriptor.py`. The relative imports below them already do the
right thing.

### 6.2 Both scripts point at a structures folder that is not there

`tests/test_steinhardt_check_l_values.py` sets

```python
STRUCTURE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "structures"))
```

which resolves to `C:\Users\vaish\structures`. That folder does not exist. The `.lmp` files
are in `C:\Users\vaish\Compute_descriptors\structures`. Running the script today prints
`Missing:` four times, then dies inside Figure 2 with

```
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

because `Q_values` is an empty list by the time it gets indexed. The figures in `results/`
were made before the folders moved.

`examples/run_steinhardt.py` has the same problem one level worse: its
`os.path.join(dirname, "..", "..", "structures")` resolves to `C:\Users\structures`.

Two fixes, in order of preference. Either copy the six `.lmp` files into
`steinhardt_descriptor/structures/` so the repository carries its own test data and both
scripts become `os.path.join(PROJECT_DIR, "structures")`, which is what a standalone repo
should do. Or point them at the real location:

```python
STRUCTURE_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "..", "Compute_descriptors", "structures"))
```

Either way, add a clear failure instead of a silent one:

```python
if not labels:
    raise SystemExit("No structures found in %s" % STRUCTURE_DIR)
```

### 6.3 The W_hat guard is an absolute threshold and will eventually fail

In `bond_order.py`:

```python
if norm2 > 1.0e-30:
    w_hat_out[k] = w / (norm2 ** 1.5)
else:
    w_hat_out[k] = 0.0
```

When `Q_l` is zero by symmetry, `W_hat_l` is genuinely `0/0` and returning 0 is a reasonable
convention. The problem is the threshold. For a perfect crystal `norm2` at `l = 2` is pure
round-off, and it grows with system size because the code takes the worst atom out of many:

| Atoms | max Q_2 | max norm2 | Margin below the 1e-30 guard |
|---:|---:|---:|---:|
| 32 | 1.33e-16 | 7.00e-33 | 143x |
| 108 | 2.45e-16 | 2.39e-32 | 42x |
| 256 | 2.24e-16 | 1.99e-32 | 50x |
| 500 | 4.36e-16 | 7.55e-32 | 13x |

The margin is down to a factor of 13 at 500 atoms. A grain-boundary cell of 10^5 atoms will
cross it, and when it does the affected atoms jump from a clean `W_hat_2 = 0` to a random
value of order 0.2.

That is worse than it sounds, because `W_hat` is scale-free. Rattling the FCC cell by
different amounts shows the point:

| Rattle amplitude | Q_2 | W_hat_2 range across atoms |
|---|---:|---|
| 0 (perfect) | 1.3e-16 | 0 to 0 (guard fires) |
| 1e-9 A | 2.3e-10 | -0.2378 to +0.2353 |
| 1e-6 A | 2.3e-07 | -0.2378 to +0.2353 |
| 1e-4 A | 2.3e-05 | -0.2378 to +0.2354 |

The `W_hat_2` spread is identical at every amplitude, from a physically meaningless 1e-9 A
displacement up to a real 1e-4 A one. `W_hat_l` does not report how much distortion there is;
it reports the direction of a vector that may be entirely round-off. Fed into a model, such a
column looks like a unit-scale feature and is pure noise.

Fix: guard on `Q_l`, which is a physical quantity with a meaningful scale, rather than on the
raw squared norm:

```python
q = np.sqrt(FOUR_PI / (2.0 * l + 1.0) * norm2)
q_out[k] = q
w = order_parameter_w(qlm, l, tables[k])
w_out[k] = w
if q > 1.0e-8:
    w_hat_out[k] = w / (norm2 ** 1.5)
else:
    w_hat_out[k] = 0.0
```

`1e-8` sits about eight orders of magnitude above float64 round-off and about three below any
`Q_l` that means something. Returning `np.nan` instead of `0.0` would be even safer, since it
forces the caller to notice, but zero keeps the array usable downstream.

### 6.4 test_steinhardt_check_l_values.py is a plotting script, not a test

It lives in `tests/`, its name starts with `test_`, so pytest collects it, but it contains no
`test_*` function and pytest reports zero tests from it. It also calls `plt.show()` three
times, which blocks on a machine with a GUI backend, and it writes files into `results/` as a
side effect of collection.

Move it to `examples/plot_l_values.py` or `scripts/`, and if the numbers in it are worth
locking down, add a real assertion test next to the existing ones:

```python
def test_odd_l_nonzero_for_hcp():
    atoms, cutoff = hcp()
    result = steinhardt_from_atoms(atoms, cutoff=cutoff, l_values=(3, 5))
    assert result["Q"][:, 0] == pytest.approx(0.076073, abs=1e-5)
    assert result["Q"][:, 1] == pytest.approx(0.251586, abs=1e-5)
```

### 6.5 Smaller points

The `l = 0` case returns `Q_0 = 1.0` and `W_hat_0 = 1.0` for every atom of every structure.
Both are correct and both are constant, so `l = 0` is a wasted column if it ever ends up in a
feature vector.

Odd `l` should be excluded from `include_w` and `include_w_hat` requests, per section 4.5.
The current code happily returns the zero columns. A one-line warning in
`bond_order_parameters` would save confusion later.

`order_parameter_q` is defined and exported but the batch path recomputes the same expression
inline inside `bond_order_from_qlm`. Harmless duplication; worth collapsing so there is only
one place to change the normalisation convention.

The BCC row of Figure 1 is the 14-neighbour result, but the axis label just says `BCC`.
Anyone comparing it against the more commonly quoted 8-neighbour numbers (0.509 and 0.629)
will think the code is wrong. Label it `BCC (14 nn)`.

The `Usage` block at the end of `README.md` refers to `structures/fcc_perfect_crystal.lmp`,
which does not exist relative to the repository. Same fix as 6.2. The README's fenced code
block is also missing its closing backticks.

---

## 7. Summary of what to change

1. Delete the `sys.path.insert` block at the top of `steinhardt/descriptor.py`.
2. Copy the six `.lmp` files into `steinhardt_descriptor/structures/` and repoint
   `STRUCTURE_DIR` in both `tests/test_steinhardt_check_l_values.py` and
   `examples/run_steinhardt.py`; raise a clear error when nothing is found.
3. Change the `W_hat` guard from `norm2 > 1e-30` to a test on `Q_l > 1e-8`.
4. Move `test_steinhardt_check_l_values.py` out of `tests/` and drop the `plt.show()` calls,
   or convert it into a real pytest test.
5. Restrict `W_l` and `W_hat_l` to even `l`.
6. Relabel the BCC row in Figure 1 and close the code fence in `README.md`.

None of these affect the numbers in section 3, which are correct as they stand.
