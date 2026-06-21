# HOD Models in AUM

This document describes the Halo Occupation Distribution (HOD) models
implemented in `src/hod.cpp`. All masses are in units of h^{-1} M_sun and
the input variable `xm` denotes log10(M).

---

## HOD Parameters (`hodpars` struct)

| Parameter   | Description |
|-------------|-------------|
| `Mmin`      | log10 of the characteristic minimum mass for central occupation |
| `siglogM`   | Width of the central transition in dex |
| `Msat`      | log10 of the satellite mass scale |
| `alpsat`    | Power-law slope of the satellite occupation |
| `Mcut`      | log10 of the satellite cutoff mass |
| `fac`       | Auxiliary amplitude factor (model-dependent) |
| `csbycdm`   | Ratio of satellite concentration to dark matter halo concentration |
| `hodtype`   | Integer selecting the HOD model (0--7) |
| `Acen`      | Central assembly bias strength, range [-1, 1] (hodtype 6) |
| `Asat`      | Satellite assembly bias strength, range [-1, 1] (hodtype 6) |
| `Mq`        | log10 of high-mass quenching scale (hodtype 7) |
| `sigq`      | Width of the quenching transition in dex (hodtype 7) |
| `sig_lnc`   | Scatter in ln(c) at fixed mass for concentration-split decoration (hodtype 6) |

Additional parameters set via `set_inc_params(inc_alp, inc_xM)`:

| Parameter | Description |
|-----------|-------------|
| `inc_alp` | Slope of the low-mass incompleteness ramp |
| `inc_xM`  | log10(M) above which the sample is complete |

The incompleteness factor applied to the occupation is:

```
I(xm) = 1                                   if xm >= inc_xM
       = clamp(1 + inc_alp * (xm - inc_xM), 0, 1)   if xm < inc_xM
```

---

## Model 0 -- White+2012 (default)

Central occupation:

```
<Ncen>(M) = (1/2) [1 + erf((xm - Mmin) / siglogM)] * I(xm)
```

Satellite occupation (zero below Mcut):

```
<Nsat>(M) = <Ncen>(M) * (10^(xm - Msat) - 10^(Mcut - Msat))^alpsat     for xm > Mcut
          = 0                                                              for xm <= Mcut
```

---

## Model 1 -- Zheng+2005

Central occupation:

```
<Ncen>(M) = (1/2) [1 + erf((xm - Mmin) / siglogM)]
```

Satellite occupation:

```
<Nsat>(M) = <Ncen>(M) * 10^(alpsat * (xm - Msat)) * exp(-10^(Mcut - xm))
```

Note: Satellites are modulated by Ncen and have an exponential cutoff at the
low-mass end rather than a hard threshold.

---

## Model 2 -- Tabulated

Both Ncen(M) and Nsat(M) are provided as tabulated arrays and interpolated
with cubic splines (in log10 space) via `init_Nc_spl` and `init_Ns_spl`.

```
<Ncen>(M) = min(10^{spline_Nc(xm)}, 1.0)     for nc_mmin <= xm <= nc_mmax
          = 0                                   otherwise

<Nsat>(M) = 10^{spline_Ns(xm)}                for ns_mmin <= xm <= ns_mmax
          = 0                                   otherwise
```

---

## Model 3 -- Modified White

Central occupation scaled by the `fac` parameter:

```
<Ncen>(M) = fac * (1/2) [1 + erf((xm - Mmin) / siglogM)]
```

Satellite occupation (zero below Mcut):

```
<Nsat>(M) = (10^(xm - Msat) - 10^(Mcut - Msat))^alpsat     for xm > Mcut
          = 0                                                  for xm <= Mcut
```

Note: Unlike Model 0, satellites are not modulated by Ncen.

---

## Model 4 -- Modified White variant

Central occupation with a linear ramp from `fac` to 1 between Mmin and Mcut:

```
slope = (1 - fac) / (Mcut - Mmin)
F(xm) = clamp(fac + (xm - Mmin) * slope, 0, 1)

<Ncen>(M) = F(xm) * (1/2) [1 + erf((xm - Mmin) / siglogM)]
```

Satellite occupation (pure power law, no cutoff):

```
<Nsat>(M) = (10^(xm - Msat))^alpsat
```

---

## Model 5 -- Zheng+2007 five-parameter

Central occupation (identical to Model 1):

```
<Ncen>(M) = (1/2) [1 + erf((xm - Mmin) / siglogM)]
```

Satellite occupation with a threshold mass M0 = 10^Mcut:

```
M  = 10^xm
M0 = 10^Mcut
M1 = 10^Msat

<Nsat>(M) = ((M - M0) / M1)^alpsat     for M > M0
          = 0                            for M <= M0
```

Note: This differs from Model 1 in that the satellite occupation uses a
linear mass subtraction rather than an exponential cutoff.

---

## Model 6 -- Decorated HOD (Hearin+2016)

Mean occupations follow the White+2012 form (Model 0) including
incompleteness:

```
<Ncen>(M) = (1/2) [1 + erf((xm - Mmin) / siglogM)] * I(xm)

<Nsat>(M) = <Ncen>(M) * (10^(xm - Msat) - 10^(Mcut - Msat))^alpsat     for xm > Mcut
```

Assembly bias is introduced by splitting haloes at each mass into two
equal-number populations by concentration:

```
c_high = c_200 * exp(+0.6745 * sig_lnc)
c_low  = c_200 * exp(-0.6745 * sig_lnc)
```

where 0.6745 is the upper quartile of the standard normal, so the split
is at the median concentration.

The occupation numbers are modulated by assembly bias parameters Acen and
Asat (clamped so occupations remain physical):

```
Acen_eff = Acen,   clamped so |Acen_eff| <= min(Ncen, 1 - Ncen) / Ncen

Ncen_high = <Ncen> * (1 + Acen_eff)
Ncen_low  = <Ncen> * (1 - Acen_eff)

Asat_eff = clamp(Asat, -1, 1)

Nsat_high = max(<Nsat> * (1 + Asat_eff), 0)
Nsat_low  = max(<Nsat> * (1 - Asat_eff), 0)
```

Positive Acen/Asat means more concentrated haloes host more galaxies.

The 1-halo power spectrum terms are computed separately for the high- and
low-concentration sub-populations, each with their own NFW profile u(k|M,c),
then averaged:

```
P_1h = (1/2) [P_1h(c_high, Ncen_high, Nsat_high) + P_1h(c_low, Ncen_low, Nsat_low)]
```

---

## Model 7 -- High-mass quenching

Adds a high-mass quenching factor to the White+2012 model:

```
Q(xm) = (1/2) [1 - erf((xm - Mq) / sigq)]
```

Central occupation:

```
<Ncen>(M) = (1/2) [1 + erf((xm - Mmin) / siglogM)] * I(xm) * Q(xm)
```

Satellite occupation (zero below Mcut):

```
<Nsat>(M) = <Ncen>(M) * (10^(xm - Msat) - 10^(Mcut - Msat))^alpsat     for xm > Mcut
```

Here Q(xm) suppresses occupation at the high-mass end: Q -> 1 for M << Mq
and Q -> 0 for M >> Mq.

---

## Halo Model Power Spectra

Galaxy number densities are computed by integrating over the halo mass
function:

```
n_cen = integral dn/dM * <Ncen>(M) dM
n_sat = integral dn/dM * <Nsat>(M) dM
n_gal = n_cen + n_sat

f_cen = n_cen / n_gal
f_sat = n_sat / n_gal
```

### Galaxy-galaxy power spectrum

```
P_gg(k) = 2 f_cen f_sat P_gg^{1h,cs}(k)
         + f_sat^2 P_gg^{1h,ss}(k)
         + f_cen^2 P_gg^{2h,cc}(k)
         + 2 f_cen f_sat P_gg^{2h,cs}(k)
         + f_sat^2 P_gg^{2h,ss}(k)
```

where the 1-halo central-satellite and satellite-satellite terms are:

```
P_gg^{1h,cs}(k) = integral dn/dM <Ncen><Nsat> u_s(k|M) u_cen(k|M) / (n_cen n_sat) dM
P_gg^{1h,ss}(k) = integral dn/dM <Nsat>^2 |u_s(k|M)|^2 / n_sat^2 dM
```

u_s(k|M) is the Fourier transform of the NFW satellite profile with
concentration c_s = csbycdm * c_200. u_cen(k|M) accounts for possible
mis-centering of centrals (see below).

The 2-halo terms use either the linear power spectrum P_lin(k) (without halo
exclusion) or the halo-exclusion corrected quasi-linear spectrum Q_k
(with halo exclusion enabled, the default):

```
P_gg^{2h}(k) = [integral dn/dM b(M) <N> u(k|M) dM]^2 * P_lin(k)        (no exclusion)
P_gg^{2h}(k) = sum_ij w_i w_j u_i u_j Q_k(i,j)                          (halo exclusion)
```

### Galaxy-matter cross power spectrum

```
P_gm(k) = f_cen P_gm^{1h,c}(k) + f_sat P_gm^{1h,s}(k)
         + f_cen P_gm^{2h,c}(k) + f_sat P_gm^{2h,s}(k)
```

where:

```
P_gm^{1h,c}(k) = integral dn/dM <Ncen> (M/rho) u_d(k|M) u_cen(k|M) / n_cen dM
P_gm^{1h,s}(k) = integral dn/dM <Nsat> (M/rho) u_s(k|M) u_d(k|M) / n_sat dM
```

### Galaxy bias

```
b_gal(z) = [integral dn/dM b(M) (<Ncen> + <Nsat>) dM] / n_gal
```

---

## Central Mis-centering

Set via `set_cen_offset_params(fcen_off, off_rbyrs)`.

A fraction `fcen_off` of centrals are offset from the halo center with a
Gaussian kernel of width sigma = off_rbyrs * r_s (where r_s = r_200/c_200):

```
u_cen(k|M) = (1 - fcen_off) + fcen_off * exp(-k^2 sigma^2 / 2)
```

When fcen_off = 0 (default), u_cen = 1.

---

## Observable Quantities

### Projected correlation function w_p(R_p)

```
w_p(R_p) = 2 R_p integral_0^{pi_max/R_p} xi_gg(sqrt(y^2 + 1) * R_p) dy
```

### Projected correlation function with Kaiser RSD correction

Uses the Hamilton (1992) decomposition with Legendre multipoles:

```
xi_s(r_p, mu) = [1 + (2/3)f/b + (1/5)(f/b)^2] xi_gg
              + P_2(mu) [(4/3)f/b + (4/7)(f/b)^2] (xi_gg - xi_gg_bar)
              + P_4(mu) [(8/35)(f/b)^2] (xi_gg + (5/2)xi_gg_bar - (7/2)xi_gg_barbar)
```

where f = -d ln D / d ln(1+z) is the growth rate, b is the galaxy bias, and:

```
xi_gg_bar(r)    = (3/r^3) integral_0^r xi_gg(s) s^2 ds
xi_gg_barbar(r) = (5/r^5) integral_0^r xi_gg(s) s^4 ds
```

### Excess surface density (ESD / Delta Sigma)

```
Sigma(R_p)       = 2 rho_bar R_p integral_0^infty xi_gm(sqrt(y^2 + 1) * R_p) dy

Delta Sigma(R_p) = Sigma_bar(<R_p) - Sigma(R_p)
                 = (2/R_p^2) integral_0^{R_p} Sigma(R') R' dR' - Sigma(R_p)
```

in units of h M_sun pc^{-2} (the code divides by 10^12 to convert from
(h^{-1} Mpc)^{-2}).

### Surface mass density (Sigma)

`Sigma(R_p, z)` can be computed with a finite line-of-sight integration
(pi_max > 0) or as a full projection (pi_max < 0).

### Scale-dependent bias and cross-correlation coefficient

```
b(r, z) = sqrt(xi_gg(r, z) / xi_mm(r, z))
r_cc(r, z) = xi_gm(r, z) / sqrt(xi_mm(r, z) * xi_gg(r, z))
```

---

## References

- White et al. 2012 (Model 0, default)
- Zheng et al. 2005 (Model 1)
- Zheng et al. 2007 (Model 5)
- Hearin et al. 2016 (Model 6, Decorated HOD)
- van den Bosch et al. 2013 (halo exclusion)
- Hamilton 1992 (Kaiser RSD multipole expansion)
