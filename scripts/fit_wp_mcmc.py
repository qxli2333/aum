#!/usr/bin/env python
"""
MCMC fit of HOD parameters to galaxy projected correlation function wp(rp)
with Kaiser RSD correction, and optional ESD (Delta Sigma) joint fitting.

Uses the aum halo model to compute wp_Kaiser(rp) and emcee for sampling.
Fits the five standard HOD parameters (Mmin, siglogM, Msat, alpsat, Mcut)
with an optional galaxy number density constraint.

Input data: pycorr rppi-mode .npy files (default) or plain-text files.

Usage:
    python fit_wp_mcmc.py --data wp_rppi.npy --z 0.3
    python fit_wp_mcmc.py --data wp_rppi.npy --z 0.3 --esd Dsigma.csv
    python fit_wp_mcmc.py --data wp.txt --cov cov.txt --z 0.5 --format text
    python fit_wp_mcmc.py --config config.yaml
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import emcee

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import hod as h


# ---------------------------------------------------------------------------
# Configuration defaults (override via CLI or config file)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    # Cosmology
    Om0=0.307115, Omk=0.0, w0=-1.0, wa=0.0,
    hval=0.6777, Omb=0.048206, th=2.726,
    s8=0.8228, nspec=0.96, ximax=0.90309, cfac=1.0,

    # HOD model
    hodtype=0,
    csbycdm=1.0,
    fac=1.0,
    Acen=0.0, Asat=0.0, Mq=16.0, sigq=0.5, sig_lnc=0.0,

    # Wp computation
    z_eff=0.5,
    pimax=40.0,

    # Number density constraint (set ng_obs=0 to disable)
    ng_obs=0.0,
    ng_err=0.0,

    # Sampler
    nwalkers=32,
    nsteps=2000,
    nburn=500,
    thin=1,

    # Priors [min, max] for each fitted parameter
    prior_Mmin=[11.0, 15.0],
    prior_siglogM=[0.01, 2.0],
    prior_Msat=[12.0, 16.0],
    prior_alpsat=[0.5, 2.0],
    prior_Mcut=[10.0, 15.5],
    prior_Mq=[13.0, 17.0],
    prior_sigq=[0.01, 2.0],
    prior_fac=[0.01, 10.0],

    # Covariance correction
    hartlap_nmocks=0,

    # Output
    output="chains/wp_hod_chain.h5",
)


# ---------------------------------------------------------------------------
# Global hod object (initialized once, reused via hod_renew)
# ---------------------------------------------------------------------------
HOD_OBJ = None
COSMO_PARS = None
HOD_FIXED = None


def init_hod(cfg):
    """Build the global hod object with the fiducial cosmology."""
    global HOD_OBJ, COSMO_PARS, HOD_FIXED

    p = h.cosmo()
    p.Om0 = cfg["Om0"]; p.Omk = cfg["Omk"]
    p.w0 = cfg["w0"]; p.wa = cfg["wa"]
    p.hval = cfg["hval"]; p.Omb = cfg["Omb"]
    p.th = cfg["th"]; p.s8 = cfg["s8"]
    p.nspec = cfg["nspec"]; p.ximax = cfg["ximax"]
    p.cfac = cfg["cfac"]
    COSMO_PARS = p

    q = h.hodpars()
    q.Mmin = 13.0; q.siglogM = 0.5; q.Msat = 14.0
    q.alpsat = 1.0; q.Mcut = 13.5
    q.csbycdm = cfg["csbycdm"]; q.fac = cfg["fac"]
    q.hodtype = cfg["hodtype"]
    q.Acen = cfg["Acen"]; q.Asat = cfg["Asat"]
    q.Mq = cfg["Mq"]; q.sigq = cfg["sigq"]
    q.sig_lnc = cfg["sig_lnc"]
    HOD_FIXED = cfg

    HOD_OBJ = h.hod(p, q)


def _set_hod_params(theta):
    """Update HOD object with new parameters."""
    params = dict(zip(PARAM_NAMES, theta))
    q = h.hodpars()
    q.Mmin = params["Mmin"]
    q.siglogM = params["siglogM"]
    q.Msat = params["Msat"]
    q.alpsat = params["alpsat"]
    q.Mcut = params["Mcut"]
    q.csbycdm = HOD_FIXED["csbycdm"]
    q.fac = params.get("fac", HOD_FIXED["fac"])
    q.hodtype = HOD_FIXED["hodtype"]
    q.Acen = params.get("Acen", HOD_FIXED["Acen"])
    q.Asat = params.get("Asat", HOD_FIXED["Asat"])
    q.Mq = params.get("Mq", HOD_FIXED["Mq"])
    q.sigq = params.get("sigq", HOD_FIXED["sigq"])
    q.sig_lnc = params.get("sig_lnc", HOD_FIXED["sig_lnc"])
    HOD_OBJ.hod_renew(q)


def compute_wp(theta, rp_bins, z_eff, pimax):
    """
    Compute model wp_Kaiser(rp) for a given set of HOD parameters.

    Returns
    -------
    wp_model : ndarray
    ng_model : float
    """
    _set_hod_params(theta)

    nbins = len(rp_bins)
    rp_arr = h.doubleArray(nbins)
    wp_arr = h.doubleArray(nbins)
    for i in range(nbins):
        rp_arr[i] = rp_bins[i]

    HOD_OBJ.Wp_Kaiser(z_eff, nbins, rp_arr, wp_arr, pimax)

    wp_model = np.array([wp_arr[i] for i in range(nbins)])
    ng_model = HOD_OBJ.ncenz(z_eff) + HOD_OBJ.nsatz(z_eff)

    return wp_model, ng_model


def compute_esd(theta, rp_bins, z_eff):
    """
    Compute model ESD (Delta Sigma) for a given set of HOD parameters.

    Units: h M_sun pc^{-2} (aum divides by 1e12 internally).

    Returns
    -------
    esd_model : ndarray
    """
    _set_hod_params(theta)

    nbins = len(rp_bins)
    rp_arr = h.doubleArray(nbins)
    esd_arr = h.doubleArray(nbins)
    for i in range(nbins):
        rp_arr[i] = rp_bins[i]

    esdbins2 = nbins + 6
    HOD_OBJ.ESD(z_eff, nbins, rp_arr, esd_arr, esdbins2, True)

    return np.array([esd_arr[i] for i in range(nbins)])


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
PARAM_NAMES = None
PRIORS = None

BASE_PARAMS = ["Mmin", "siglogM", "Msat", "alpsat", "Mcut"]
EXTRA_PARAMS_BY_HODTYPE = {
    6: ["Acen", "Asat"],
    7: ["Mq", "sigq"],
    8: ["fac", "Mq", "sigq"],  # mHMQ: fac=Ac, Mq=gamma, sigq=As
}
FIDUCIAL = {
    "Mmin": 13.0, "siglogM": 0.5, "Msat": 14.0, "alpsat": 1.0, "Mcut": 13.5,
    "Acen": 0.0, "Asat": 0.0, "Mq": 15.0, "sigq": 0.5, "fac": 1.0,
}
FIDUCIAL_BY_HODTYPE = {
    8: {"Mmin": 12.5, "siglogM": 0.4, "Msat": 13.5, "alpsat": 1.0,
        "Mcut": 12.0, "fac": 1.0, "Mq": -1.0, "sigq": 1.0},
}
SCATTER = {
    "Mmin": 0.1, "siglogM": 0.05, "Msat": 0.1, "alpsat": 0.05, "Mcut": 0.1,
    "Acen": 0.05, "Asat": 0.05, "Mq": 0.2, "sigq": 0.1, "fac": 0.1,
}


def log_prior(theta):
    for val, (lo, hi) in zip(theta, PRIORS):
        if val < lo or val > hi:
            return -np.inf
    return 0.0


def log_likelihood(theta, rp_bins, wp_data, Cinv_wp, z_eff, pimax,
                   ng_obs, ng_err, esd_rp, esd_data, Cinv_esd):
    wp_model, ng_model = compute_wp(theta, rp_bins, z_eff, pimax)

    if not np.all(np.isfinite(wp_model)) or np.any(wp_model <= 0):
        return -np.inf

    residual_wp = wp_data - wp_model
    chi2 = residual_wp @ Cinv_wp @ residual_wp

    if ng_obs > 0 and ng_err > 0:
        chi2 += ((ng_model - ng_obs) / ng_err) ** 2

    if esd_rp is not None:
        esd_model = compute_esd(theta, esd_rp, z_eff)
        if not np.all(np.isfinite(esd_model)):
            return -np.inf
        residual_esd = esd_data - esd_model
        chi2 += residual_esd @ Cinv_esd @ residual_esd

    return -0.5 * chi2


def log_probability(theta, rp_bins, wp_data, Cinv_wp, z_eff, pimax,
                    ng_obs, ng_err, esd_rp, esd_data, Cinv_esd):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, rp_bins, wp_data, Cinv_wp, z_eff, pimax,
                        ng_obs, ng_err, esd_rp, esd_data, Cinv_esd)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_pycorr_wp(path, pimax):
    """
    Load wp(rp) and jackknife covariance from a pycorr rppi-mode .npy file.

    Parameters
    ----------
    path : str
        Path to pycorr .npy file (must be rppi mode).
    pimax : float
        Line-of-sight integration limit in h^-1 Mpc.
        Validated against the file's pi edges.

    Returns
    -------
    rp, wp, cov : ndarrays
    pimax_used : float
        The actual pimax used (clamped to file's pi range if needed).
    """
    from pycorr import TwoPointCorrelationFunction, project_to_wp

    result = TwoPointCorrelationFunction.load(path)
    if result.mode != "rppi":
        raise ValueError(
            f"Expected rppi-mode pycorr file, got mode='{result.mode}'. "
            f"Use rppi files (e.g., from the wp/ directory)."
        )

    pi_max_file = result.edges[1][-1]
    if pimax > pi_max_file:
        print(f"WARNING: requested pimax={pimax} exceeds file pi range "
              f"[{result.edges[1][0]}, {pi_max_file}]. "
              f"Clamping to {pi_max_file}.")
        pimax = pi_max_file

    rp, wp, cov = project_to_wp(result, pimax=pimax,
                                return_sep=True, return_cov=True)

    mask = ~np.isnan(wp) & ~np.isnan(np.diag(cov))
    return rp[mask], wp[mask], cov[np.ix_(mask, mask)], pimax


def load_esd_data(path, rp_range=None):
    """
    Load ESD (Delta Sigma) data from a CSV file.

    Expected columns: rp, ds, ds_err.

    Returns
    -------
    rp, esd, esd_err : ndarrays
    """
    df = pd.read_csv(path)
    rp = df["rp"].values
    esd = df["ds"].values
    esd_err = df["ds_err"].values

    if rp_range is not None:
        rp_min, rp_max = rp_range
        sel = (rp >= rp_min) & (rp <= rp_max)
        rp, esd, esd_err = rp[sel], esd[sel], esd_err[sel]

    return rp, esd, esd_err


def load_wp_text(path):
    """Load wp from plain text file (legacy format)."""
    data = np.loadtxt(path)
    rp = data[:, 0]
    wp = data[:, 1]
    wp_err = data[:, 2] if data.shape[1] > 2 else None
    return rp, wp, wp_err


def load_covariance_text(path, nbins):
    """Load covariance matrix from plain text file (legacy format)."""
    data = np.loadtxt(path)
    if data.ndim == 2 and data.shape == (nbins, nbins):
        return data
    if data.ndim == 2 and data.shape[1] >= 3:
        cov = np.zeros((nbins, nbins))
        for row in data:
            i, j = int(row[0]), int(row[1])
            cov[i, j] = row[2]
            cov[j, i] = row[2]
        return cov
    raise ValueError(f"Cannot parse covariance from {path} for {nbins} bins")


def load_config(path):
    """Load YAML config file, falling back to defaults."""
    import yaml
    with open(path) as f:
        user_cfg = yaml.safe_load(f)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(user_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global PRIORS

    parser = argparse.ArgumentParser(
        description="MCMC HOD fit to wp_Kaiser(rp) with optional ESD")
    parser.add_argument("--data", type=str,
                        help="wp data file (pycorr .npy or text)")
    parser.add_argument("--format", type=str, default="pycorr",
                        choices=["pycorr", "text"],
                        help="Input format: pycorr (default) or text")
    parser.add_argument("--cov", type=str, default=None,
                        help="Covariance file (text format only)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")
    parser.add_argument("--z", type=float, default=None,
                        help="Effective redshift")
    parser.add_argument("--pimax", type=float, default=None,
                        help="pimax in h^-1 Mpc")
    parser.add_argument("--rp-range", type=float, nargs=2, default=None,
                        metavar=("RP_MIN", "RP_MAX"),
                        help="rp range for wp fit (h^-1 Mpc)")
    parser.add_argument("--esd", type=str, default=None,
                        help="ESD CSV file (enables joint wp+ESD fitting)")
    parser.add_argument("--esd-rp-range", type=float, nargs=2, default=None,
                        metavar=("RP_MIN", "RP_MAX"),
                        help="rp range for ESD fit (h^-1 Mpc)")
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--nsteps", type=int, default=None)
    parser.add_argument("--nburn", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output chain file (.h5 or .npy)")
    parser.add_argument("--ng", type=float, nargs=2, default=None,
                        metavar=("NG_OBS", "NG_ERR"),
                        help="Galaxy number density constraint")
    parser.add_argument("--hodtype", type=int, default=None,
                        help="HOD model type")
    parser.add_argument("--hartlap-nmocks", type=int, default=None,
                        help="Number of mocks for Hartlap correction")
    args = parser.parse_args()

    # Build config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = dict(DEFAULT_CONFIG)

    # CLI overrides
    if args.z is not None: cfg["z_eff"] = args.z
    if args.pimax is not None: cfg["pimax"] = args.pimax
    if args.nwalkers is not None: cfg["nwalkers"] = args.nwalkers
    if args.nsteps is not None: cfg["nsteps"] = args.nsteps
    if args.nburn is not None: cfg["nburn"] = args.nburn
    if args.output is not None: cfg["output"] = args.output
    if args.hodtype is not None: cfg["hodtype"] = args.hodtype
    if args.hartlap_nmocks is not None: cfg["hartlap_nmocks"] = args.hartlap_nmocks
    if args.ng is not None:
        cfg["ng_obs"] = args.ng[0]
        cfg["ng_err"] = args.ng[1]

    # ---- Load wp data ----
    if args.data is None and "data" not in cfg:
        parser.error("Provide --data or set 'data' in config file")
    data_path = args.data or cfg["data"]

    if args.format == "pycorr":
        rp_bins, wp_data, cov, pimax_used = load_pycorr_wp(
            data_path, cfg["pimax"])
        cfg["pimax"] = pimax_used
        print(f"Loaded {len(rp_bins)} wp bins from pycorr file {data_path}")
        print(f"  pimax = {pimax_used:.1f} h^-1 Mpc")
    else:
        rp_bins, wp_data, wp_err = load_wp_text(data_path)
        pimax_used = cfg["pimax"]
        cov_path = args.cov or cfg.get("cov", None)
        if cov_path is not None:
            cov = load_covariance_text(cov_path, len(rp_bins))
            print(f"Loaded covariance from {cov_path}")
        elif wp_err is not None:
            cov = np.diag(wp_err**2)
            print("Using diagonal covariance from wp_err column")
        else:
            parser.error("Provide --cov or include wp_err column in data")
        print(f"Loaded {len(rp_bins)} wp bins from {data_path}")

    # Apply rp range cut for wp
    if args.rp_range is not None:
        rp_min, rp_max = args.rp_range
        sel = (rp_bins >= rp_min) & (rp_bins <= rp_max)
        rp_bins = rp_bins[sel]
        wp_data = wp_data[sel]
        cov = cov[np.ix_(sel, sel)]
        print(f"  rp range cut: [{rp_min:.3f}, {rp_max:.3f}] -> {len(rp_bins)} bins")

    nbins_wp = len(rp_bins)
    print(f"  rp range: [{rp_bins[0]:.4f}, {rp_bins[-1]:.4f}] h^-1 Mpc")

    # Inverse covariance with optional Hartlap correction
    Cinv_wp = np.linalg.inv(cov)
    nmocks = cfg["hartlap_nmocks"]
    if nmocks > 0:
        hartlap = (nmocks - nbins_wp - 2.0) / (nmocks - 1.0)
        if hartlap <= 0:
            print(f"WARNING: Hartlap factor non-positive ({hartlap:.3f})")
            hartlap = 1.0
        Cinv_wp *= hartlap
        print(f"Applied Hartlap correction: factor = {hartlap:.4f} (nmocks={nmocks})")

    # ---- Load ESD data (optional) ----
    esd_rp = esd_data = Cinv_esd = None
    if args.esd is not None:
        esd_rp, esd_data, esd_err = load_esd_data(
            args.esd, rp_range=args.esd_rp_range)
        Cinv_esd = np.diag(1.0 / esd_err**2)
        print(f"Loaded {len(esd_rp)} ESD bins from {args.esd}")
        print(f"  ESD rp range: [{esd_rp[0]:.4f}, {esd_rp[-1]:.4f}] h^-1 Mpc")
        print(f"  ESD units: h M_sun pc^-2 (must match aum ESD output)")

    # ---- Set parameter list based on hodtype ----
    hodtype = cfg["hodtype"]
    PARAM_NAMES = list(BASE_PARAMS)
    if hodtype in EXTRA_PARAMS_BY_HODTYPE:
        PARAM_NAMES += EXTRA_PARAMS_BY_HODTYPE[hodtype]
    globals()["PARAM_NAMES"] = PARAM_NAMES

    # hodtype=8 uses Mq as gamma (skewness), which can be negative
    if hodtype == 8 and cfg["prior_Mq"] == DEFAULT_CONFIG["prior_Mq"]:
        cfg["prior_Mq"] = [-3.0, 3.0]

    PRIORS = [cfg[f"prior_{p}"] for p in PARAM_NAMES]
    globals()["PRIORS"] = PRIORS

    print(f"Fitted parameters ({len(PARAM_NAMES)}): {PARAM_NAMES}")

    # ---- Initialize HOD ----
    print(f"\nInitializing HOD model (hodtype={hodtype})...")
    init_hod(cfg)

    pimax = cfg["pimax"]

    # Validate: compute wp_Kaiser at fiducial to check timing
    fid = FIDUCIAL_BY_HODTYPE.get(hodtype, FIDUCIAL)
    theta0 = [fid.get(p, FIDUCIAL[p]) for p in PARAM_NAMES]
    t0 = time.time()
    wp_test, ng_test = compute_wp(theta0, rp_bins, cfg["z_eff"], pimax)
    t_wp = time.time() - t0

    t_esd = 0.0
    if esd_rp is not None:
        t0 = time.time()
        esd_test = compute_esd(theta0, esd_rp, cfg["z_eff"])
        t_esd = time.time() - t0
        print(f"First wp_Kaiser eval: {t_wp:.2f}s, ESD eval: {t_esd:.2f}s")
    else:
        print(f"First wp_Kaiser eval: {t_wp:.2f}s")

    theta1 = [FIDUCIAL[p] + SCATTER[p] for p in PARAM_NAMES]
    t0 = time.time()
    wp_test2, _ = compute_wp(theta1, rp_bins, cfg["z_eff"], pimax)
    t_eval = time.time() - t0
    if esd_rp is not None:
        t0b = time.time()
        compute_esd(theta1, esd_rp, cfg["z_eff"])
        t_eval += time.time() - t0b

    print(f"Subsequent eval (wp+esd): {t_eval:.3f}s")
    print(f"Fiducial ng = {ng_test:.4e}")

    # Estimate total runtime
    nwalkers = cfg["nwalkers"]
    nsteps = cfg["nsteps"]
    nburn = cfg["nburn"]
    total_evals = nwalkers * (nsteps + nburn)
    est_hours = total_evals * t_eval / 3600.0
    print(f"Estimated runtime: {est_hours:.1f} hours "
          f"({nwalkers} walkers x {nsteps+nburn} steps x {t_eval:.3f}s)")

    # Initialize walkers near fiducial
    ndim = len(PARAM_NAMES)
    p0 = np.array(theta0)
    scatter_arr = np.array([SCATTER[p] for p in PARAM_NAMES])
    pos = p0 + scatter_arr * np.random.randn(nwalkers, ndim)
    for i in range(nwalkers):
        for j in range(ndim):
            pos[i, j] = np.clip(pos[i, j], PRIORS[j][0], PRIORS[j][1])

    # Set up sampler
    sampler_args = (rp_bins, wp_data, Cinv_wp, cfg["z_eff"], pimax,
                    cfg["ng_obs"], cfg["ng_err"],
                    esd_rp, esd_data, Cinv_esd)

    output_path = cfg["output"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    use_h5 = output_path.endswith(".h5")
    backend = None
    if use_h5:
        backend = emcee.backends.HDFBackend(output_path)
        backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=sampler_args, backend=backend
    )

    # Run burn-in
    print(f"\nRunning burn-in ({nburn} steps)...")
    t0 = time.time()
    state = sampler.run_mcmc(pos, nburn, progress=True)
    t_burn = time.time() - t0
    print(f"Burn-in done in {t_burn/60:.1f} min")

    # Reset and run production
    sampler.reset()
    if use_h5:
        backend.reset(nwalkers, ndim)

    print(f"Running production ({nsteps} steps)...")
    t0 = time.time()
    sampler.run_mcmc(state, nsteps, progress=True)
    t_prod = time.time() - t0
    print(f"Production done in {t_prod/60:.1f} min")

    # Save results
    if not use_h5:
        chain = sampler.get_chain(thin=cfg["thin"], flat=True)
        log_prob = sampler.get_log_prob(thin=cfg["thin"], flat=True)
        save_dict = dict(
            chain=chain,
            log_prob=log_prob,
            param_names=PARAM_NAMES,
            rp=rp_bins,
            wp_data=wp_data,
            cov_wp=cov,
            config={k: v for k, v in cfg.items() if not isinstance(v, list)},
        )
        if esd_rp is not None:
            save_dict["esd_rp"] = esd_rp
            save_dict["esd_data"] = esd_data
            save_dict["esd_err"] = np.sqrt(np.diag(np.linalg.inv(Cinv_esd)))
        np.savez(output_path, **save_dict)

    # Summary
    chain = sampler.get_chain(flat=True)
    print(f"\nChain shape: {chain.shape}")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Autocorrelation times: {tau}")
    except Exception:
        print("Autocorrelation time estimation failed (chain may be too short)")

    print("\nBest-fit parameters (median +/- 1sigma):")
    percentiles = np.percentile(chain, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        med = percentiles[1, i]
        lo = med - percentiles[0, i]
        hi = percentiles[2, i] - med
        print(f"  {name:>10s} = {med:.4f}  +{hi:.4f}  -{lo:.4f}")

    # Compute best-fit wp
    best_theta = percentiles[1]
    wp_best, ng_best = compute_wp(best_theta, rp_bins, cfg["z_eff"], pimax)
    residual = wp_data - wp_best
    chi2_wp = residual @ Cinv_wp @ residual
    ndof = nbins_wp - ndim
    print(f"\nBest-fit wp chi2/dof = {chi2_wp:.2f}/{ndof} = {chi2_wp/ndof:.2f}")
    print(f"Best-fit ng = {ng_best:.4e}")

    if esd_rp is not None:
        esd_best = compute_esd(best_theta, esd_rp, cfg["z_eff"])
        residual_esd = esd_data - esd_best
        chi2_esd = residual_esd @ Cinv_esd @ residual_esd
        ndof_esd = len(esd_rp)
        print(f"Best-fit ESD chi2/dof = {chi2_esd:.2f}/{ndof_esd} = {chi2_esd/ndof_esd:.2f}")

    print(f"\nChain saved to {output_path}")


if __name__ == "__main__":
    main()
