#!/usr/bin/env python
"""Plot data vs best-fit model with 1-sigma band for wp and ESD."""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_utils import (load_chain, init_hod_from_cfg, set_params,
                        eval_wp, eval_esd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chain", help="Chain .npz file")
    parser.add_argument("--nsample", type=int, default=200,
                        help="Number of chain samples for error band")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    data = load_chain(args.chain)
    chain = data["chain"]
    log_prob = data["log_prob"]
    param_names = data["param_names"]
    cfg = data["cfg"]
    rp = data["rp"]
    wp_data = data["wp_data"]
    cov_wp = data["cov_wp"]
    wp_err = np.sqrt(np.diag(cov_wp))

    z = cfg["z_eff"]
    pimax = cfg["pimax"]

    has_esd = "esd_rp" in data
    if has_esd:
        esd_rp = data["esd_rp"]
        esd_data = data["esd_data"]
        esd_err = data.get("esd_err", None)

    hod_obj, cfg = init_hod_from_cfg(cfg)

    nsample = min(args.nsample, len(chain))
    idx = np.random.choice(len(chain), nsample, replace=False)

    wp_samples = np.zeros((nsample, len(rp)))
    if has_esd:
        esd_samples = np.zeros((nsample, len(esd_rp)))

    print(f"Evaluating {nsample} model samples...")
    for i, j in enumerate(idx):
        set_params(hod_obj, chain[j], param_names, cfg)
        wp_samples[i] = eval_wp(hod_obj, rp, z, pimax)
        if has_esd:
            esd_samples[i] = eval_esd(hod_obj, esd_rp, z)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{nsample}")

    # Best-fit = max log_prob sample
    ibest = np.argmax(log_prob)
    set_params(hod_obj, chain[ibest], param_names, cfg)
    wp_best = eval_wp(hod_obj, rp, z, pimax)
    if has_esd:
        esd_best = eval_esd(hod_obj, esd_rp, z)

    wp_lo, wp_med, wp_hi = np.percentile(wp_samples, [16, 50, 84], axis=0)

    nrows = 2 if has_esd else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 4.5 * nrows), squeeze=False)

    # --- wp panel ---
    ax = axes[0, 0]
    ax.errorbar(rp, wp_data, yerr=wp_err, fmt="ko", ms=4, capsize=2,
                label="Data", zorder=3)
    ax.plot(rp, wp_best, "r-", lw=1.5, label="Best fit", zorder=2)
    ax.fill_between(rp, wp_lo, wp_hi, color="r", alpha=0.2, label=r"$1\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]", fontsize=12)
    ax.set_ylabel(r"$w_p(r_p)$ [$h^{-1}$ Mpc]", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f"wp Kaiser (hodtype={int(cfg['hodtype'])}, z={z:.2f})")

    # --- ESD panel ---
    if has_esd:
        esd_lo, esd_med, esd_hi = np.percentile(esd_samples, [16, 50, 84], axis=0)
        ax2 = axes[1, 0]
        if esd_err is not None:
            ax2.errorbar(esd_rp, esd_data, yerr=esd_err, fmt="ko", ms=4,
                         capsize=2, label="Data", zorder=3)
        else:
            ax2.plot(esd_rp, esd_data, "ko", ms=4, label="Data", zorder=3)
        ax2.plot(esd_rp, esd_best, "r-", lw=1.5, label="Best fit", zorder=2)
        ax2.fill_between(esd_rp, esd_lo, esd_hi, color="r", alpha=0.2,
                         label=r"$1\sigma$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]", fontsize=12)
        ax2.set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot\,{\rm pc}^{-2}$]",
                       fontsize=12)
        ax2.legend(fontsize=10)
        ax2.set_title(r"$\Delta\Sigma$ (ESD)")

    fig.tight_layout()
    out = args.output or args.chain.replace(".npz", "_wp_esd_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
