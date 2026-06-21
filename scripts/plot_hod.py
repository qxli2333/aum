#!/usr/bin/env python
"""Plot HOD (Ncen, Nsat, Ntot) with 1-sigma bands from chain."""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_utils import load_chain, init_hod_from_cfg, set_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chain", help="Chain .npz file")
    parser.add_argument("--nsample", type=int, default=300,
                        help="Number of chain samples to evaluate")
    parser.add_argument("--mrange", type=float, nargs=2, default=[10.5, 15.5],
                        metavar=("MMIN", "MMAX"),
                        help="log10(M) range")
    parser.add_argument("--nbins", type=int, default=80)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    data = load_chain(args.chain)
    chain = data["chain"]
    param_names = data["param_names"]
    cfg = data["cfg"]

    hod_obj, cfg = init_hod_from_cfg(cfg)

    xm = np.linspace(args.mrange[0], args.mrange[1], args.nbins)
    nsample = min(args.nsample, len(chain))
    idx = np.random.choice(len(chain), nsample, replace=False)

    ncen_all = np.zeros((nsample, len(xm)))
    nsat_all = np.zeros((nsample, len(xm)))

    for i, j in enumerate(idx):
        set_params(hod_obj, chain[j], param_names, cfg)
        for k, m in enumerate(xm):
            ncen_all[i, k] = hod_obj.ncen(m)
            nsat_all[i, k] = hod_obj.nsat(m)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{nsample}")

    ntot_all = ncen_all + nsat_all

    def get_bands(arr):
        lo, med, hi = np.percentile(arr, [16, 50, 84], axis=0)
        return med, lo, hi

    ncen_med, ncen_lo, ncen_hi = get_bands(ncen_all)
    nsat_med, nsat_lo, nsat_hi = get_bands(nsat_all)
    ntot_med, ntot_lo, ntot_hi = get_bands(ntot_all)

    M = 10**xm

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(M, ncen_med, "b-", label=r"$\langle N_{\rm cen}\rangle$")
    ax.fill_between(M, ncen_lo, ncen_hi, color="b", alpha=0.2)

    ax.plot(M, nsat_med, "r--", label=r"$\langle N_{\rm sat}\rangle$")
    ax.fill_between(M, nsat_lo, nsat_hi, color="r", alpha=0.2)

    ax.plot(M, ntot_med, "k-", lw=1.5, label=r"$\langle N_{\rm tot}\rangle$")
    ax.fill_between(M, ntot_lo, ntot_hi, color="gray", alpha=0.15)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel(r"$M_{200}$ [$h^{-1}\,M_\odot$]", fontsize=13)
    ax.set_ylabel(r"$\langle N \rangle$", fontsize=13)
    ax.set_title(f"HOD (hodtype={int(cfg['hodtype'])}, z={cfg['z_eff']:.2f})")
    ax.legend(fontsize=11)
    fig.tight_layout()

    out = args.output or args.chain.replace(".npz", "_hod.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
