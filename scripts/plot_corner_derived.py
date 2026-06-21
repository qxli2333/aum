#!/usr/bin/env python
"""Corner plot of galaxy bias and satellite fraction from HOD chain."""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_utils import load_chain, init_hod_from_cfg, set_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chain", help="Chain .npz file")
    parser.add_argument("--nsample", type=int, default=500,
                        help="Number of chain samples to evaluate")
    parser.add_argument("-o", "--output", default=None,
                        help="Output figure path")
    args = parser.parse_args()

    data = load_chain(args.chain)
    chain = data["chain"]
    param_names = data["param_names"]
    cfg = data["cfg"]
    z = cfg["z_eff"]

    hod_obj, cfg = init_hod_from_cfg(cfg)

    nsample = min(args.nsample, len(chain))
    idx = np.random.choice(len(chain), nsample, replace=False)

    bias_arr = np.zeros(nsample)
    fsat_arr = np.zeros(nsample)

    for i, j in enumerate(idx):
        set_params(hod_obj, chain[j], param_names, cfg)
        ncen = hod_obj.ncenz(z)
        nsat = hod_obj.nsatz(z)
        ng = ncen + nsat
        if ng > 0:
            fsat_arr[i] = nsat / ng
            bias_arr[i] = hod_obj.galaxy_bias(z)
        else:
            bias_arr[i] = np.nan
            fsat_arr[i] = np.nan
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{nsample}")

    mask = np.isfinite(bias_arr) & np.isfinite(fsat_arr)
    bias_arr = bias_arr[mask]
    fsat_arr = fsat_arr[mask]
    print(f"Valid samples: {mask.sum()}/{nsample}")

    try:
        import corner
        samples = np.column_stack([bias_arr, fsat_arr])
        fig = corner.corner(samples, labels=["$b_g$", "$f_{\\rm sat}$"],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 12})
    except ImportError:
        fig, axes = plt.subplots(2, 2, figsize=(7, 7))

        axes[0, 0].hist(bias_arr, bins=30, color="steelblue", histtype="stepfilled",
                        alpha=0.7)
        axes[0, 0].set_xlabel("$b_g$")
        med_b = np.median(bias_arr)
        lo_b, hi_b = np.percentile(bias_arr, [16, 84])
        axes[0, 0].set_title(f"$b_g = {med_b:.2f}^{{+{hi_b-med_b:.2f}}}_{{-{med_b-lo_b:.2f}}}$",
                             fontsize=11)

        axes[1, 1].hist(fsat_arr, bins=30, color="indianred", histtype="stepfilled",
                        alpha=0.7)
        axes[1, 1].set_xlabel("$f_{\\rm sat}$")
        med_f = np.median(fsat_arr)
        lo_f, hi_f = np.percentile(fsat_arr, [16, 84])
        axes[1, 1].set_title(f"$f_{{\\rm sat}} = {med_f:.3f}^{{+{hi_f-med_f:.3f}}}_{{-{med_f-lo_f:.3f}}}$",
                             fontsize=11)

        axes[1, 0].scatter(bias_arr, fsat_arr, s=2, alpha=0.3, c="gray")
        axes[1, 0].set_xlabel("$b_g$")
        axes[1, 0].set_ylabel("$f_{\\rm sat}$")

        axes[0, 1].axis("off")
        fig.tight_layout()

    out = args.output or args.chain.replace(".npz", "_corner_bias_fsat.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    print(f"\nbias = {np.median(bias_arr):.3f} "
          f"+{np.percentile(bias_arr,84)-np.median(bias_arr):.3f} "
          f"-{np.median(bias_arr)-np.percentile(bias_arr,16):.3f}")
    print(f"fsat = {np.median(fsat_arr):.4f} "
          f"+{np.percentile(fsat_arr,84)-np.median(fsat_arr):.4f} "
          f"-{np.median(fsat_arr)-np.percentile(fsat_arr,16):.4f}")


if __name__ == "__main__":
    main()
