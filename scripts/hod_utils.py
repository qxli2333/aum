"""Shared utilities for loading chains and evaluating HOD models."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import hod as h


def load_chain(path):
    d = np.load(path, allow_pickle=True)
    chain = d["chain"]
    log_prob = d["log_prob"]
    param_names = list(d["param_names"])
    cfg = dict(d["config"].item())
    result = dict(
        chain=chain, log_prob=log_prob, param_names=param_names, cfg=cfg,
        rp=d["rp"], wp_data=d["wp_data"], cov_wp=d["cov_wp"],
    )
    if "esd_rp" in d.files:
        result["esd_rp"] = d["esd_rp"]
        result["esd_data"] = d["esd_data"]
        if "esd_err" in d.files:
            result["esd_err"] = d["esd_err"]
    return result


def init_hod_from_cfg(cfg):
    p = h.cosmo()
    p.Om0 = cfg["Om0"]; p.Omk = cfg["Omk"]
    p.w0 = cfg["w0"]; p.wa = cfg["wa"]
    p.hval = cfg["hval"]; p.Omb = cfg["Omb"]
    p.th = cfg["th"]; p.s8 = cfg["s8"]
    p.nspec = cfg["nspec"]; p.ximax = cfg["ximax"]
    p.cfac = cfg["cfac"]

    q = h.hodpars()
    q.Mmin = 13.0; q.siglogM = 0.5; q.Msat = 14.0
    q.alpsat = 1.0; q.Mcut = 13.5
    q.csbycdm = cfg["csbycdm"]; q.fac = cfg.get("fac", 1.0)
    q.hodtype = int(cfg["hodtype"])
    q.Acen = cfg.get("Acen", 0.0); q.Asat = cfg.get("Asat", 0.0)
    q.Mq = cfg.get("Mq", 16.0); q.sigq = cfg.get("sigq", 0.5)
    q.sig_lnc = cfg.get("sig_lnc", 0.0)

    return h.hod(p, q), cfg


def set_params(hod_obj, theta, param_names, cfg):
    params = dict(zip(param_names, theta))
    q = h.hodpars()
    q.Mmin = params["Mmin"]
    q.siglogM = params["siglogM"]
    q.Msat = params["Msat"]
    q.alpsat = params["alpsat"]
    q.Mcut = params["Mcut"]
    q.csbycdm = cfg["csbycdm"]
    q.fac = params.get("fac", cfg.get("fac", 1.0))
    q.hodtype = int(cfg["hodtype"])
    q.Acen = params.get("Acen", cfg.get("Acen", 0.0))
    q.Asat = params.get("Asat", cfg.get("Asat", 0.0))
    q.Mq = params.get("Mq", cfg.get("Mq", 16.0))
    q.sigq = params.get("sigq", cfg.get("sigq", 0.5))
    q.sig_lnc = params.get("sig_lnc", cfg.get("sig_lnc", 0.0))
    hod_obj.hod_renew(q)


def eval_wp(hod_obj, rp_bins, z, pimax):
    nbins = len(rp_bins)
    rp_arr = h.doubleArray(nbins)
    wp_arr = h.doubleArray(nbins)
    for i in range(nbins):
        rp_arr[i] = rp_bins[i]
    hod_obj.Wp_Kaiser(z, nbins, rp_arr, wp_arr, pimax)
    return np.array([wp_arr[i] for i in range(nbins)])


def eval_esd(hod_obj, rp_bins, z):
    nbins = len(rp_bins)
    rp_arr = h.doubleArray(nbins)
    esd_arr = h.doubleArray(nbins)
    for i in range(nbins):
        rp_arr[i] = rp_bins[i]
    hod_obj.ESD(z, nbins, rp_arr, esd_arr, nbins + 6, True)
    return np.array([esd_arr[i] for i in range(nbins)])
