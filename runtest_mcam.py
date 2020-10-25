#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Run experiments to compare strategies for multiple cameras."""

from multiprocessing import Pool
import numpy as np
from eomdp import simulate as sim

FMPATH = 'save/fm_fold%d_cost%d.npz'
PPATH = 'save/mcp_ri%03d_bi%04d_f%d_c%d.npy'
SPATH = 'save/mcs_rg%03d_bp_%04d_nc%d_ri%03d_bi%04d_f%d_c%d.npz'
BIS = [b/4 for b in range(4, 41)]  # Search range for per-device depth
RIS = [r/40 for r in range(2, 21)]  # Search range for per-device rate

OPATH = 'save/mcam_rg%03d_bp%04d_nc%d_c%d.npz'
PLIST = [(rg, bp, ncam, cost)
         for rg in [0.05, 0.1, 0.25]
         for bp in [1, 2]
         for ncam in range(2, 9)
         for cost in [1]]


def loadpolicy(fold, ri_, bi_, cost):
    """Loads pre-computed policy."""
    return np.load(PPATH % (int(ri_*1000), int(bi_*100), fold, cost))


def getscores(fold, r_g, b_p, ncam, cost):
    """Form matrix of training set simulation scores."""

    scores = np.zeros((len(RIS), len(BIS)), np.float64)
    for j, _b in enumerate(BIS):
        for i, _r in enumerate(RIS):
            if _r <= r_g and _b < b_p:
                scores[i, j] = np.NAN
                continue
            if _r < r_g and _b <= b_p:
                scores[i, j] = np.NAN
                continue

            sres = np.load(SPATH % (int(r_g*1000), int(b_p*100), ncam,
                                    int(_r*1000), int(_b*100),
                                    fold, cost))
            scores[i, j] = sres['gain']

    rbi = np.unravel_index(np.nanargmax(scores), scores.shape)
    rbi = (RIS[rbi[0]], BIS[rbi[1]])

    return scores, rbi


def runtest(params_rbnc):
    """Run test with (rate, per-cam bdepth, ncam, cost_idx)"""

    r_g, b_p, ncam, cost = params_rbnc
    b_g = b_p*ncam

    npz = {'iso': 0.0, 'hier': 0.0, 'smart': 0.0}
    for fold in range(3):
        dset = np.load(FMPATH % (fold, cost))
        metr_ts = dset['metric_ts']
        rew_ts = dset['wcost_ts']-dset['scost_ts']

        iso_p = loadpolicy(fold, r_g, b_p, cost)
        smart_p = loadpolicy(fold, r_g, b_g, cost)

        scores, rbi = getscores(fold, r_g, b_p, ncam, cost)
        hier_p = loadpolicy(fold, rbi[0], rbi[1], cost)

        iso_g, iso_o = sim.mcsimulate((r_g, b_p), (r_g, b_g), ncam,
                                      iso_p, (metr_ts, rew_ts))
        hier_g, hier_o = sim.mcsimulate(rbi, (r_g, b_g), ncam,
                                        hier_p, (metr_ts, rew_ts))
        smart_g, smart_o = sim.simulate(r_g, b_g, smart_p, (metr_ts, rew_ts))
        smart_o = smart_o[2]

        mwcost = np.mean(dset['wcost_ts'])
        npz['iso'] = npz['iso'] + (mwcost - iso_g)/3.0
        npz['smart'] = npz['smart'] + (mwcost - smart_g)/3.0
        npz['hier'] = npz['hier'] + (mwcost - hier_g)/3.0

        if fold == 0:
            npz['hscores'], npz['hbest'] = scores, rbi
            npz['ris'], npz['bis'] = np.float64(RIS), np.float64(BIS)
            npz['iso_o'] = iso_o
            npz['hier_o'] = hier_o
            npz['smart_o'] = smart_o

    np.savez_compressed(OPATH % (int(r_g*1000), int(b_p*100), ncam, cost),
                        **npz)
    print("Completed r_g=%f, b_p=%f, ncam=%d, cost=%d" % params_rbnc)


if __name__ == "__main__":
    with Pool() as p:
        p.map(runtest, PLIST, chunksize=1)
