#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Run all policies against all multi-device settings on train set."""

from multiprocessing import Pool
import numpy as np
from eomdp import simulate as sim

FMPATH = 'save/fm_fold%d_cost%d.npz'
PPATH = 'save/mcp_ri%03d_bi%04d_f%d_c%d.npy'
OPATH = 'save/mcs_rg%03d_bp_%04d_nc%d_ri%03d_bi%04d_f%d_c%d.npz'
PLIST = [(f, rg, bp, ncam, r/40, b/4, c)
         for ncam in range(2, 9)
         for rg in [0.05, 0.1, 0.25]
         for bp in [1, 2]
         for b in range(4, 41)
         for r in range(2, 21)
         for f in range(3)
         for c in [1]]


def runtest(params):
    """Run test with (fold, input/output (r,b), cost)"""

    fold, r_g, b_p, ncam, r_i, b_i, cost = params
    b_g = ncam*b_p
    if r_i <= r_g and b_i < b_p:
        return
    if r_i < r_g and b_i <= b_p:
        return

    dset = np.load(FMPATH % (fold, cost))
    tdata = (dset['metric_tr'], dset['wcost_tr']-dset['scost_tr'])
    policy = np.load(PPATH % (int(r_i*1000), int(b_i*100), fold, cost))

    gain, occups = sim.mcsimulate((r_i, b_i), (r_g, b_g), ncam, policy, tdata)

    npz = {'gain': gain, 'occups': occups}
    np.savez_compressed(OPATH % (int(r_g*1000), int(b_p*100), ncam,
                                 int(r_i*1000), int(b_i*100),
                                 fold, cost), **npz)
    print("Completed %d, %f, %f, %d, %f, %f, %d" % params)
    return


if __name__ == "__main__":
    with Pool() as p:
        p.map(runtest, PLIST, chunksize=1)
