#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Get and save policies many (r,b) combinations to test with multiple cams."""

from multiprocessing import Pool
import numpy as np
from eomdp import policy as po

FMPATH = 'save/fm_fold%d_cost%d.npz'
OPATH = 'save/mcp_ri%03d_bi%04d_f%d_c%d.npy'
PLIST = [(f, r/40, b/4, c)
         for b in range(4, 41)
         for r in range(2, 21)
         for f in range(3)
         for c in [1]] + [(f, r, b, c)
                          for r in [0.05, 0.1, 0.25]
                          for b in range(12, 17, 2)
                          for f in range(3)
                          for c in [1]]


def runtest(params_frbc):
    """Run test with (fold, rate, bdepth, cost)"""

    fold, rate, bdepth, cost = params_frbc

    dset = np.load(FMPATH % (fold, cost))
    metr_tr = dset['metric_tr']
    rew_tr = dset['wcost_tr']-dset['scost_tr']
    policy = po.mdp(rate, bdepth, (metr_tr, rew_tr))

    np.save(OPATH % (int(rate*1000), int(bdepth*100), fold, cost), policy)
    print("Completed frbc=%d, %f, %f, %d" % (fold, rate, bdepth, cost))


if __name__ == "__main__":
    with Pool() as p:
        p.map(runtest, PLIST, chunksize=1)
