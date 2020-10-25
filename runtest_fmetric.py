#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Run experiments to fit metrics for different costs, and save results."""

from multiprocessing import Pool
import numpy as np
from eomdp import utils as ut
from eomdp import policy as po

DSET = 'ofa_imgnet.npz'
SPATH = 'save/fm_fold%d_cost%d.npz'
PLIST = [(f, c) for f in range(3) for c in range(3)]


def runtest(params_fc):
    """Run test with (fold, cost_index)"""

    npz = {}

    fold, cost = params_fc
    dset = np.load(DSET)

    idx_tr = dset['split'] != fold
    idx_ts = dset['split'] == fold

    lgt_tr, gt_tr = dset['wlogit'][idx_tr], dset['gt'][idx_tr]
    lgt_ts = dset['wlogit'][idx_ts]
    tinv = ut.calib(lgt_tr, gt_tr)
    npz['tinv'] = tinv

    wrank_tr, srank_tr = dset['wrank'][idx_tr], dset['srank'][idx_tr]
    wrank_ts, srank_ts = dset['wrank'][idx_ts], dset['srank'][idx_ts]

    entr_tr, entr_ts = ut.entropy(lgt_tr, tinv), ut.entropy(lgt_ts, tinv)
    wcost_tr, scost_tr = ut.cost(wrank_tr, srank_tr, cost)
    wcost_ts, scost_ts = ut.cost(wrank_ts, srank_ts, cost)
    npz['wcost_tr'], npz['scost_tr'] = wcost_tr, scost_tr
    npz['wcost_ts'], npz['scost_ts'] = wcost_ts, scost_ts
    npz['entr_tr'] = entr_tr

    rew_tr = wcost_tr - scost_tr
    mtog = po.fitmetric(entr_tr, rew_tr)
    npz['mtog'] = np.stack(mtog)

    npz['metric_tr'] = np.interp(entr_tr, *mtog)
    npz['metric_ts'] = np.interp(entr_ts, *mtog)

    np.savez_compressed(SPATH % (fold, cost), **npz)
    print("Completed fold %d, cost %d" % (fold, cost))


if __name__ == "__main__":
    with Pool() as p:
        p.map(runtest, PLIST, chunksize=1)
