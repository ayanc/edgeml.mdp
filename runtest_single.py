#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Run experiments to derive and simulate policies for single cameras."""

from multiprocessing import Pool
import numpy as np
from eomdp import simulate as sim
from eomdp import policy as po

FMPATH = 'save/fm_fold%d_cost%d.npz'
OPATH = 'save/1cam_r%03d_bdepth%02d_cost%d.npz'
PLIST = [(r/20, b/2, c)
         for b in range(2, 11)
         for r in range(1, 11)
         for c in range(3)]


def runtest(params_rbc):
    """Run test with (rate, bdepth, cost)"""

    rate, bdepth, cost = params_rbc

    npz = {'lb': 0., 'wcost': 0., 'scost': 0.,
           'naivecost': 0., 'mdpcost': 0.}
    for fold in range(3):
        dset = np.load(FMPATH % (fold, cost))

        metr_tr = dset['metric_tr']
        rew_tr = dset['wcost_tr']-dset['scost_tr']
        metr_ts = dset['metric_ts']
        rew_ts = dset['wcost_ts']-dset['scost_ts']

        policy = po.mdp(rate, bdepth, (metr_tr, rew_tr))

        nvpolicy = np.percentile(metr_tr, (1.0-rate)*100.0)
        lbrew = np.mean(rew_ts * (metr_ts >= nvpolicy))
        nvpolicy = nvpolicy * np.ones_like(policy)

        mdprew, stats = sim.simulate(rate, bdepth, policy, (metr_ts, rew_ts))
        nvprew, nst = sim.simulate(rate, bdepth, nvpolicy, (metr_ts, rew_ts))

        mwcost = np.mean(dset['wcost_ts'])
        npz['wcost'] = npz['wcost'] + mwcost/3.0
        npz['scost'] = npz['scost'] + np.mean(dset['scost_ts'])/3.0
        npz['lb'] = npz['lb'] + (mwcost - lbrew)/3.0
        npz['mdpcost'] = npz['mdpcost'] + (mwcost - mdprew)/3.0
        npz['naivecost'] = npz['naivecost'] + (mwcost-nvprew)/3.0

        if fold == 0:
            npz['send_m'] = stats[0][:, 0] / stats[0][:, 1]
            npz['send_s'] = stats[1]
            npz['occup_s'] = stats[2]
            npz['policy'] = np.mean(policy >= metr_tr[:, np.newaxis], 0)
            npz['nsrate'] = np.sum(nst[1])
            npz['naive_m'] = nst[0][:, 0] / nst[0][:, 1]

    np.savez_compressed(OPATH % (int(rate*1000), int(bdepth*10), cost), **npz)
    print("Completed r=%f, b=%f, cost=%d" % (rate, bdepth, cost))


if __name__ == "__main__":
    with Pool() as p:
        p.map(runtest, PLIST, chunksize=1)
