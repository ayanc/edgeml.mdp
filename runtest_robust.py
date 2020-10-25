#!/usr/bin/env python3
# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Run single camera experiments with different train-test statistics."""

from multiprocessing import Pool
import numpy as np
from eomdp import simulate as sim
from eomdp import policy as po

FMPATH = 'save/fm_fold%d_cost%d.npz'
OPATH = 'save/1cam_r%03d_bdepth%02d_cost%d_dev%d.npz'
PLIST = [(r, b/2, c, dev)
         for b in range(2, 11)
         for r in [0.05, 0.1, 0.25]
         for c in [1]
         for dev in [-50, -25, -10, 10, 25, 50]]


def runtest(params_rbcd):
    """Run test with (rate, bdepth, cost, deviation)"""

    rate, bdepth, cost, dev = params_rbcd

    npz = {'dev': dev, 'mdpcost': 0., 'naivecost': 0.}
    for fold in range(3):
        dset = np.load(FMPATH % (fold, cost))

        metr_tr = dset['metric_tr']
        rew_tr = dset['wcost_tr']-dset['scost_tr']
        metr_ts = dset['metric_ts']
        rew_ts = dset['wcost_ts']-dset['scost_ts']

        idx = np.argsort(metr_tr)
        didx = int(dev/100*len(idx))
        idx = idx[didx:] if dev > 0 else idx[:didx]
        metr_d, rew_d = metr_tr[idx], rew_tr[idx]

        policy = po.mdp(rate, bdepth, (metr_d, rew_d))
        mdprew, stats = sim.simulate(rate, bdepth, policy, (metr_ts, rew_ts))

        nvpolicy = np.percentile(metr_d, (1.0-rate)*100.0) * \
            np.ones_like(policy)

        nvprew, _ = sim.simulate(rate, bdepth, nvpolicy, (metr_ts, rew_ts))

        mwcost = np.mean(dset['wcost_ts'])
        npz['mdpcost'] = npz['mdpcost'] + (mwcost - mdprew)/3.0
        npz['naivecost'] = npz['naivecost'] + (mwcost - nvprew)/3.0

        if fold == 0:
            npz['send_m'] = stats[0][:, 0] / stats[0][:, 1]
            npz['send_s'] = stats[1]
            npz['occup_s'] = stats[2]
            npz['policy'] = np.mean(policy >= metr_tr[:, np.newaxis], 0)

    np.savez_compressed(OPATH % (int(rate*1000), int(bdepth*10),
                                 cost, dev), **npz)
    print("Completed r=%f, b=%f, cost=%d, dev=%d" % (rate, bdepth, cost, dev))


if __name__ == "__main__":
    with Pool() as p:
        p.map(runtest, PLIST, chunksize=1)
