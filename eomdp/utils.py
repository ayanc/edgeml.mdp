# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Utility functions for calibration and cost & entropy computation."""

import numpy as np

_COSTS = ['Top1-Error', 'Top5-Error', 'Rank']


def cost(wrank, srank, ctype=0):
    """Compute  costs from rank of GT in weak and strong classifier outputs."""
    if ctype == 0:
        return np.float64(wrank > 1), np.float64(srank > 1)
    if ctype == 1:
        return np.float64(wrank > 5), np.float64(srank > 5)
    return np.float64(np.minimum(10, wrank)), \
        np.float64(np.minimum(10, srank))


def entropy(logits, tinv):
    """Compute entropy from logits + calibration temperature."""
    logits = np.float64(np.transpose(logits))
    lnum = tinv*(logits - np.max(logits, 0))
    pden = np.sum(np.exp(lnum), 0)

    return - np.sum(np.exp(lnum)*lnum, 0)/pden + np.log(pden)


def calib(logits, gtlbl, _lb=0.0, _ub=2.0, _crounds=6):
    """Calibrate logits to find best factor that minimizes x-entropy."""
    logits, gtlbl = np.float32(logits), np.int64(gtlbl)
    logits = logits - np.max(logits, 1, keepdims=True)
    base = -np.mean(logits[np.arange(len(gtlbl)), gtlbl])

    for _ in range(_crounds):
        tinvs = np.linspace(_lb, _ub, 10, dtype=np.float32)
        xent = base*tinvs + np.mean(np.log(
            np.sum(np.exp(tinvs*logits[:, :, np.newaxis]), 1)
        ), 0)

        best = tinvs[np.argmin(xent)]
        _lb = np.maximum(0., best - tinvs[1] + tinvs[0])
        _ub = best + tinvs[1] - tinvs[0]

    return best


def getqpm(rate, bdepth, maxp=100):
    """Return integer (q,p,m) such that q/p ~ rate, m/p ~ bdepth."""
    denom = np.arange(maxp, dtype=np.int64)+1
    rerr, berr = denom*rate, denom*bdepth
    err = (rerr-np.floor(rerr)+berr-np.floor(berr))/denom
    p = denom[np.argmin(err)]
    q = np.int64(np.floor(rate*p))
    m = np.int64(np.floor(bdepth*p))
    gcd = np.gcd(np.gcd(q, p), m)
    return q//gcd, p//gcd, m//gcd


def getvpidx(rate, bdepth):
    """Get the token numbers for indices of value and policy vectors."""
    qpm = getqpm(rate, bdepth)
    vidx = np.arange(qpm[0], qpm[2]+1, dtype=np.float64)/qpm[1]
    pidx = np.arange(qpm[1], qpm[2]+1, dtype=np.float64)/qpm[1]
    return vidx, pidx
