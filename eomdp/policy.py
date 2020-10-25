# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Functions for determining optimal offloading policy."""

import numpy as np
from . import utils as ut

_HFITRANGE = np.power(2.0, np.arange(-8, -3.5, 0.5))


def fitmetric(etrain, rtrue, _hrange=_HFITRANGE):
    """
    Fit mapping from entropy to offloading metric.

    Returns a tuple f = (xbins, ybins). Call with np.interp(theta, *f)
    to map vector theta of entropy values to metric.
    """
    xbins = np.linspace(np.min(etrain), np.max(etrain), 1000)

    # Fit ent -> rew with RBF bandwidth _h, and predict
    # on values in xbins.
    def pred(_h, ent, rew):
        outr = np.zeros_like(xbins)
        for idx in range(0, len(xbins), 100):
            _wt = -(xbins[idx:(idx+100), np.newaxis] - ent)**2
            _wt = _wt - np.max(_wt, 1, keepdims=True)
            _wt = np.exp(_wt/(_h*_h))
            _wt = _wt / np.sum(_wt, 1, keepdims=True)
            outr[idx:(idx+100)] = np.sum(_wt*rew, 1)
        return outr

    # Fit on (et0,rt0) and check on (et1,rt1)
    # to find best bandwidth.
    et0, rt0 = etrain[::2], rtrue[::2]
    et1, rt1 = etrain[1::2], rtrue[1::2]

    hrange = _hrange*(xbins[-1]-xbins[0])
    hbest, fbest = None, np.inf
    for _h in hrange:
        rpred1 = np.interp(et1, xbins, pred(_h, et0, rt0))
        cost = np.mean(np.abs(rpred1-rt1)**2)
        if cost < fbest:
            hbest, fbest = _h, cost

    # Final result is with best h fit to all data.
    ybins = pred(hbest, etrain, rtrue)

    return (xbins, ybins)


def mdp(rate, bdepth, traindata, discount=0.9999, itparam=(1e4, 1e-6)):
    """Find optimal policy thresholds given token bucket parameters."""

    qpm = ut.getqpm(rate, bdepth)
    assert qpm[0] < qpm[1]
    assert qpm[2] >= qpm[1]

    # Sort metrics and compute F(theta) and G(theta)
    def summarize():
        metrics, rewards = traindata
        idx = np.argsort(-metrics)
        metrics, rewards = np.float64(metrics[idx]), np.float64(rewards[idx])
        gtheta = np.cumsum(rewards) / len(rewards)
        ftheta = np.float64(np.arange(1, len(rewards)+1)) / len(rewards)
        fgt = (np.reshape(ftheta, (-1, 1)), np.reshape(gtheta, (-1, 1)))
        return metrics, fgt

    metrics, fgt = summarize()
    thresh = np.amax(np.abs(metrics))*itparam[1]

    # Do value iterations
    value, policy = np.zeros((qpm[2]-qpm[0]+1), np.float64), None
    for i in range(int(itparam[0])):
        vprev, pprev = value.copy(), policy
        vprev = np.concatenate((vprev, [vprev[-1]]*qpm[0]))

        # If n < P/P, can't send
        value[:(qpm[1]-qpm[0])] = discount*vprev[qpm[0]:qpm[1]]

        # If n >= P/P:
        vnosend = vprev[qpm[1]:(qpm[2]+1)]
        vsend = vprev[:(qpm[2]-qpm[1]+1)]
        score = fgt[1] + discount*(fgt[0]*vsend + (1-fgt[0])*vnosend)
        value[(qpm[1]-qpm[0]):] = np.amax(score, 0)
        policy = metrics[np.argmax(score, 0)]

        if i > 0:
            if np.max(np.abs(policy-pprev)) < thresh:
                break

    return policy
