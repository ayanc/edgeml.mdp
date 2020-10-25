# - Ayan Chakrabarti <ayan.chakrabarti@gmail.com>
"""Functions for simulating sending with a given policy."""

import numpy as np
from numba import jit
from . import utils as ut


@jit(nopython=True)
def _simulate(qpm, policy, dset_mr, rsz_is):
    """Compiled implementation of simulate."""

    assert qpm[0] < qpm[1]
    assert qpm[2] >= qpm[1]
    assert len(policy) == qpm[2]-qpm[1]+1

    send_m = np.zeros((len(dset_mr[0]), 2), np.int64)
    send_s = np.zeros((qpm[2]-qpm[1]+1), np.float64)
    scmp = np.arange(qpm[1], qpm[2]+1).reshape((-1, 1))
    occup_s = np.zeros((qpm[2]-qpm[0]+1), np.float64)
    ocmp = np.arange(qpm[0], qpm[2]+1).reshape((-1, 1))

    nstate = qpm[2]*np.ones((rsz_is[1]), np.int64)
    imidx = np.random.randint(0, len(dset_mr[0]), rsz_is)
    avg_gain = 0.

    for i in range(rsz_is[0]):
        tidx = imidx[i, :]

        ifsend = dset_mr[0][tidx] >= policy[np.maximum(0, nstate-qpm[1])]
        ifsend = np.logical_and(ifsend, nstate >= qpm[1])

        occup_s = occup_s + np.sum(ocmp == nstate, 1)
        send_s = send_s + np.sum(scmp == nstate[ifsend], 1)
        for j, jidx in enumerate(tidx):
            send_m[jidx, 1] = send_m[jidx, 1] + 1
            if ifsend[j]:
                send_m[jidx, 0] = send_m[jidx, 0] + 1

        avg_gain = avg_gain + np.sum(np.where(ifsend, dset_mr[1][tidx], 0))
        nstate = nstate - np.where(ifsend, qpm[1], 0) + qpm[0]
        nstate = np.minimum(nstate, qpm[2])

    send_m = send_m[np.argsort(dset_mr[0]), :]

    denom = rsz_is[0]*rsz_is[1]
    occup_s, send_s = occup_s/denom, send_s/denom
    return avg_gain/denom, (send_m, send_s, occup_s)


def simulate(rate, bdepth, policy, dset_mr, rsz_is=(1e5, 1e2)):
    """Simulate policy on a single camera."""

    qpm = ut.getqpm(rate, bdepth)
    rsz_is = (int(rsz_is[0]), int(rsz_is[1]))
    return _simulate(qpm, policy, dset_mr, rsz_is)


@jit(nopython=True)
def _mcsimulate(qpm_i, qpm_g, ncam, policy, dset_mr, rsz_is):
    """Compiled implementation of mcsimulate."""

    assert qpm_i[0] < qpm_i[1]
    assert qpm_i[2] >= qpm_i[1]
    assert len(policy) == qpm_i[2]-qpm_i[1]+1
    assert qpm_g[0] < qpm_g[1]
    assert qpm_g[2] >= qpm_g[1]

    occup_s = np.zeros((qpm_g[2]-qpm_g[0]+1), np.float64)
    ocmp = np.arange(qpm_g[0], qpm_g[2]+1).reshape((-1, 1))

    imidx = np.random.randint(0, len(dset_mr[0]), (rsz_is[0], rsz_is[1]*ncam))
    nistate = qpm_i[2]*np.ones((rsz_is[1]*ncam), np.int64)
    ngstate = qpm_g[2]*np.ones((rsz_is[1]), np.int64)
    avg_gain = 0.

    for i in range(rsz_is[0]):
        tidx = imidx[i, :]

        ifsend = dset_mr[0][tidx] >= policy[np.maximum(0, nistate-qpm_i[1])]
        ifsend = np.logical_and(ifsend, nistate >= qpm_i[1])
        nistate = nistate - np.where(ifsend, qpm_i[1], 0) + qpm_i[0]
        nistate = np.minimum(nistate, qpm_i[2])

        for j in range(ncam):
            occup_s = occup_s + np.sum(ocmp == ngstate, 1)

            ifsendj, tidxj = ifsend[j::ncam], tidx[j::ncam]
            ifsendj = np.logical_and(ifsendj, ngstate >= qpm_g[1])
            avg_gain = avg_gain + \
                np.sum(np.where(ifsendj, dset_mr[1][tidxj], 0))
            ngstate = ngstate - np.where(ifsendj, qpm_g[1], 0) + qpm_g[0]
            ngstate = np.minimum(ngstate, qpm_g[2])

    denom = rsz_is[0]*rsz_is[1]*ncam
    return avg_gain/denom, occup_s/denom


def mcsimulate(rb_i, rb_g, ncam, policy, dset_mr, rsz_is=(1e5, 1e2)):
    """Simulate policy on multiple cameras interacting with a switch."""

    qpm_i, qpm_g = ut.getqpm(*rb_i), ut.getqpm(*rb_g)
    rsz_is = (int(rsz_is[0]), int(rsz_is[1]))
    return _mcsimulate(qpm_i, qpm_g, ncam, policy, dset_mr, rsz_is)
