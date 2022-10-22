#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu  Sep 22, 2022

@author: Wojciech chacholski

Copyright Wojciech chacholski, 2022
This software is to be used only for activities related  to TDA course  SF2956 2022
"""



import numpy as np

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial as spatial
from ripser import ripser

from stablerank_custom.rtorf import Pcf
from stablerank_custom.rtorf import Pcnif
from stablerank_custom.rtorf import zero_pcnif
from stablerank_custom.rtorf import one_pcnif

from stablerank_custom.barcodes import BC
from stablerank_custom.barcodes import Contour
from stablerank_custom.barcodes import standard_contour
from stablerank_custom.barcodes import empty_space_bc
from stablerank_custom.barcodes import one_point_bc

from stablerank_custom.sample import Sample
from stablerank_custom.sample import get_sample

inf = float("inf")
nan = np.nan



class Distance(object):
    def __init__(self, content, content_demography):
        self.content = content
        self.content_demography = content_demography

    def size(self):
        if isinstance(self.content, str):
            return 0
        return int((1 + np.sqrt(1 + 8 * len(self.content))) / 2)

    def square_form(self):
        if isinstance(self.content, str):
            return np.empty([0, 0])
        return spatial.distance.squareform(self.content, checks=False)

    def get_h0sr(self,
                 sample=None,
                 clustering_method="single",
                 contour=standard_contour(),
                 w_p=inf,
                 w_q=1,
                 reduced=True):
 
        if isinstance(self.content, str) and self.content == "empty":
            return zero_pcnif()
        if sample is None:
            return _d_to_h0sr(self.content, clustering_method, contour, w_p, w_q, reduced)
        if isinstance(sample, Sample):
            return self._get_h0sr_single(sample, clustering_method, contour, w_p, w_q, reduced)
        shape = np.shape(sample)
        out = np.empty(shape, dtype=Pcnif)
        for _i, s in np.ndenumerate(sample):
            out[_i] = self._get_h0sr_single(s, clustering_method, contour, w_p, w_q, reduced)
        return out

    def _get_h0sr_single(self, sample, clustering_method, contour, w_p, w_q, reduced):
        if isinstance(sample.sample, str) and sample.sample == "all":
            return _d_to_h0sr(self.content, clustering_method, contour, w_p, w_q, reduced)
        if isinstance(sample.sample, str) and sample.sample == "empty":
            return zero_pcnif()
        d = self.square_form()
        number_instances = len(sample.sample)
        f = zero_pcnif()
        inst = 0
        while inst < number_instances:
            ind = sample.sample[inst]
            dd = spatial.distance.squareform(d[np.ix_(ind, ind)], checks=False)
            g = _d_to_h0sr(dd, clustering_method, contour, w_p, w_q, reduced)
            f += g
            inst += 1
        return f / number_instances

    def get_bc(self, sample=None, maxdim=1, thresh=inf, coeff=2, reduced=True):
        if sample is None:
            return _d_to_bc(self.square_form(), maxdim, thresh, coeff, reduced)
        if isinstance(sample, Sample):
            return self._get_bc_single(sample, maxdim, thresh, coeff, reduced)
        shape = sample.shape
        out = {"H" + str(d): np.empty(shape, dtype=set) for d in range(maxdim + 1)}
        for _i in np.ndindex(*shape):
            _bar_code = self._get_bc_single(sample[_i], maxdim, thresh, coeff, reduced)
            for k in out.keys():
                out[k][_i] = _bar_code[k]
        return out

    def _get_bc_single(self, sample, maxdim, thresh, coeff, reduced):
        if isinstance(sample.sample, str) and sample.sample == "all":
            _b = _d_to_bc(self.square_form(), maxdim, thresh, coeff, reduced)
            return {a: set([_b[a]]) for a in _b.keys()}
        if isinstance(sample.sample, str) and sample.sample == "empty":
            _b = empty_space_bc(maxdim)
            return {a: set([_b[a]]) for a in _b.keys()}
        d = self.square_form()
        number_instances = len(sample.sample)
        out = {"H" + str(d): set() for d in range(maxdim+1)}
        inst = 0
        while inst < number_instances:
            ind = sample.sample[inst]
            _b = _d_to_bc(d[np.ix_(ind, ind)], maxdim, thresh, coeff, reduced)
            for _h in out.keys():
                out[_h].add(_b[_h])
            inst += 1
        return out


def bc_to_sr(bar_code, degree="H1", contour=standard_contour(),  w_p=inf, w_q=1):
    _b = bar_code[degree]
    if isinstance(_b, BC):
        return _b.stable_rank(contour, w_p, w_q)
    if isinstance(_b, set):
        return _bc_to_sr_single(_b, contour,  w_p, w_q)
    out = np.empty(_b.shape, dtype=Pcnif)
    for _i in np.ndindex(out.shape):
        out[_i] = _bc_to_sr_single(_b[_i], contour,  w_p, w_q)
    return out


def _bc_to_sr_single(bar_code, contour,  w_p, w_q):
    f = zero_pcnif()
    if len(bar_code) > 0:
        for inst in bar_code:
            g = inst.stable_rank(contour, w_p, w_q)
            f += g
        return f * (1 / len(bar_code))
    return f


###################################################################


def _d_to_h0sr(d, clustering_method, contour, w_p, w_q, reduced):
    """d is assumed to be a 1D condense distance matrix"""
    if len(d) == 0:
        if reduced is True:
            return zero_pcnif()
        return one_pcnif()
    else:
        link = linkage(d, clustering_method)
        g = _linkage_to_stable_rank(link, contour, w_p, w_q, reduced)
        return g



def _d_to_bc(d, maxdim, thresh, coeff, reduced):
    """d is assumed to be a 2D square ndarray distance matrix with 0 on the diagonal"""
    dgms = ripser(d, maxdim=maxdim, thresh=thresh, coeff=coeff,
                             distance_matrix=True, do_cocycles=False)["dgms"]
    if reduced is True:
        out = {"H0": BC(dgms[0][:-1])}
        for h in range(1, maxdim + 1):
            out["H" + str(h)] = BC(dgms[h])
        return out
    return {"H" + str(h): BC(dgms[h]) for h in range(maxdim + 1)}


def _linkage_to_stable_rank(link, contour, w_p, w_q, reduced):
    i = 0
    b = np.empty([0, 2], dtype="double")
    while i < len(link):
        end = link[i, 2]
        if end != 0:
            b = np.vstack((b, [0, end]))
        i += 1
    if reduced is False:
        b = np.vstack((b, [0, inf]))
    return BC(b).stable_rank(contour, w_p, w_q)


def link_to_sr(link):
    return _linkage_to_stable_rank(link, contour=standard_contour(), w_p=inf, w_q=1, reduced=False)

