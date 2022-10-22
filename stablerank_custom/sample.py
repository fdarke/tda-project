
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu  Sep 22, 2022

@author: Wojciech chacholski

Copyright Wojciech chacholski, 2022
This software is to be used only for activities related  to TDA course  SF2956 2022
"""

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

inf = float("inf")


class Sample(object):
    def __init__(self, s, out_of):
        """
        Parameters
        ----------
        s: str, nd-array
            it can be either the string "all" or "empty", or 2-D nd-array of integers.
        out_of: int
        """
        self.sample = s
        self.out_of = out_of



def get_sample(number_instances, sample_size, probabilities):
    p = probabilities
    if isinstance(p, int):
        if sample_size <= p:
            out = np.zeros([number_instances, sample_size], dtype=int)
            for inst in range(number_instances):
                out[inst] = np.sort(np.random.choice(p, sample_size, replace=False))
            return Sample(out, p)
        return Sample("empty", p)
    return _get_sample_single(number_instances, sample_size, p)


def _get_sample_single(number_instances, sample_size, probabilities):
    out_of = len(probabilities)
    _tm = np.sum(probabilities)
    if _tm == 0:
        return Sample("empty", out_of)
    size = np.count_nonzero(probabilities)
    if sample_size <= size:
        out = np.zeros([number_instances, sample_size], dtype=int)
        for inst in range(number_instances):
            out[inst] = np.sort(np.random.choice(out_of, sample_size, replace=False, p=probabilities/_tm))
        return Sample(out, out_of)
    return Sample("empty", out_of)


class Distribution(object):

    def __init__(self, content):
        self.content = content

    def __call__(self, x):
        return self._evaluate(x)

    def _evaluate(self, x):
        v = 0
        for d in self.content:
            v += _single_evaluate(d, x)
        s = np.sum(v, axis=-1)
        if np.ndim(v) == 1:
            if s != 0:
                v = v / s
            return v
        for _i in np.ndindex(s.shape):
            if s[_i] != 0:
                v[_i] = v[_i] / s[_i]
        return v

    def plot(self, interval, precision=-2, base=10, ax=None, **kwargs):
        if precision >= 0:
            step = base ** precision
        else:
            step = 1 / base ** (-precision)
        x = np.arange(interval[0], interval[1], step)
        y = self(x)
        s = np.sum(y)
        if s != 0:
            y = y / (np.sum(y) * step)
        if ax is None:
            ax = plt
        return ax.plot(x, y, **kwargs)

    def __add__(self, other):
        return Distribution([*self.content, *other.content])

    def __radd__(self, other):
        return self + other

    def __mul__(self, r):
        if r == 1:
            return self
        out = self.content.copy()
        for f in out:
            f["coeff"] = f["coeff"] * r
        return Distribution(out)

    def __rmul__(self, r):
        return self * r


def get_distribution(name="uniform", **kwargs):
    if name == "uniform":
        interval = kwargs.get("interval", [-inf, inf])
        return Distribution([{"name": name, "interval": interval, "coeff": np.double(1)}])
    if name == "norm":
        loc = kwargs.get("loc", np.double(0))
        scale = kwargs.get("scale", np.double(1))
        return Distribution([{"name": name, "loc": loc, "scale": scale, "coeff": np.double(1)}])
    if name == "plf":
        return Distribution([{"name": name, "plf": kwargs["plf"], "coeff": np.double(1)}])


def _single_evaluate(content, x):
    name = content["name"]
    coeff = content["coeff"]
    if name == "uniform":
        b = content["interval"][0]
        e = content["interval"][1]
        return np.greater_equal(x, b) * np.less_equal(x, e) * coeff * 1.0
    if name == "norm":
        return norm.pdf(x, loc=content["loc"], scale=content["scale"]) * coeff
    if name == "plf":
        a = content["plf"]
        domain = np.array(a[0], dtype="double")
        domain = np.concatenate([[domain[0] - 1], domain, [domain[-1] + 1]])
        values = np.array(a[1], dtype="double")
        values = np.concatenate([[values[0]], values, [values[-1]]])
        return interp1d(domain, values, fill_value="extrapolate", assume_sorted=True)(x) * coeff
    else:
        raise ValueError("""You can choose only between uniform, norm, or plf distributions""")
