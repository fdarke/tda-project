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
from scipy.interpolate import interp1d

inf = float("inf")
nan = float("nan")


class Pcf(object):

    def __init__(self, a):
        """
        Parameters
        ----------
        a: nd-array
            of shape 2xn of real numbers. It is assumed n>0 and that the 0-th row, referred to as the domain, starts
            with 0 and is followed by strictly increasing real numbers. The values of the second row can be arbitrary
            finite real numbers.
        """
        self.content = a

    def __call__(self, x):
        x_place = np.maximum(np.searchsorted(self.content[0], x, side='right')-1, 0)
        return self.content[1][x_place]

    def _extend(self, d):
        dom = self.content[0]
        domain = np.unique(np.concatenate([dom, d]))
        parameters = np.searchsorted(dom, domain, side='right') - 1
        val = self.content[1][parameters]
        return np.vstack((domain, val))

    def simplify(self):
        c = self.content[1][:-1] - self.content[1][1:]
        k = np.insert(np.where(c != 0)[0] + 1, 0, 0)
        return self.content[:, k]

    def plot(self, interval=None, ext_l=0.1, ext_r=0.1, ax=None, **kwargs):
        return _plot_pcfs(self.content[0], self.content[1], interval=interval, ext_l=ext_l, ext_r=ext_r,
                          ax=ax, **kwargs)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            c = np.array(self.content, copy=True)
            c[1] += other
            if isinstance(self, Pcnif) and other >= 0:
                return Pcnif(c)
            if isinstance(self, Density) and c > 0:
                return Density(c)
            return Pcf(c)
        if isinstance(other, Pcf):
            f = self._extend(other.content[0])
            g = other._extend(self.content[0])
            f[1] += g[1]
            if isinstance(self, Pcnif) and isinstance(other, Pcnif):
                return Pcnif(f)
            if isinstance(self, Density) and isinstance(other, Density):
                return Density(f)
            return Pcf(f)
        raise ValueError("""we can only add to a Pcf a Pcf or a real number""")

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        c = np.array(self.content, copy=True)
        c[1] = -c[1]
        return Pcf(c)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c = np.array(self.content, copy=True)
            c[1] *= other
            if other > 0:
                if isinstance(self, Pcnif):
                    return Pcnif(c)
                if isinstance(self, Density):
                    return Density(c)
                return Pcf(c)
            if other < 0:
                return Pcf(c)
            if isinstance(self, Pcnif):
                return Pcnif(np.array([[0], [0]], dtype="double"))
            return Pcf(np.array([[0], [0]], dtype="double"))
        if isinstance(other, Pcf):
            f = self._extend(other.content[0])
            g = other._extend(self.content[0])
            f[1] *= g[1]
            if isinstance(self, Pcnif) and isinstance(other, Pcnif):
                return Pcnif(f)
            if isinstance(self, Density) and isinstance(other, Density):
                return Density(f)
            return Pcf(f)
        raise ValueError("""we can only multiply a Pcf by a Pcf or a real number""")

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            c = np.array(self.content, copy=True)
            c[1] -= other
            return Pcf(c)
        if isinstance(other, Pcf):
            f = self._extend(other.content[0])
            g = other._extend(self.content[0])
            f[1] -= g[1]
            return Pcf(f)
        raise ValueError("""we can only subtract from a Pcf a Pcf or a real number""")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            c = np.array(self.content, copy=True)
            c[1] = -c[1]
            c[1] += other
            return Pcf(c)
        raise ValueError("""we can subtract a Pcf only from a real number or a  Pcf""")

    def __pow__(self, p):
        c = np.array(self.content, copy=True)
        c[1] = c[1] ** p
        if isinstance(self, Pcnif) and p >= 1:
            return Pcnif(c)
        if isinstance(self, Density):
            return Density(c)
        return Pcf(c)

    def __abs__(self):
        c = np.array(self.content, copy=True)
        c[1] = np.absolute(c[1])
        return Pcf(c)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            c = np.array(self.content, copy=True)
            c[1] = c[1] / other
            if other > 0:
                if isinstance(self, Pcnif):
                    return Pcnif(c)
                if isinstance(self, Density):
                    return Density(c)
                return Pcf(c)
            if other < 0:
                return Pcf(c)
            raise ValueError("""we can not dived by 0""")
        raise ValueError("""A Pcf can only be divided by a non zero real. To divide by a PCF try 
                self*(other**(-1))""")

    def __rtruediv__(self, other):
        return other * (self ** (-1))

    def antiderivative(self):
        values = np.diff(self.content[0])
        values = np.concatenate([values, [1]], dtype="double")
        values = values * self.content[1]
        values = np.cumsum(values)
        values = np.concatenate([[0], values], dtype="double")
        domain = self.content[0]
        domain = np.concatenate([domain, [domain[-1] + 1]], dtype="double")
        return interp1d(domain, values, fill_value="extrapolate", assume_sorted=True)

    def integrate(self, intervals=None):
        if intervals is None:
            intervals = [0, inf]
        inter = np.array(intervals, dtype="double")
        ad = self.antiderivative()
        last_value = self.content[1, -1]
        # intervals is assumed to be 1-D array on length 2
        if np.ndim(inter) == 1:
            if inter[1] < inf:
                return ad(inter[1]) - ad(inter[0])
            if last_value == 0:
                return ad(self.content[0, -1]) - ad(inter[0])
            if last_value > 0:
                return inf
            return -inf
        # intervals is assumed to be 2-D array of shape nx2 array
        inf_places = np.where(inter[:, 1] == inf)
        if len(inf_places) == 0:
            return ad(inter[:, 1]) - ad(inter[:, 0])
        if last_value == 0:
            inter[inf_places, 1] = self.content[0, -1]
            return ad(inter[:, 1]) - ad(inter[:, 0])
        finite_places = np.where(inter[:, 1] < inf)
        out = np.zeros(len(inter), dtype="double")
        out[finite_places] = ad(inter[finite_places, 1]) - ad(inter[finite_places, 0])
        if last_value > 0:
            out[inf_places] = inf
            return out
        out[inf_places] = -inf
        return out

    def dot(self, other, interval=None):
        return (self*other).integrate(intervals=interval)

    def lp_distance(self, other, p=1, interval=None):
        if interval is None:
            interval = [0, inf]
        if interval[1] == inf and self.content[1][-1] != other.content[1][-1]:
            return inf
        else:
            return (abs(self - other) ** p).integrate(interval) ** (1 / p)


class Density(Pcf):

    def plot(self, interval=None, ext_l=0, ext_r=0.1, ax=None, **kwargs):
        return _plot_pcfs(self.content[0], self.content[1], interval=interval, ext_l=ext_l, ext_r=ext_r,
                          ax=ax, **kwargs)

    def inverse_antiderivative(self):
        values = np.diff(self.content[0])
        values = np.concatenate([values, [1]])
        values = values * self.content[1]
        values = np.cumsum(values)
        values = np.concatenate([[0], values])
        domain = self.content[0]
        domain = np.concatenate([domain, [domain[-1] + 1]])
        return interp1d(values, domain, fill_value="extrapolate", assume_sorted=True)



class Pcnif(Pcf):

    def plot(self, interval=None, ext_l=0, ext_r=0.1, ax=None, **kwargs):
        return _plot_pcfs(self.content[0], self.content[1], interval=interval, ext_l=ext_l, ext_r=ext_r,
                          ax=ax, **kwargs)

    def interleaving_distance(self, other):
        if self.content[1, -1] != other.content[1, -1]:
            return inf

        cont1 = self.simplify()
        cont2 = other.simplify()
        cont1_set = set(cont1[1])
        cont2_set = set(cont2[1])
        common_vals = np.unique(np.concatenate((cont1[1], cont2[1])))

        c1 = 0
        c2 = 0
        intv = 0
        for v in common_vals[::-1]:

            cand = abs(cont1[0, c1] - cont2[0, c2])
            intv = cand if cand > intv else intv

            if v in cont1_set and c1 + 1 < len(cont1[1]):
                c1 += 1
            if v in cont2_set and c2 + 1 < len(cont2[1]):
                c2 += 1

        return intv


def zero_pcnif():
    return Pcnif(np.array([[0], [0]], dtype="double"))


def one_pcnif():
    return Pcnif(np.array([[0], [1]], dtype="double"))


def _plot_pcfs(domain, values, interval=None, ext_l=0.1, ext_r=0.1, ax=None, **kwargs):
    if interval is None:
        x = np.concatenate([[-ext_l], domain[1:], [domain[-1] + ext_r]])
        y = np.concatenate([values, [values[-1]]])
        if ax is None:
            ax = plt
        return ax.step(x, y, where='post', **kwargs)
    if interval[1] == inf:
        e = domain[-1] + ext_r
    else:
        e = interval[1] + ext_r
    b = max(interval[0] - ext_l, - ext_l)
    e_place = max(np.searchsorted(domain, e,  side='right'), 1)
    b_place = max(np.searchsorted(domain, b, side='left'), 1)
    x = np.concatenate([[max(b, -ext_l)], domain[b_place:e_place], [e]])
    y = np.concatenate([[values[b_place-1]], values[b_place:e_place], [values[e_place-1]]])
    print(x, y)
    if ax is None:
        ax = plt
    return ax.step(x, y, where='post', **kwargs)


