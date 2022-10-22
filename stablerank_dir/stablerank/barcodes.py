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

from stablerank.rtorf import Pcf
from stablerank.rtorf import Density
from stablerank.rtorf import Pcnif
from stablerank.rtorf import zero_pcnif
inf = float("inf")


class Contour(object):
    
    def __init__(self, plif="standard", truncation=inf):
        self.plif = plif
        self.truncation = truncation

    def plot_plif(self, interval, step=0.1, ax=None, **kwargs):
        x = np.arange(interval[0], interval[1], step)
        if ax is None:
            ax = plt
        if isinstance(self.plif, str) and self.plif == "standard":
            return ax.plot(x, x, **kwargs)
        return ax.plot(x, self.plif(x), **kwargs)


def standard_contour(truncation=inf):
    return Contour("standard", truncation=truncation)


def get_contour(density, kind, truncation):
    if isinstance(density, str) and density == "standard":
        return Contour("standard", truncation)
    if kind == "dist":
        f = density.antiderivative()
        return Contour(f, truncation)
    if kind == "area":
        g = density.inverse_antiderivative()
        return Contour(g, truncation)
    raise ValueError("""the parameter kind should be either "dist" or "area".""")


class BC(object):
    def __init__(self, bars):
        self.bars = bars

    def plot(self, ax=None, color_finite_bars="red", color_infinite_bars="blue"):
        bars = self.bars
        if ax is None:
            ax = plt
        ax.yticks([])
        if len(bars) > 0:
            m = np.amax(bars[bars != np.inf])
            ind_fin = np.isfinite(bars).all(axis=1)
            bars_fin = bars[ind_fin]
            pos_fin = np.arange(0, len(bars_fin))
            plt.hlines(pos_fin, bars_fin[:, 0], bars_fin[:, 1], color=color_finite_bars, linewidth=0.6)
            ind_inf = np.isinf(bars).any(axis=1)
            bars_inf = bars[ind_inf]
            pos_inf = np.arange(len(bars_fin),
                                len(bars_fin) + len(bars_inf))
            if m == 0:
                ends = np.ones(len(bars_inf))*10
            else:
                ends = np.ones(len(bars_inf)) * 2 * m
            ax.hlines(pos_inf, bars_inf[:, 0], ends, color=color_infinite_bars, linewidth=0.6)
        else:
            ax.text(0.2, 0.2, "empty bar code")

    def persistence_diagram(self,
                            ax=None,
                            ext=0.2,
                            finite_bars_size=None,
                            finite_bars_color="red",
                            infinite_bars_size=None,
                            infinite_bars_color="blue",
                            diagonal_color="black",
                            diagonal_width=None):

        if ax is None:
            ax = plt
        bars = self.bars
        finite_bars = np.array([a for a in bars if a[1] < inf])
        infinite_bars = np.array([a for a in bars if a[1] == inf])
        m = np.amax(finite_bars)
        ax.xlim([0 - ext, m + ext])
        ax.ylim([0 - ext, m + ext])
        if diagonal_width is None:
            ax.axline((1, 1), slope=1, c=diagonal_color)
        else:
            ax.axline((1, 1), slope=1, c=diagonal_color, linewidth=diagonal_width)
        if finite_bars_size is None:
            ax.scatter(finite_bars[:, 0], finite_bars[:, 1], c=finite_bars_color)
        else:
            ax.scatter(finite_bars[:, 0], finite_bars[:, 1], s=finite_bars_size, c=finite_bars_color)
        if len(infinite_bars) > 0:
            if infinite_bars_size is None:
                ax.scatter(infinite_bars[:, 0], np.ones(len(infinite_bars)) * (m + ext), c=infinite_bars_color)
            else:
                ax.scatter(infinite_bars[:, 0], np.ones(len(infinite_bars)) * (m + ext), s=infinite_bars_size,
                           c=infinite_bars_color)

    def length(self, contour=standard_contour()):
        if len(self.bars) == 0:
            return np.empty([0], dtype="double")
        plif = contour.plif
        truncation = contour.truncation
        if isinstance(plif, str) and plif == "standard":
            if truncation == inf:
                return self.bars[:, 1] - self.bars[:, 0]
            else:
                bars = np.array(self.bars, dtype="double")
                bars[np.where(bars > truncation)] = truncation
                return bars[:, 1] - bars[:, 0]
        else:
            if truncation == inf:
                inf_places = np.where(self.bars[:, 1] == inf)
                if len(inf_places) == 0:
                    return plif(self.bars[:, 1]) - plif(self.bars[:, 0])
                else:
                    finite_places = np.where(self.bars[:, 1] < inf)
                    out = np.zeros(len(self.bars), dtype="double")
                    out[finite_places] = plif(self.bars[finite_places, 1]) - plif(self.bars[finite_places, 0])
                    out[inf_places] = inf
                    return out
            else:
                bars = np.array(self.bars, dtype="double")
                bars[np.where(bars > truncation)] = truncation
                return plif(bars[:, 1]) - plif(bars[:, 0])

    def stable_rank(self, contour=standard_contour(), w_p=inf, w_q=1):
        if len(self.bars) == 0:
            return zero_pcnif()

        bar_length = self.length(contour)

        if w_p == inf:
            sort_length = np.unique(bar_length, return_counts=True)
            dom = sort_length[0]
            values = sort_length[1]
            if dom[-1] == inf:
                dom = dom[:-1]
            else:
                values = np.concatenate([values, [0]],  dtype="double")
            if len(dom) == 0 or dom[0] != 0:
                dom = np.concatenate([[0], dom],  dtype="double")
            else:
                values = values[1:]
            values = np.cumsum(values[::-1])[::-1]
            out = np.asarray(np.vstack((dom, values)), dtype="double")
            return Pcnif(out)
        finite_length = np.sort([_l for _l in bar_length if 0 < _l < inf])
        dom = np.cumsum(finite_length ** w_p) ** (1 / w_p)
        if w_q == inf:
            dom *= (1 / 2)
        else:
            dom *= 2 ** ((1 - w_q) / w_q)
        dom = np.concatenate([[0], dom])
        n_finite = len(finite_length)
        n_inf = len(bar_length) - n_finite
        values = np.array(np.arange(1, n_finite + 1)[::-1] + n_inf, dtype="double")
        values = np.concatenate([values, [n_inf]])
        return Pcnif(np.vstack((dom, values)))

    def betti(self):
        b = self.bars
        dom = b.reshape(-1)
        dom = np.unique(dom[dom < inf])
        val = [sum((b[:, 0] <= k) & (b[:, 1] > k)) for k in dom]

        out = np.concatenate(([[-inf], [0.]], np.vstack([dom, val])), axis=1, dtype="double")
        return Pcf(out)



def empty_space_bc(maxdim):
    out = {"H" + str(h): BC(np.empty([0, 2], dtype="double")) for h in range(maxdim+1)}
    return out


def one_point_bc(maxdim):
    out = {"H0": BC(np.array([[0, inf]], dtype="double"))}
    h = 1
    while h <= maxdim:
        out["H" + str(h)] = BC(np.empty([0, 2], dtype="double"))
        h += 1
    return out
