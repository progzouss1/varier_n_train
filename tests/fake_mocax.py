# -*- coding: utf-8 -*-
"""FAUSSES bibliotheques MoCaX -- UNIQUEMENT pour tester la tuyauterie (cache,
reprise, tableaux) sur une machine sans MoCaX. Les resultats n'ont AUCUNE valeur.

  fake_mocaxpy()        : Mocax / MocaxDomain / MocaxNs (vrai tenseur de
                          Tchebychev plein, interpolation de Lagrange)
  fake_mocax_extend()   : MocaxExtend (API minimale ; "fit" = regression
                          quadratique, pas un tensor-train)
"""
import types

import numpy as np


def _cheb_nodes(a, b, n):
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)[::-1]
    return a + (b - a) * (x + 1) / 2


def _lagrange(nodes, x):
    L = np.ones(len(nodes))
    for j in range(len(nodes)):
        for m in range(len(nodes)):
            if m != j:
                L[j] *= (x - nodes[m]) / (nodes[j] - nodes[m])
    return L


def fake_mocaxpy():
    mod = types.ModuleType("mocaxpy")

    class MocaxDomain(object):
        def __init__(self, r):
            self.r = [list(map(float, x)) for x in r]

    class MocaxNs(object):
        def __init__(self, ns):
            self.ns = list(ns)

    class Mocax(object):
        def __init__(self, f, d, domain, err, ns):
            self.d = d
            self.nodes = [_cheb_nodes(a, b, n) for (a, b), n in zip(domain.r, ns.ns)]
            self.shape = tuple(len(x) for x in self.nodes)
            self.vals = None

        def get_evaluation_points(self):
            grid = np.meshgrid(*self.nodes, indexing="ij")
            return np.stack([g.ravel() for g in grid], axis=1).tolist()

        def set_original_function_values(self, v):
            self.vals = np.asarray(v, dtype=float).reshape(self.shape)

        def get_error_threshold(self):
            return float("nan")

        def eval(self, p):
            for (lo, hi) in [(x[0], x[-1]) for x in self.nodes]:
                pass
            T = self.vals
            for i in range(self.d):
                T = np.tensordot(_lagrange(self.nodes[i], p[i]), T, axes=([0], [0]))
            return float(T)

    mod.Mocax, mod.MocaxDomain, mod.MocaxNs = Mocax, MocaxDomain, MocaxNs
    return mod


def fake_mocax_extend():
    mod = types.ModuleType("mocax_extend")

    class MocaxExtend(object):
        def __init__(self, d, num_cheb_pts, ranges):
            self.d = d
            self.num_cheb_pts = list(num_cheb_pts)
            self.intervals = np.asarray(ranges, dtype=float)
            self.cheb_pts_per_dimension = [_cheb_nodes(a, b, n) for (a, b), n
                                           in zip(ranges, num_cheb_pts)]

        def get_tensor_size(self):
            return int(np.prod(self.num_cheb_pts))       # bug volontaire (n^d)

        def subgrid_by_number(self, k):
            n = len(self.cheb_pts_per_dimension[0])
            ent = np.random.randint(0, n, size=(k, self.d))
            return np.column_stack([self.cheb_pts_per_dimension[j][ent[:, j]]
                                    for j in range(self.d)]).tolist()

        def set_subgrid(self, pts):
            P = np.asarray(pts, dtype=float)
            self.ent = np.column_stack([
                np.argmin(np.abs(P[:, [j]] - self.cheb_pts_per_dimension[j][None, :]), axis=1)
                for j in range(self.d)])
            self.P = P

        def set_subgrid_values(self, v):
            self.v = np.asarray(v, dtype=float)

        def gen_train_val_data(self, percentage=0.8):
            n = len(self.v)
            idx = np.random.RandomState(0).permutation(n)
            k = int(round(percentage * n))
            self.itr, self.iva = idx[:k], idx[k:]
            self.subgrid_entries_val = self.ent[self.iva]
            self.subgrid_vals_val = self.v[self.iva]

        def get_training_set_size(self):
            return len(self.itr)

        def get_validating_set_size(self):
            return len(self.iva)

        def _feat(self, X):
            X = (np.asarray(X, float) - self.intervals[:, 0]) / np.ptp(self.intervals, axis=1)
            return np.column_stack([np.ones(len(X)), X, X ** 2])

        def run_rank_adaptive_algo(self, **kw):
            F = self._feat(self.P[self.itr])
            self.coef = np.linalg.lstsq(F, self.v[self.itr], rcond=None)[0]
            print("ranks [1, 2, 1]")
            self.ranks = [1] + [2] * (self.d - 1) + [1]

        def cheb_tensor_evals(self, X):
            return self._feat(np.atleast_2d(X)) @ self.coef

    mod.MocaxExtend = MocaxExtend
    return mod
