# -*- coding: utf-8 -*-
"""Pricers Monte-Carlo.

Deux pricers, deux usages -- et ils ne se melangent pas :

  UnitPricer  : pricer "front office" d'UN instrument, regime noCRN.
                A CHAQUE appel : generateur re-cree avec LA MEME seed, tirage
                complet de M x d normales, Cholesky, trajectoires, payoff.
                -> cout d'un pricing FO complet ; c'est LUI qui fait le full
                repricing, V0, et l'etude de convergence.
                Il renvoie le prix UNITAIRE Pi_j (sans le nominal) ; la position
                alpha_j est appliquee a l'agregation.

  GroupPricer : pricer de la FONCTION CIBLE du proxy V_d(x) = sum_j alpha_j Pi_j(x)
                en regime CRN : paquet de normales fige a l'init, reutilise a
                tous les appels, les 20 payoffs du groupe sur les memes
                trajectoires. Sert UNIQUEMENT aux noeuds des proxies.

Remarque : GroupPricer(seed=s) et UnitPricer(seed=s) tirent exactement le meme
paquet (meme generateur, meme forme (M, d), meme Cholesky) : la valeur d'un
noeud = sum_j alpha_j * UnitPricer_j(noeud) (a l'arrondi flottant pres).
"""

import math
from typing import Dict

import numpy as np


def correlated_normals(rng, corr, n):
    """n tirages ~ N(0, corr), shape (n, d), via Cholesky."""
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0]))
    return z @ L.T


def _spot_vector(point, d):
    s0 = np.asarray([float(point[i]) for i in range(d)], dtype=float)
    if np.any(s0 <= 0):
        raise ValueError("Spot negatif ou nul dans le point evalue.")
    return s0


# =========================================================================
#  Pricer unitaire (front office), noCRN, meme seed a chaque appel
# =========================================================================
class UnitPricer(object):
    """Callable point -> Pi_j(point) (prix unitaire, SANS nominal)."""

    def __init__(self, derivative, market, n_paths, seed):
        self.dv = derivative
        sub = market.sub(derivative.underlying_ids)
        self.corr = np.asarray(sub.corr_matrix, dtype=float)
        self.vol = np.asarray(sub.vols, dtype=float)
        self.r = float(sub.rate)
        self.d = len(self.vol)
        self.M = int(n_paths)
        self.seed = int(seed)
        self.T = float(derivative.maturity)
        self.n_calls = 0

    def __call__(self, point, additional_data=None, seed=None):
        """seed=None -> la seed fixe du pricer (R1) ; sinon seed de CET appel (R2)."""
        self.n_calls += 1
        s0 = _spot_vector(point, self.d)
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        G = correlated_normals(rng, self.corr, self.M)   # tirage complet
        T = self.T
        ST = s0 * np.exp((self.r - 0.5 * self.vol ** 2) * T
                         + self.vol * math.sqrt(T) * G)
        return float(math.exp(-self.r * T) * self.dv.payoff(ST).mean())

    def __repr__(self):
        return ("UnitPricer(id=%d, d=%d, M=%d, seed=%d, noCRN/meme-seed)"
                % (self.dv.derivative_id, self.d, self.M, self.seed))


# =========================================================================
#  Pricer de groupe CRN (noeuds des proxies)
# =========================================================================
class GroupPricer(object):
    """Callable x -> V_d(x), CRN (paquet fige a l'init), signature MoCaX."""

    def __init__(self, derivatives, market, n_paths, seed):
        self.derivs = list(derivatives)
        if not self.derivs:
            raise ValueError("Aucun derive fourni.")
        ids = self.derivs[0].underlying_ids
        for dv in self.derivs:
            if dv.underlying_ids != ids:
                raise ValueError("Groupe non homogene (produit %d)." % dv.derivative_id)
        sub = market.sub(ids)
        self.r = float(sub.rate)
        self.vol = np.asarray(sub.vols, dtype=float)
        self.d = len(self.vol)
        self.M = int(n_paths)
        self.seed = int(seed)
        self.n_calls = 0
        self._maturities = sorted({float(dv.maturity) for dv in self.derivs})
        self._by_T = {T: [dv for dv in self.derivs if abs(dv.maturity - T) < 1e-12]
                      for T in self._maturities}
        self._G = correlated_normals(np.random.default_rng(self.seed),
                                     sub.corr_matrix, self.M)   # CRN : fige

    def _terms(self, point):
        s0 = _spot_vector(point, self.d)
        for T in self._maturities:
            ST = s0 * np.exp((self.r - 0.5 * self.vol ** 2) * T
                             + self.vol * math.sqrt(T) * self._G)
            disc = math.exp(-self.r * T)
            for dv in self._by_T[T]:
                yield dv, dv.nominal * disc * float(dv.payoff(ST).mean())

    def __call__(self, point, additional_data=None):
        self.n_calls += 1
        return float(sum(v for _, v in self._terms(point)))

    def price_breakdown(self, point) -> Dict[int, float]:
        return {dv.derivative_id: v for dv, v in self._terms(point)}

    def __repr__(self):
        return ("GroupPricer(d=%d, n_products=%d, M=%d, seed=%d, CRN)"
                % (self.d, len(self.derivs), self.M, self.seed))
