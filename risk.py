# -*- coding: utf-8 -*-
"""Scenarios de risque, boite d'approximation, mesures VaR / ES.

UN seul jeu de chocs G ~ N(0, C_25), shape (N, 25), tire une fois (seed dediee)
et mis en cache. LE MEME G sert aux deux horizons :

    s_h^(k) = S0 * exp((r - sigma^2/2) h + sigma sqrt(h) G^(k))

Le groupe d lit le prefixe s_h^(k)[:d] (groupes emboites).
"""

import os

import numpy as np

from cache import ensure_dir, load_json, load_npy, save_json, save_npy
from config import fingerprint
from pricing import correlated_normals


# =========================================================================
#  Scenarios (caches)
# =========================================================================
def scenarios_key(cfg):
    return fingerprint(cfg.market, cfg.risk.n_scenarios, cfg.risk.seed_scenarios,
                       cfg.risk.horizons)


def scenarios_dir(cfg):
    return os.path.join(cfg.run.cache_dir, "scenarios", scenarios_key(cfg))


def get_scenarios(cfg, market, verbose=True):
    """-> (G, {h: S_h (N, 25)}). Genere une fois, relit ensuite."""
    root = scenarios_dir(cfg)
    hs = list(cfg.risk.horizons)
    files = [os.path.join(root, "G.npy")] + \
        [os.path.join(root, "S_%s.npy" % h) for h in hs]
    if all(os.path.exists(f) for f in files):
        if verbose:
            print("Scenarios charges du cache : %s" % root)
        G = load_npy(files[0])
        return G, {h: load_npy(os.path.join(root, "S_%s.npy" % h)) for h in hs}

    ensure_dir(root)
    rng = np.random.default_rng(cfg.risk.seed_scenarios)
    G = correlated_normals(rng, market.corr_matrix, cfg.risk.n_scenarios)
    S = {}
    for h, dt in cfg.risk.horizons.items():
        drift = (market.rate - 0.5 * market.vols ** 2) * dt
        S[h] = market.spots * np.exp(drift + market.vols * np.sqrt(dt) * G)
        save_npy(os.path.join(root, "S_%s.npy" % h), S[h])
    save_npy(os.path.join(root, "G.npy"), G)
    save_npy(os.path.join(root, "S0.npy"), market.spots.reshape(1, -1))
    save_json(os.path.join(root, "manifest.json"),
              dict(n=cfg.risk.n_scenarios, seed=cfg.risk.seed_scenarios,
                   horizons=cfg.risk.horizons, spots=market.spots,
                   vols=market.vols, rho=cfg.market.rho, rate=market.rate))
    if verbose:
        print("Scenarios generes et mis en cache : %s" % root)
    return G, S


# =========================================================================
#  Boite d'approximation : enveloppe empirique des scenarios, marge 1 %
# =========================================================================
def empirical_box(S_by_h, d, box_cfg):
    """(d, 2) : lo_i = 0.99 * min_{k,h} s_{h,i}^k ; hi_i = 1.01 * max_{k,h}."""
    allS = np.vstack([np.asarray(S)[:, :d] for S in S_by_h.values()])
    lo = box_cfg.factor_lo * allS.min(axis=0)
    hi = box_cfg.factor_hi * allS.max(axis=0)
    return np.stack([lo, hi], axis=1)


def inside_box(P, box, tol=1e-12):
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return np.all((P >= box[:, 0] - tol) & (P <= box[:, 1] + tol), axis=1)


# =========================================================================
#  Mesures de risque
# =========================================================================
def _lbl(a):
    s = ("%.1f" % (100 * a)).rstrip("0").rstrip(".")
    return s.replace(".", "_")


def measure_names(risk_cfg):
    return (["VaR_%s" % _lbl(a) for a in risk_cfg.var_levels]
            + ["ES_%s" % _lbl(a) for a in risk_cfg.es_levels])


def risk_measures(L, risk_cfg):
    """L = V0 - V_h (perte > 0 = baisse). VaR_a = quantile_a(L) ;
    ES_a = moyenne des pertes >= quantile_a(L) (estimateur empirique)."""
    L = np.asarray(L, dtype=float)
    out = dict(L_mean=float(L.mean()), L_std=float(L.std()),
               L_min=float(L.min()), L_max=float(L.max()))
    for a in risk_cfg.var_levels:
        out["VaR_%s" % _lbl(a)] = float(np.quantile(L, a))
    for a in risk_cfg.es_levels:
        q = np.quantile(L, a)
        tail = L[L >= q]
        out["ES_%s" % _lbl(a)] = float(tail.mean())
        out["n_tail_ES_%s" % _lbl(a)] = int(tail.size)
    return out


def price_errors(p_proxy, p_ref):
    """Erreurs du proxy vs full repricing sur les memes scenarios.

    erel_sc : erreur relative a ECHELLE GLOBALE (|ecart| / moyenne |p_ref|),
              robuste aux changements de signe de V_d -> la metrique a lire.
    erel    : ponctuelle, explose pres des zeros de V_d (indicatif seulement).
    """
    p = np.asarray(p_proxy, dtype=float)
    r = np.asarray(p_ref, dtype=float)
    eabs = np.abs(p - r)
    erel = eabs / np.maximum(np.abs(r), 1e-12)
    scale = max(float(np.mean(np.abs(r))), 1e-12)
    return dict(eabs_max=float(eabs.max()), eabs_mean=float(eabs.mean()),
                erel_sc_max=float(eabs.max() / scale),
                erel_sc_mean=float(eabs.mean() / scale),
                erel_max=float(erel.max()), erel_mean=float(erel.mean()),
                bias_mean=float((p - r).mean()), ref_scale=scale)
