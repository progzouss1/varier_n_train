# -*- coding: utf-8 -*-
"""LE full repricing (unique) du portefeuille : pricing UNITAIRE noCRN.

Taches : (etape, produit) avec etape in {"s0", "1j", "10j"}.
  - chaque tache price UN instrument sur tous les points de l'etape
    (1 point pour s0, N scenarios pour un horizon) ;
  - chaque pricing = UnitPricer (generateur re-cree, MEME seed) ;
  - le temps de CHAQUE pricing est mesure et stocke ;
  - checkpoint tous les `checkpoint_every` pricings (reprise exacte) ;
  - une tache terminee n'est JAMAIS recalculee (cache).

Parallelisme optionnel (n_workers > 1) : processus independants, une tache par
processus. Les temps par pricing restent mesures dans chaque processus ; le
temps "sequentiel equivalent" = somme des temps de pricing, le temps mur est
stocke a part.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import pandas as pd

from cache import (ensure_dir, load_json, load_npy, save_json, save_npy,
                   save_npz, stamp)
from config import fingerprint
from model import OPTION_TYPES
from pricing import UnitPricer
from risk import get_scenarios, scenarios_dir

PRICER_VERSION = "unit-noCRN-sameseed-v1"


# =========================================================================
#  Emplacements
# =========================================================================
def fr_key(cfg):
    return fingerprint(cfg.market, cfg.portfolio, cfg.risk.n_scenarios,
                       cfg.risk.seed_scenarios, cfg.risk.horizons,
                       cfg.mc.n_paths, cfg.mc.seed_pricing, PRICER_VERSION)


def fr_dir(cfg):
    return os.path.join(cfg.run.cache_dir, "full_repricing", fr_key(cfg))


def stages(cfg):
    return ["s0"] + list(cfg.risk.horizons)


def _paths(root, stage, pid):
    base = os.path.join(root, stage, "p%03d" % pid)
    return dict(values=base + ".npy", times=base + "_times.npy",
                meta=base + ".json", partial=base + ".partial.npz")


def _write_manifest(cfg, market, pf):
    root = ensure_dir(fr_dir(cfg))
    fn = os.path.join(root, "manifest.json")
    if not os.path.exists(fn):
        save_json(fn, dict(
            pricer=PRICER_VERSION, n_paths=cfg.mc.n_paths,
            seed_pricing=cfg.mc.seed_pricing,
            n_scenarios=cfg.risk.n_scenarios, horizons=cfg.risk.horizons,
            scenarios_dir=scenarios_dir(cfg),
            products=[dict(id=dv.derivative_id, d=dv.dimension,
                           type=OPTION_TYPES[dv.option_type],
                           callput=dv.callput_flag, strike=dv.strike,
                           maturity=dv.maturity, nominal=dv.nominal)
                      for dv in sorted(pf, key=lambda x: x.derivative_id)],
            created=stamp()))
    return root


# =========================================================================
#  Une tache (execute dans le processus courant OU un worker)
# =========================================================================
def _limit_threads():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _price_task(dv, market, n_paths, seed, S_file, stage, root,
                checkpoint_every, progress_every=0, limit_threads=False,
                label=""):
    if limit_threads:
        _limit_threads()
    p = _paths(root, stage, dv.derivative_id)
    if os.path.exists(p["meta"]) and os.path.exists(p["values"]):
        return load_json(p["meta"])

    S = load_npy(S_file, mmap=True)
    d = dv.dimension
    n = S.shape[0]
    vals = np.full(n, np.nan)
    tms = np.full(n, np.nan)
    k0 = 0
    if os.path.exists(p["partial"]):                     # reprise
        z = np.load(p["partial"])
        k0 = int(z["n_done"])
        vals[:k0], tms[:k0] = z["values"][:k0], z["times"][:k0]

    pricer = UnitPricer(dv, market, n_paths, seed)
    t_wall0 = perf_counter()
    for k in range(k0, n):
        pt = np.asarray(S[k, :d], dtype=float)
        t0 = perf_counter()
        vals[k] = pricer(pt)
        tms[k] = perf_counter() - t0
        done = k + 1
        if checkpoint_every and done % checkpoint_every == 0 and done < n:
            save_npz(p["partial"], values=vals, times=tms, n_done=done)
        if progress_every and (done % progress_every == 0 or done == n):
            el = np.nansum(tms)
            print("  %s %-4s p%03d (d=%2d) : %6d/%d  %.3fs/pricing  ETA tache %.0fs   "
                  % (label, stage, dv.derivative_id, d, done, n,
                     el / done, el / done * (n - done)), end="\r", flush=True)
    wall = perf_counter() - t_wall0

    meta = dict(id=dv.derivative_id, d=d, stage=stage, n=n, n_paths=n_paths,
                seed=seed, t_pricing_sum=float(np.sum(tms)),
                t_pricing_mean=float(np.mean(tms)),
                t_pricing_median=float(np.median(tms)),
                t_pricing_max=float(np.max(tms)),
                t_wall_last_session=float(wall), resumed_from=k0, **stamp())
    save_npy(p["values"], vals)
    save_npy(p["times"], tms)
    save_json(p["meta"], meta)
    if os.path.exists(p["partial"]):
        os.remove(p["partial"])
    return meta


# =========================================================================
#  Orchestration
# =========================================================================
def run_full_repricing(cfg, market, pf, stages_to_run=None, product_ids=None,
                       n_workers=None, verbose=True):
    """Lance (ou complete) LE full repricing. Idempotent et reprenable."""
    _write_manifest(cfg, market, pf)
    root = fr_dir(cfg)
    get_scenarios(cfg, market, verbose=verbose)          # assure le cache
    sdir = scenarios_dir(cfg)
    s_files = {"s0": os.path.join(sdir, "S0.npy")}
    s_files.update({h: os.path.join(sdir, "S_%s.npy" % h) for h in cfg.risk.horizons})

    stg = list(stages_to_run or stages(cfg))
    dvs = sorted(pf, key=lambda x: x.derivative_id)
    if product_ids is not None:
        dvs = [dv for dv in dvs if dv.derivative_id in set(product_ids)]
    todo = [(s, dv) for s in stg for dv in dvs
            if not os.path.exists(_paths(root, s, dv.derivative_id)["meta"])]
    n_all = len(stg) * len(dvs)
    if verbose:
        print("Full repricing : %d taches (etape x produit), %d deja en cache, "
              "%d a calculer -> %s" % (n_all, n_all - len(todo), len(todo), root))
    if not todo:
        return status(cfg, pf)

    nw = int(n_workers or cfg.run.n_workers or 1)
    t0 = time.time()
    if nw <= 1:
        for i, (s, dv) in enumerate(todo):
            _price_task(dv, market, cfg.mc.n_paths, cfg.mc.seed_pricing,
                        s_files[s], s, root, cfg.run.checkpoint_every,
                        progress_every=cfg.run.progress_every if verbose else 0,
                        label="[%d/%d]" % (i + 1, len(todo)))
        if verbose:
            print()
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_price_task, dv, market, cfg.mc.n_paths,
                              cfg.mc.seed_pricing, s_files[s], s, root,
                              cfg.run.checkpoint_every, 0, True): (s, dv)
                    for s, dv in todo}
            for i, f in enumerate(as_completed(futs)):
                s, dv = futs[f]
                m = f.result()
                if verbose:
                    el = time.time() - t0
                    print("  [%d/%d] %-4s p%03d (d=%2d) : %.0fs de pricing | "
                          "mur ecoule %.0fs, ETA %.0fs"
                          % (i + 1, len(todo), s, dv.derivative_id, dv.dimension,
                             m["t_pricing_sum"], el,
                             el / (i + 1) * (len(todo) - i - 1)), flush=True)
    if verbose:
        print("Full repricing : session terminee en %.0f s (mur)." % (time.time() - t0))
    return status(cfg, pf)


def status(cfg, pf):
    """DataFrame etape x produit : 'ok', 'partiel k/N' ou '-'."""
    root = fr_dir(cfg)
    rows = []
    for dv in sorted(pf, key=lambda x: x.derivative_id):
        r = dict(id=dv.derivative_id, d=dv.dimension)
        for s in stages(cfg):
            p = _paths(root, s, dv.derivative_id)
            if os.path.exists(p["meta"]):
                r[s] = "ok"
            elif os.path.exists(p["partial"]):
                r[s] = "partiel %d" % int(np.load(p["partial"])["n_done"])
            else:
                r[s] = "-"
        rows.append(r)
    return pd.DataFrame(rows).set_index("id")


def is_complete(cfg, pf):
    st = status(cfg, pf)
    return bool((st[stages(cfg)] == "ok").all().all())


# =========================================================================
#  Estimation du cout (avant de lancer) -- cachee
# =========================================================================
def benchmark(cfg, market, pf, n_probe=3, verbose=True):
    """Temps d'UN pricing unitaire par produit (mediane de n_probe) + projection."""
    root = ensure_dir(fr_dir(cfg))
    fn = os.path.join(root, "benchmark_n%d.json" % n_probe)
    if os.path.exists(fn):
        b = load_json(fn)
        if verbose:
            print("Benchmark charge du cache (%s, %s)" % (b["timestamp"], b["host"]))
    else:
        per = {}
        for dv in sorted(pf, key=lambda x: x.derivative_id):
            pr = UnitPricer(dv, market, cfg.mc.n_paths, cfg.mc.seed_pricing)
            ts = []
            for _ in range(n_probe):
                t0 = perf_counter()
                pr(market.spots[:dv.dimension])
                ts.append(perf_counter() - t0)
            per[dv.derivative_id] = float(np.median(ts))
        n_eval = cfg.risk.n_scenarios * len(cfg.risk.horizons) + 1
        b = dict(n_probe=n_probe, per_product=per, n_paths=cfg.mc.n_paths,
                 t_one_portfolio_repricing=float(sum(per.values())),
                 n_evals_per_product=n_eval,
                 t_projected_total=float(sum(per.values()) * n_eval), **stamp())
        save_json(fn, b)
    if verbose:
        print("1 repricing complet du portefeuille (100 pricings unitaires, M=%s) "
              ": %.2f s" % ("{:,}".format(cfg.mc.n_paths), b["t_one_portfolio_repricing"]))
        print("Projection full repricing (%d pricings / produit) : %.0f s = %.1f h "
              "(sequentiel)" % (b["n_evals_per_product"], b["t_projected_total"],
                                b["t_projected_total"] / 3600))
    return b


# =========================================================================
#  Lecture des resultats
# =========================================================================
class FullRepricing(object):
    """Resultats caches du full repricing, agreges a la demande."""

    def __init__(self, cfg, pf):
        root = fr_dir(cfg)
        dvs = sorted(pf, key=lambda x: x.derivative_id)
        self.cfg = cfg
        self.ids = np.array([dv.derivative_id for dv in dvs])
        self.dims = np.array([dv.dimension for dv in dvs])
        self.nominals = np.array([dv.nominal for dv in dvs], dtype=float)
        self.products = dvs
        self.prices, self.times, self.meta = {}, {}, {}
        missing = []
        for s in stages(cfg):
            cols_v, cols_t, metas = [], [], []
            for dv in dvs:
                p = _paths(root, s, dv.derivative_id)
                if not os.path.exists(p["meta"]):
                    missing.append((s, dv.derivative_id))
                    continue
                cols_v.append(load_npy(p["values"]))
                cols_t.append(load_npy(p["times"]))
                metas.append(load_json(p["meta"]))
            if cols_v and len(cols_v) == len(dvs):
                self.prices[s] = np.column_stack(cols_v)     # (N, P) unitaires
                self.times[s] = np.column_stack(cols_t)      # (N, P) secondes
                self.meta[s] = metas
        if missing:
            raise RuntimeError(
                "Full repricing incomplet : %d taches manquantes (ex. %s). "
                "Lancer run_full_repricing (ou run_all.py --stage fr)."
                % (len(missing), missing[:5]))
        self.root = root

    # --------------------------------------------------------- valeurs
    def _mask(self, d=None):
        return np.ones_like(self.ids, bool) if d is None else (self.dims == d)

    def values(self, stage, d=None):
        """V_d(s) = sum_{j in G_d} alpha_j Pi_j(s) ; d=None -> portefeuille."""
        m = self._mask(d)
        return self.prices[stage][:, m] @ self.nominals[m]

    def V0(self, d=None):
        return float(self.values("s0", d)[0])

    # ----------------------------------------------------------- temps
    def time_sum(self, stage, d=None):
        """Somme des temps MESURES des pricings unitaires (sequentiel equivalent)."""
        return float(self.times[stage][:, self._mask(d)].sum())

    def time_per_scenario(self, stage, d=None):
        """Temps moyen pour repricer le perimetre sur UN scenario."""
        return self.time_sum(stage, d) / self.times[stage].shape[0]

    def product_table(self):
        rows = []
        for j, dv in enumerate(self.products):
            r = dict(id=dv.derivative_id, d=dv.dimension,
                     type=OPTION_TYPES[dv.option_type],
                     sens="Call" if dv.callput_flag == 0 else "Put",
                     K=dv.strike, T=dv.maturity, alpha=dv.nominal,
                     prix_unitaire_s0=self.prices["s0"][0, j],
                     contribution_s0=dv.nominal * self.prices["s0"][0, j])
            for s in stages(self.cfg):
                r["t_%s_s" % s] = float(self.times[s][:, j].sum())
            r["t_1_pricing_moy_s"] = float(np.mean(np.concatenate(
                [self.times[s][:, j] for s in stages(self.cfg)])))
            rows.append(r)
        return pd.DataFrame(rows).set_index("id")


# =========================================================================
#  Etude de convergence de V(s0) en M (pricer unitaire, meme seed) -- cachee
# =========================================================================
def convergence_study(cfg, market, pf, M_list, verbose=True):
    """V(s0) = sum_j alpha_j Pi_j(s0) pour chaque M ; resultats + temps caches."""
    fn = os.path.join(cfg.run.cache_dir, "convergence",
                      fingerprint(cfg.market, cfg.portfolio, cfg.mc.seed_pricing,
                                  PRICER_VERSION) + ".json")
    res = load_json(fn) if os.path.exists(fn) else {}
    for M in M_list:
        key = str(int(M))
        if key in res:
            if verbose:
                print("M=%9s  V=%.4f  (%.1fs, cache)" % ("{:,}".format(M),
                      res[key]["V"], res[key]["t"]))
            continue
        t0 = perf_counter()
        by_d = {}
        for dv in pf:
            v = dv.nominal * UnitPricer(dv, market, M, cfg.mc.seed_pricing)(
                market.spots[:dv.dimension])
            by_d[dv.dimension] = by_d.get(dv.dimension, 0.0) + v
        res[key] = dict(V=float(sum(by_d.values())), V_by_d=by_d,
                        t=perf_counter() - t0, **stamp())
        save_json(fn, res)
        if verbose:
            print("M=%9s  V=%.4f  (%.1fs)" % ("{:,}".format(M), res[key]["V"], res[key]["t"]))
    rows = [dict(M=int(k), V=v["V"], t_s=v["t"],
                 **{"V_d%d" % int(d): x for d, x in v["V_by_d"].items()})
            for k, v in res.items() if int(k) in set(int(m) for m in M_list)]
    return pd.DataFrame(rows).set_index("M").sort_index()
