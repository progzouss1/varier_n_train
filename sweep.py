# -*- coding: utf-8 -*-
"""Balayage n_train des proxies Tensor-Train, UN proxy par (groupe, horizon).

Deux regimes de pricing des noeuds, TOUS DEUX noCRN et UNITAIRES (pricer front
office : 20 pricings par noeud, un par instrument, sommes avec les nominaux) :

  R1 : noCRN MEME seed     -- generateur re-cree avec la seed fixe a chaque
                              pricing (comme le full repricing de reference).
  R2 : noCRN seed VARIABLE -- chaque pricing (noeud x produit) recoit sa propre
                              seed : c'est un pricer FO dont on ne controle pas
                              la seed. Les seeds sont derivees de
                              (d, horizon, n_train, noeud, produit) : differentes
                              d'un pricing a l'autre, mais identiques d'une
                              execution a l'autre (sinon le cache n'aurait pas
                              de sens).

Pour un meme (d, horizon, n_train), R1 et R2 utilisent la MEME sous-grille de
points : seule la seed des pricings differe. Chaque TT price SES noeuds (aucune
reutilisation entre TT).

Boite du proxy (d, h) = enveloppe des scenarios de l'horizon h SEUL, marge 1 %.

Cache : cache/sweep/
  subgrids/d10_1j_n01500_<fp>/   nodes.npy, subgrid.json      (partage R1/R2)
  R1/d10_1j_n01500_<fp>/         chunks/*.npz (checkpoints du pricing des noeuds)
                                 node_unit_prices.npy (n, 20), node_unit_times.npy,
                                 node_seeds.npy, node_values.npy, node_times.npy,
                                 nodes.json, fit.json, fit_log.txt,
                                 eval_s0.npy, eval_<h>.npy
  R2/...                         idem
"""

import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from cache import ensure_dir, load_json, load_npy, save_json, save_npy, save_npz, stamp
from config import TTConfig, fingerprint
from full_repricing import fr_key
from pricing import UnitPricer
from risk import empirical_box, inside_box, measure_names, price_errors, risk_measures

SWEEP_VERSION = "v1"
REGIMES = ("R1", "R2")
REGIME_LABEL = {"R1": "noCRN meme seed", "R2": "noCRN seed variable (pricer FO)"}


# =========================================================================
#  Configuration
# =========================================================================
def _default_grid():
    return {5: (300, 600, 900, 1200, 1500),
            10: (300, 600, 900, 1200, 1500),
            15: (500, 1000, 1500, 2000, 2500),
            20: (700, 1400, 2100, 2800, 3500),
            25: (1000, 2000, 3000, 4000, 5000)}


def _default_tt():
    # hyperparametres FIXES (seul n_train varie) ; n_train ici est ignore
    return {5: TTConfig(max_rank=10), 10: TTConfig(max_rank=10),
            15: TTConfig(max_rank=10), 20: TTConfig(max_rank=8),
            25: TTConfig(max_rank=8)}


@dataclass
class SweepConfig:
    n_train: Dict[int, Tuple[int, ...]] = field(default_factory=_default_grid)
    tt: Dict[int, TTConfig] = field(default_factory=_default_tt)
    seed_r2_base: int = 20261002          # racine des seeds R2
    chunk_size: int = 100                 # noeuds par checkpoint / tache parallele
    n_workers: int = 1                    # processus pour le pricing des noeuds
    fit_timeout: Optional[float] = 7200.0  # s ; None = pas de plafond (fit in-process)

    def tt_for(self, d, n):
        return replace(self.tt[d], n_train=int(n))

    @classmethod
    def smoke(cls):
        return cls(n_train={5: (40, 80), 10: (60, 120), 15: (80, 160),
                            20: (100, 200), 25: (120, 240)},
                   tt={d: TTConfig(degree=3, max_rank=4) for d in (5, 10, 15, 20, 25)},
                   chunk_size=25, fit_timeout=120.0)


# =========================================================================
#  Emplacements
# =========================================================================
def _sweep_root(cfg):
    return os.path.join(cfg.run.cache_dir, "sweep")


def subgrid_key(d, h, tt, box):
    return fingerprint(d, h, tt.degree, tt.n_train, tt.seed_subgrid,
                       np.round(np.asarray(box), 10).tolist(), SWEEP_VERSION)


def subgrid_dir(cfg, d, h, tt, box):
    return os.path.join(_sweep_root(cfg), "subgrids", "d%02d_%s_n%05d_%s"
                        % (d, h, tt.n_train, subgrid_key(d, h, tt, box)))


def tt_dir(cfg, sw, regime, d, h, tt, box):
    key = fingerprint(fr_key(cfg), subgrid_key(d, h, tt, box), regime, tt,
                      sw.seed_r2_base if regime == "R2" else None, SWEEP_VERSION)
    return os.path.join(_sweep_root(cfg), regime, "d%02d_%s_n%05d_%s"
                        % (d, h, tt.n_train, key))


def horizon_box(cfg, S, d, h):
    """Boite du proxy (d, h) : enveloppe des scenarios de l'horizon h SEUL."""
    return empirical_box({h: S[h]}, d, cfg.box)


# =========================================================================
#  Seeds R2
# =========================================================================
def r2_seed(base, d, h_index, n_train, k, pid):
    """Seed propre au pricing (noeud k, produit pid) du TT (d, h, n_train)."""
    ss = np.random.SeedSequence([int(base), int(d), int(h_index), int(n_train),
                                 int(k), int(pid)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


# =========================================================================
#  1) Sous-grille (partagee R1 / R2)
# =========================================================================
def get_subgrid(cfg, me, d, h, tt, box, verbose=True):
    root = subgrid_dir(cfg, d, h, tt, box)
    fn = os.path.join(root, "nodes.npy")
    if os.path.exists(fn):
        return load_npy(fn), load_json(os.path.join(root, "subgrid.json"))
    if me is None:
        raise RuntimeError("Sous-grille d=%d %s n=%d absente du cache et MoCaXExtend "
                           "non charge." % (d, h, tt.n_train))
    from mocax_tt import make_subgrid
    pts, info = make_subgrid(me, box, tt)
    info.update(d=d, horizon=h, box=box, **stamp())
    save_npy(fn, pts)
    save_json(os.path.join(root, "subgrid.json"), info)
    if verbose:
        print("    sous-grille : %d points (%d uniques), grille pleine %.3g"
              % (info["n_train"], info["n_train_unique"], float(info["tensor_size"])))
    return pts, info


# =========================================================================
#  2) Pricing des noeuds : unitaire, noCRN, R1 / R2 -- par paquets caches
# =========================================================================
def _limit_threads():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _price_chunk(dvs, market, n_paths, seed_fixed, pts, k0, regime, seed_ctx,
                 out_file, limit_threads=False):
    if limit_threads:
        _limit_threads()
    pricers = [UnitPricer(dv, market, n_paths, seed_fixed) for dv in dvs]
    n, P = len(pts), len(dvs)
    vals, tms = np.empty((n, P)), np.empty((n, P))
    seeds = np.zeros((n, P), dtype=np.uint64)
    for i in range(n):
        pt = pts[i]
        for j, (dv, pr) in enumerate(zip(dvs, pricers)):
            s = seed_fixed if regime == "R1" else r2_seed(*seed_ctx, k0 + i, dv.derivative_id)
            t0 = perf_counter()
            vals[i, j] = pr(pt, seed=s)
            tms[i, j] = perf_counter() - t0
            seeds[i, j] = s
    save_npz(out_file, values=vals, times=tms, seeds=seeds)
    return out_file


def price_nodes(cfg, sw, market, pf, d, regime, h_index, tt, pts, root, verbose=True):
    """-> (V_d aux noeuds, temps par noeud). Paquets deja calcules relus."""
    fv = os.path.join(root, "node_values.npy")
    if os.path.exists(fv):
        return load_npy(fv), load_npy(os.path.join(root, "node_times.npy"))
    dvs = pf.group(d)
    cdir = ensure_dir(os.path.join(root, "chunks"))
    cs = int(sw.chunk_size)
    starts = list(range(0, len(pts), cs))
    files = {k0: os.path.join(cdir, "c%06d.npz" % k0) for k0 in starts}
    todo = [k0 for k0 in starts if not os.path.exists(files[k0])]
    ctx = (sw.seed_r2_base, d, h_index, tt.n_train)
    args = lambda k0, lim: (dvs, market, cfg.mc.n_paths, cfg.mc.seed_pricing,
                            np.asarray(pts[k0:k0 + cs], dtype=float), k0, regime,
                            ctx, files[k0], lim)
    t_wall = perf_counter()
    if todo and verbose:
        print("    noeuds %s : %d noeuds x %d produits, %d/%d paquets a calculer"
              % (regime, len(pts), len(dvs), len(todo), len(starts)))
    if sw.n_workers <= 1:
        for i, k0 in enumerate(todo):
            _price_chunk(*args(k0, False))
            if verbose:
                el = perf_counter() - t_wall
                print("      paquet %d/%d  %.0fs  ETA %.0fs   " % (
                    i + 1, len(todo), el, el / (i + 1) * (len(todo) - i - 1)),
                    end="\r", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=sw.n_workers) as ex:
            futs = [ex.submit(_price_chunk, *args(k0, True)) for k0 in todo]
            for i, f in enumerate(as_completed(futs)):
                f.result()
                if verbose:
                    el = perf_counter() - t_wall
                    print("      paquet %d/%d  mur %.0fs  ETA %.0fs   " % (
                        i + 1, len(todo), el, el / (i + 1) * (len(todo) - i - 1)),
                        end="\r", flush=True)
    if todo and verbose:
        print()
    Z = [np.load(files[k0]) for k0 in starts]
    U = np.vstack([z["values"] for z in Z])
    T = np.vstack([z["times"] for z in Z])
    SD = np.vstack([z["seeds"] for z in Z])
    nominals = np.array([dv.nominal for dv in dvs], dtype=float)
    vals, tms = U @ nominals, T.sum(axis=1)
    save_npy(os.path.join(root, "node_unit_prices.npy"), U)
    save_npy(os.path.join(root, "node_unit_times.npy"), T)
    save_npy(os.path.join(root, "node_seeds.npy"), SD)
    save_npy(os.path.join(root, "node_times.npy"), tms)
    save_npy(fv, vals)
    save_json(os.path.join(root, "nodes.json"), dict(
        regime=regime, n_nodes=len(pts), n_products=len(dvs),
        n_paths=cfg.mc.n_paths, t_nodes=float(tms.sum()),
        t_node_mean=float(tms.mean()), t_unit_pricing_mean=float(T.mean()),
        n_distinct_seeds=int(len(np.unique(SD))), n_workers=sw.n_workers,
        t_wall_last_session=perf_counter() - t_wall, **stamp()))
    return vals, tms


# =========================================================================
#  3) Fit + evaluation (sous-processus avec plafond de temps)
# =========================================================================
def _fit_eval(me, box, tt, pts, vals, eval_points, dvs, market, n_paths, seed, root):
    """Fit ALS puis evaluation ; ecrit fit.json, fit_log.txt, eval_*.npy."""
    from mocax_tt import fit_tt
    out = dict(**stamp())
    try:
        obj, eval_fn, times, info = fit_tt(me, box, tt, pts, vals, verbose=False)
    except Exception as exc:
        out.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   diag=getattr(exc, "diag", None), traceback=traceback.format_exc())
        save_json(os.path.join(root, "fit.json"), out)
        return
    with open(os.path.join(root, "fit_log.txt"), "w") as f:
        f.write(info.pop("fit_log") or "")
    pricers = [UnitPricer(dv, market, n_paths, seed) for dv in dvs]
    t_eval, n_out = {}, {}
    for s, P in eval_points.items():
        P = np.atleast_2d(P)
        inside = inside_box(P, box)
        res = np.full(len(P), np.nan)
        t0 = perf_counter()
        if inside.any():
            res[inside] = eval_fn(P[inside])
        for k in np.where(~inside)[0]:          # jamais d'extrapolation
            res[k] = sum(dv.nominal * pr(P[k]) for dv, pr in zip(dvs, pricers))
        t_eval[s] = perf_counter() - t0
        n_out[s] = int((~inside).sum())
        save_npy(os.path.join(root, "eval_%s.npy" % s), res)
    out.update(status="ok", t_fit=float(times["t_extension"]), t_eval=t_eval,
               n_outside=n_out, info=info,
               n_rank_rounds=len(info.get("rank_trajectory") or []))
    save_json(os.path.join(root, "fit.json"), out)


def fit_and_eval(sw, me, box, tt, pts, vals, eval_points, dvs, market, cfg, root,
                 verbose=True):
    fj = os.path.join(root, "fit.json")
    a = (me, box, tt, pts, vals, eval_points, dvs, market, cfg.mc.n_paths,
         cfg.mc.seed_pricing, root)
    t0 = perf_counter()
    if sw.fit_timeout is None:
        _fit_eval(*a)
    else:
        p = mp.get_context("fork").Process(target=_fit_eval, args=a)
        p.start()
        p.join(sw.fit_timeout)
        if p.is_alive():
            p.terminate()
            p.join(10)
            if p.is_alive():
                p.kill()
            save_json(fj, dict(status="timeout", t_fit=float(sw.fit_timeout),
                               error="fit ALS arrete au plafond de %.0f s" % sw.fit_timeout,
                               **stamp()))
        elif not os.path.exists(fj):
            save_json(fj, dict(status="crashed", t_fit=perf_counter() - t0,
                               error="processus du fit termine (code %s) sans resultat"
                               % p.exitcode, **stamp()))
    f = load_json(fj)
    if verbose:
        if f["status"] == "ok":
            i = f["info"]
            print("    fit : %.1fs, rangs %s (max %s), val_err %.3e, %d tours de rang"
                  % (f["t_fit"], i.get("ranks"), i.get("rank_max"), i.get("val_err"),
                     f["n_rank_rounds"]))
        else:
            print("    fit : %s -- %s" % (f["status"].upper(), f.get("error")))
    return f


# =========================================================================
#  Orchestration
# =========================================================================
def run_one(cfg, sw, market, pf, S, me, regime, d, h, n, verbose=True,
            retry_failed=False):
    hs = list(cfg.risk.horizons)
    tt = sw.tt_for(d, n)
    box = horizon_box(cfg, S, d, h)
    root = ensure_dir(tt_dir(cfg, sw, regime, d, h, tt, box))
    fj = os.path.join(root, "fit.json")
    if os.path.exists(fj):
        st = load_json(fj)["status"]
        if st == "ok" or not retry_failed:
            if verbose:
                print("  %s d=%-2d %-3s n=%-5d : cache (%s)" % (regime, d, h, n, st))
            return root
        os.remove(fj)
    if verbose:
        print("  %s d=%-2d %-3s n=%-5d :" % (regime, d, h, n))
    pts, _ = get_subgrid(cfg, me, d, h, tt, box, verbose)
    vals, tms = price_nodes(cfg, sw, market, pf, d, regime, hs.index(h), tt, pts,
                            root, verbose)
    if verbose:
        print("    noeuds : %.0fs (%.3fs/noeud)" % (tms.sum(), tms.mean()))
    if me is None:
        raise RuntimeError("MoCaXExtend non charge : fit impossible.")
    from mocax_tt import patch_mocax
    patch_mocax(me)
    eval_points = {"s0": market.spots[:d].reshape(1, -1), h: np.asarray(S[h])[:, :d]}
    fit_and_eval(sw, me, box, tt, pts, vals, eval_points, pf.group(d), market, cfg,
                 root, verbose)
    return root


def run_sweep(cfg, sw, market, pf, S, me, regime, groups=None, horizons=None,
              levels=None, verbose=True, retry_failed=False):
    """Lance (ou complete) le balayage d'UN regime. Idempotent, reprenable."""
    assert regime in REGIMES
    groups = list(groups or cfg.portfolio.dimensions)
    horizons = list(horizons or cfg.risk.horizons)
    t0 = perf_counter()
    for d in groups:
        grid = sw.n_train[d]
        idx = range(len(grid)) if levels is None else levels
        for h in horizons:
            for i in idx:
                run_one(cfg, sw, market, pf, S, me, regime, d, h, grid[i],
                        verbose, retry_failed)
    if verbose:
        print("Balayage %s : session terminee en %.0fs" % (regime, perf_counter() - t0))


# =========================================================================
#  Lecture des resultats
# =========================================================================
def load_tt(cfg, sw, S, regime, d, h, n):
    """dict brut d'un TT (ou None s'il n'a pas encore ete lance)."""
    tt = sw.tt_for(d, n)
    box = horizon_box(cfg, S, d, h)
    root = tt_dir(cfg, sw, regime, d, h, tt, box)
    fj = os.path.join(root, "fit.json")
    if not os.path.exists(fj):
        return None
    r = dict(root=root, fit=load_json(fj), box=box,
             nodes=load_json(os.path.join(root, "nodes.json")),
             node_values=load_npy(os.path.join(root, "node_values.npy")),
             node_times=load_npy(os.path.join(root, "node_times.npy")))
    if r["fit"]["status"] == "ok":
        r["V0"] = float(load_npy(os.path.join(root, "eval_s0.npy"))[0])
        r["V"] = load_npy(os.path.join(root, "eval_%s.npy" % h))
    return r


def _noise_vs_r1(cfg, sw, S, d, h, n, vals_r2):
    r1 = load_tt(cfg, sw, S, "R1", d, h, n)
    if r1 is None:
        return {}
    diff = vals_r2 - r1["node_values"]
    sc = max(float(np.mean(np.abs(r1["node_values"]))), 1e-12)
    return dict(bruit_noeuds_std=float(diff.std()),
                bruit_noeuds_max=float(np.abs(diff).max()),
                bruit_noeuds_rel_std=float(diff.std() / sc))


def sweep_table(cfg, sw, fr, S, regime, groups=None):
    """Une ligne par TT (d, horizon, n_train) : temps, rangs, VaR/ES vs FR."""
    rows = []
    for d in (groups or cfg.portfolio.dimensions):
        for h in cfg.risk.horizons:
            t_fr_h, t_fr_0 = fr.time_sum(h, d), fr.time_sum("s0", d)
            for n in sw.n_train[d]:
                r = load_tt(cfg, sw, S, regime, d, h, n)
                if r is None:
                    continue
                f, nd = r["fit"], r["nodes"]
                row = dict(regime=regime, d=d, horizon=h, n_train=n,
                           statut=f["status"], erreur=f.get("error"),
                           n_noeuds=nd["n_nodes"], t_nodes_s=nd["t_nodes"],
                           t_noeud_moy_s=nd["t_node_mean"],
                           t_pricing_unitaire_moy_s=nd["t_unit_pricing_mean"],
                           n_seeds_distinctes=nd["n_distinct_seeds"],
                           t_fit_s=f.get("t_fit"))
                if regime == "R2":
                    row.update(_noise_vs_r1(cfg, sw, S, d, h, n, r["node_values"]))
                if f["status"] == "ok":
                    i = f["info"]
                    row.update(rangs=i.get("ranks"), rang_max=i.get("rank_max"),
                               sature=i.get("saturated"), val_err=i.get("val_err"),
                               n_tours_rang=f.get("n_rank_rounds"),
                               trajectoire_rangs=i.get("rank_trajectory"),
                               n_hors_boite=sum(f["n_outside"].values()))
                    t_ev = f["t_eval"][h]
                    t_build = nd["t_nodes"] + f["t_fit"]
                    row.update(t_eval_s=t_ev, t_build_s=t_build, t_FR_s=t_fr_h,
                               speedup_online=t_fr_h / max(t_ev, 1e-12),
                               speedup_total=(t_fr_h + t_fr_0) / max(
                                   t_build + t_ev + f["t_eval"]["s0"], 1e-12))
                    V0f = fr.V0(d)
                    Vf = fr.values(h, d)
                    mf = risk_measures(V0f - Vf, cfg.risk)
                    mp_ = risk_measures(r["V0"] - r["V"], cfg.risk)
                    row.update(V0_FR=V0f, V0_proxy=r["V0"])
                    for k in measure_names(cfg.risk):
                        row["%s_FR" % k] = mf[k]
                        row["%s_proxy" % k] = mp_[k]
                        row["d%s_pct" % k] = 100 * (mp_[k] - mf[k]) / abs(mf[k])
                    e = price_errors(r["V"], Vf)
                    row.update({k: e[k] for k in ("erel_sc_mean", "erel_sc_max",
                                                  "eabs_max", "bias_mean")})
                rows.append(row)
    return pd.DataFrame(rows)


def portfolio_table(cfg, sw, fr, S, regime):
    """Portefeuille = somme des 5 TT du niveau i de la grille (i-eme n_train de
    chaque groupe). Un TT manquant / en echec -> full repricing du groupe, SIGNALE."""
    dims = list(cfg.portfolio.dimensions)
    n_lv = min(len(sw.n_train[d]) for d in dims)
    rows = []
    for h in cfg.risk.horizons:
        for i in range(n_lv):
            V0, V = 0.0, np.zeros(cfg.risk.n_scenarios)
            t_nodes = t_fit = t_ev = 0.0
            repli = []
            for d in dims:
                r = load_tt(cfg, sw, S, regime, d, h, sw.n_train[d][i])
                if r is None or r["fit"]["status"] != "ok":
                    repli.append(d)
                    V0 += fr.V0(d)
                    V = V + fr.values(h, d)
                    t_ev += fr.time_sum(h, d) + fr.time_sum("s0", d)
                    continue
                V0 += r["V0"]
                V = V + r["V"]
                t_nodes += r["nodes"]["t_nodes"]
                t_fit += r["fit"]["t_fit"]
                t_ev += r["fit"]["t_eval"][h] + r["fit"]["t_eval"]["s0"]
            t_fr = fr.time_sum(h) + fr.time_sum("s0")
            mf = risk_measures(fr.V0() - fr.values(h), cfg.risk)
            mp_ = risk_measures(V0 - V, cfg.risk)
            row = dict(regime=regime, horizon=h, niveau=i + 1,
                       n_train={d: sw.n_train[d][i] for d in dims},
                       groupes_en_repli_FR=repli, V0_FR=fr.V0(), V0_proxy=V0,
                       t_nodes_s=t_nodes, t_fit_s=t_fit, t_eval_s=t_ev, t_FR_s=t_fr,
                       speedup_total=t_fr / max(t_nodes + t_fit + t_ev, 1e-12))
            for k in measure_names(cfg.risk):
                row["%s_FR" % k] = mf[k]
                row["%s_proxy" % k] = mp_[k]
                row["d%s_pct" % k] = 100 * (mp_[k] - mf[k]) / abs(mf[k])
            e = price_errors(V, fr.values(h))
            row.update(erel_sc_mean=e["erel_sc_mean"], bias_mean=e["bias_mean"])
            rows.append(row)
    return pd.DataFrame(rows)


def compare_regimes(df_r1, df_r2):
    """R2 - R1 a (d, horizon, n_train) egal."""
    keys = ["d", "horizon", "n_train"]
    cols = ["statut", "t_nodes_s", "t_fit_s", "rang_max", "n_tours_rang", "val_err",
            "dVaR_99_pct", "dES_97_5_pct", "dES_99_pct", "erel_sc_mean"]
    a = df_r1[keys + [c for c in cols if c in df_r1]].set_index(keys)
    b = df_r2[keys + [c for c in cols if c in df_r2]
              + [c for c in df_r2 if c.startswith("bruit_")]].set_index(keys)
    m = a.join(b, lsuffix="_R1", rsuffix="_R2", how="outer")
    m["ratio_t_fit_R2_R1"] = m["t_fit_s_R2"] / m["t_fit_s_R1"]
    m["ratio_t_nodes_R2_R1"] = m["t_nodes_s_R2"] / m["t_nodes_s_R1"]
    return m


# =========================================================================
#  Figures
# =========================================================================
_METRICS = [("dVaR_99_pct", "Ecart VaR 99 % (%)"),
            ("dES_97_5_pct", "Ecart ES 97,5 % (%)"),
            ("dES_99_pct", "Ecart ES 99 % (%)"),
            ("erel_sc_mean", "Erreur relative a echelle (moy.)"),
            ("t_nodes_s", "t pricing noeuds (s)"),
            ("t_fit_s", "t fit ALS (s)"),
            ("t_eval_s", "t evaluation (s)"),
            ("rang_max", "Rang max"),
            ("val_err", "Erreur de validation")]


def plot_sweep(df, cfg, title, root=None, show=True):
    import matplotlib.pyplot as plt
    figs = []
    for h in cfg.risk.horizons:
        sub = df[df["horizon"] == h]
        fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
        for ax, (col, lab) in zip(axes.ravel(), _METRICS):
            for d, g in sub.groupby("d"):
                if col in g:
                    g = g.sort_values("n_train")
                    ax.plot(g["n_train"], g[col], marker="o", label="d=%d" % d)
            if col.startswith("d") and col.endswith("_pct"):
                ax.axhline(0, color="k", lw=.6)
            if col in ("t_fit_s", "val_err"):
                ax.set_yscale("log")
            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("n_train")
            ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("%s -- horizon %s" % (title, h))
        fig.tight_layout()
        if root:
            fig.savefig(os.path.join(root, "balayage_%s.png" % h), dpi=110)
        figs.append(fig)
        plt.show() if show else plt.close(fig)
    return figs


def plot_losses_max(cfg, sw, fr, S, regime, root=None, show=True):
    """Histogrammes des pertes FR vs TT au n_train MAXIMAL, par groupe et horizon."""
    import matplotlib.pyplot as plt
    hs = list(cfg.risk.horizons)
    dims = list(cfg.portfolio.dimensions)
    fig, axes = plt.subplots(len(dims), len(hs), figsize=(5.5 * len(hs), 3.2 * len(dims)))
    axes = np.atleast_2d(axes)
    for a_row, d in zip(axes, dims):
        for ax, h in zip(a_row, hs):
            n = sw.n_train[d][-1]
            r = load_tt(cfg, sw, S, regime, d, h, n)
            Lf = fr.V0(d) - fr.values(h, d)
            ax.hist(Lf, bins=80, alpha=.5, label="Full repricing")
            ax.axvline(np.quantile(Lf, .99), color="r", ls="--", lw=1, label="VaR 99 % FR")
            if r is not None and r["fit"]["status"] == "ok":
                Lp = r["V0"] - r["V"]
                ax.hist(Lp, bins=80, histtype="step", lw=1.4, label="TT")
                ax.axvline(np.quantile(Lp, .99), color="b", ls=":", lw=1,
                           label="VaR 99 % TT")
            else:
                ax.text(.5, .5, "TT absent / echec", transform=ax.transAxes, ha="center")
            ax.set_title("%s d=%d %s n_train=%d" % (regime, d, h, n), fontsize=9)
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    if root:
        fig.savefig(os.path.join(root, "pertes_ntrain_max.png"), dpi=110)
    plt.show() if show else plt.close(fig)
    return fig


def results_dir(cfg, regime):
    return ensure_dir(os.path.join(_sweep_root(cfg), "results", regime))
