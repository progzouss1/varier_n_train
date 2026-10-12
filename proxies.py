# -*- coding: utf-8 -*-
"""Proxies par groupe (Full Chebyshev ou Tensor-Train) : build + evaluation, CACHES.

Pour chaque groupe d :
  1. boite = enveloppe empirique des scenarios 1j + 10j (prefixe s[:d]), marge 1 % ;
  2. noeuds pricés par le GroupPricer CRN (M = cfg.mc.n_paths, seed = crn_seed),
     temps de CHAQUE noeud mesure, checkpoint + reprise ;
  3. Full CT : injection des valeurs ; TT : fit ALS (restarts, selection val) ;
  4. evaluation du proxy en s0 et sur les N scenarios de chaque horizon, chronometree.

Tout est ecrit dans cache/proxies/d<d>_<methode>_<empreinte>/ :
  nodes.npy, node_values.npy, node_times.npy, nodes.partial.npz (en cours)
  build.json   (temps init CRN / noeuds / fit, rangs, erreurs, statut)
  fit_log.txt  (stdout brut de l'ALS, TT)
  obj.pkl      (objet proxy, si picklable)
  eval_s0.npy, eval_<h>.npy, eval.json (temps d'evaluation, hors-boite)

Si eval.json existe, les resultats sont relus SANS MoCaX (rien n'est recalcule).
"""

import os
import pickle
from time import perf_counter

import numpy as np

from cache import (ensure_dir, load_json, load_npy, save_json, save_npy,
                   save_npz, stamp)
from config import fingerprint
from pricing import GroupPricer
from risk import empirical_box, inside_box

PROXY_VERSION = "v1"


# =========================================================================
#  Emplacements
# =========================================================================
def proxy_key(cfg, pf, d, box):
    method = cfg.proxies.methods[d]
    prods = [(dv.derivative_id, dv.option_type, dv.callput_flag, dv.strike,
              dv.maturity, dv.nominal) for dv in pf.group(d)]
    return fingerprint(cfg.market, prods, cfg.mc.n_paths, cfg.mc.crn_seed,
                       method, cfg.proxies.params_for(d),
                       np.round(np.asarray(box), 10).tolist(), PROXY_VERSION)


def proxy_dir(cfg, pf, d, box):
    return os.path.join(cfg.run.cache_dir, "proxies", "d%02d_%s_%s"
                        % (d, cfg.proxies.methods[d], proxy_key(cfg, pf, d, box)))


# =========================================================================
#  Pricing des noeuds (CRN), cache + reprise
# =========================================================================
def price_nodes(pricer, pts, root, checkpoint_every=250, verbose=True, label=""):
    """-> (valeurs, temps par noeud). Reprend un calcul interrompu."""
    fv, ft = os.path.join(root, "node_values.npy"), os.path.join(root, "node_times.npy")
    fpart = os.path.join(root, "nodes.partial.npz")
    if os.path.exists(fv) and os.path.exists(ft):
        if verbose:
            print("  noeuds : valeurs chargees du cache (%d)" % len(pts))
        return load_npy(fv), load_npy(ft)
    n = len(pts)
    vals, tms = np.full(n, np.nan), np.full(n, np.nan)
    k0 = 0
    if os.path.exists(fpart):
        z = np.load(fpart)
        k0 = int(z["n_done"])
        vals[:k0], tms[:k0] = z["values"][:k0], z["times"][:k0]
        if verbose:
            print("  noeuds : reprise a %d/%d" % (k0, n))
    for k in range(k0, n):
        t0 = perf_counter()
        vals[k] = pricer(pts[k])
        tms[k] = perf_counter() - t0
        done = k + 1
        if checkpoint_every and done % checkpoint_every == 0 and done < n:
            save_npz(fpart, values=vals, times=tms, n_done=done)
        if verbose and (done % 100 == 0 or done == n):
            el = np.nansum(tms)
            print("  %s noeuds : %5d/%d  %.3fs/noeud  ETA %.0fs   "
                  % (label, done, n, el / done, el / done * (n - done)),
                  end="\r", flush=True)
    if verbose:
        print()
    save_npy(fv, vals)
    save_npy(ft, tms)
    if os.path.exists(fpart):
        os.remove(fpart)
    return vals, tms


# =========================================================================
#  Full Chebyshev Tensor (mocaxpy)
# =========================================================================
def _set_ct_values(ct, vals):
    for name in ("set_evaluation_points", "set_original_function_values",
                 "set_function_values", "set_evaluation_values", "set_values"):
        f = getattr(ct, name, None)
        if callable(f):
            f(vals)
            return name
    raise AttributeError("Aucun setter de valeurs reconnu sur Mocax : %s"
                         % sorted(m for m in dir(ct) if m.startswith("set")))


def _build_full_cheb(mocaxpy, box, ct_cfg, pricer, root, cfg, verbose):
    d = box.shape[0]
    domain = mocaxpy.MocaxDomain([[float(a), float(b)] for a, b in box])
    ns = mocaxpy.MocaxNs([int(ct_cfg.degree)] * d)
    ct = mocaxpy.Mocax(None, d, domain, None, ns)
    nodes = np.asarray(ct.get_evaluation_points(), dtype=float)
    fn = os.path.join(root, "nodes.npy")
    if os.path.exists(fn):
        cached = load_npy(fn)
        if cached.shape != nodes.shape or not np.allclose(cached, nodes, rtol=0, atol=1e-10):
            raise RuntimeError("Noeuds CT caches != noeuds MoCaX (%s)" % root)
    else:
        save_npy(fn, nodes)
    if verbose:
        print("  Full CT d=%d : %d noeuds (%d^%d)" % (d, len(nodes), ct_cfg.degree + 1, d))
    vals, tms = price_nodes(pricer, nodes, root, cfg.run.checkpoint_every,
                            verbose, label="CT d=%d" % d)
    t = {}
    t0 = perf_counter()
    setter = _set_ct_values(ct, [float(v) for v in vals])
    t["t_fit"] = perf_counter() - t0
    try:
        err = float(ct.get_error_threshold())
    except Exception:
        err = float("nan")
    info = dict(n_nodes=len(nodes), setter=setter, err_ex_ante=err,
                degree=ct_cfg.degree)

    def eval_fn(P):
        P = np.atleast_2d(np.asarray(P, dtype=float))
        return np.array([float(ct.eval([float(x) for x in p])) for p in P])

    return ct, eval_fn, tms, t, info


# =========================================================================
#  Tensor-Train (MoCaXExtend)
# =========================================================================
def _build_tt(me, box, tt_cfg, pricer, root, cfg, verbose):
    from mocax_tt import fit_tt, make_subgrid, patch_mocax
    patch_mocax(me)
    d = box.shape[0]
    fn = os.path.join(root, "nodes.npy")
    if os.path.exists(fn):
        pts = load_npy(fn)
        sg = load_json(os.path.join(root, "subgrid.json"))
        if verbose:
            print("  TT d=%d : sous-grille chargee du cache (%d points)" % (d, len(pts)))
    else:
        pts, sg = make_subgrid(me, box, tt_cfg)
        save_npy(fn, pts)
        save_json(os.path.join(root, "subgrid.json"), sg)
    if verbose:
        print("  TT d=%d : degre %d, n_train demande %d -> %d points (%d uniques), "
              "grille pleine %.3g" % (d, tt_cfg.degree, sg["n_train_req"],
                                      sg["n_train"], sg["n_train_unique"],
                                      float(sg["tensor_size"])))
    vals, tms = price_nodes(pricer, pts, root, cfg.run.checkpoint_every,
                            verbose, label="TT d=%d" % d)
    if verbose:
        print("  TT d=%d : fit ALS (%d restart(s))..." % (d, tt_cfg.n_restarts))
    obj, eval_fn, times, info = fit_tt(me, box, tt_cfg, pts, vals, verbose)
    info.update(sg)
    with open(os.path.join(root, "fit_log.txt"), "w") as f:
        f.write(info.pop("fit_log") or "")
    return obj, eval_fn, tms, {"t_fit": times["t_extension"]}, info


# =========================================================================
#  Groupe : build + evaluation (caches)
# =========================================================================
def run_group_proxy(cfg, market, pf, d, S_by_h, mocaxpy=None, me=None,
                    verbose=True, retry_failed=False):
    """-> dict de resultats du proxy du groupe d (charge du cache si present)."""
    method = cfg.proxies.methods[d]
    params = cfg.proxies.params_for(d)
    box = empirical_box(S_by_h, d, cfg.box)
    root = ensure_dir(proxy_dir(cfg, pf, d, box))
    hs = list(cfg.risk.horizons)
    f_build, f_eval = os.path.join(root, "build.json"), os.path.join(root, "eval.json")

    # ---------- 0) tout est deja en cache : lecture seule, sans MoCaX -----
    if os.path.exists(f_eval) and os.path.exists(f_build):
        if verbose:
            print("Proxy d=%d (%s) : resultats charges du cache %s" % (d, method, root))
        return _load_result(root, d, method, box, hs)
    if os.path.exists(f_build) and not retry_failed:
        b = load_json(f_build)
        if b.get("status") == "failed":
            if verbose:
                print("Proxy d=%d (%s) : ECHEC en cache (%s). retry_failed=True "
                      "pour reessayer." % (d, method, b.get("error")))
            return dict(d=d, method=method, box=box, root=root, status="failed",
                        build=b)

    lib = mocaxpy if method == "full_cheb" else me
    if lib is None:
        raise RuntimeError("Proxy d=%d (%s) absent du cache et bibliotheque MoCaX "
                           "non chargee." % (d, method))

    # ---------- 1) pricer CRN des noeuds (paquet fige, init chronometree) --
    t0 = perf_counter()
    pricer = GroupPricer(pf.group(d), market, cfg.mc.n_paths, cfg.mc.crn_seed)
    t_init = perf_counter() - t0
    if verbose:
        print("Proxy d=%d (%s) : %r, init %.2fs" % (d, method, pricer, t_init))

    # ---------- 2) build ------------------------------------------------
    base = dict(d=d, method=method, params=params.__dict__, box=box,
                n_paths=cfg.mc.n_paths, crn_seed=cfg.mc.crn_seed,
                t_init_crn=t_init, **stamp())
    try:
        if method == "full_cheb":
            obj, eval_fn, node_t, t, info = _build_full_cheb(
                mocaxpy, box, params, pricer, root, cfg, verbose)
        else:
            obj, eval_fn, node_t, t, info = _build_tt(
                me, box, params, pricer, root, cfg, verbose)
    except Exception as exc:
        diag = getattr(exc, "diag", None)
        b = dict(base, status="failed", error="%s: %s" % (type(exc).__name__, exc),
                 diag=diag)
        nt = os.path.join(root, "node_times.npy")
        if os.path.exists(nt):
            b["t_nodes"] = float(load_npy(nt).sum())
        save_json(f_build, b)
        print("Proxy d=%d (%s) : ECHEC du build -> %s (enregistre dans %s)"
              % (d, method, b["error"], f_build))
        return dict(d=d, method=method, box=box, root=root, status="failed", build=b)

    b = dict(base, status="ok", info=info, n_nodes=int(len(node_t)),
             t_nodes=float(np.sum(node_t)), t_node_mean=float(np.mean(node_t)),
             t_fit=float(t["t_fit"]),
             t_build=float(t_init + np.sum(node_t) + t["t_fit"]))
    try:
        with open(os.path.join(root, "obj.pkl"), "wb") as f:
            pickle.dump(obj, f)
        b["pickled"] = True
    except Exception as exc:
        b["pickled"] = "non picklable : %r" % (exc,)
        try:
            os.remove(os.path.join(root, "obj.pkl"))
        except OSError:
            pass
    save_json(f_build, b)

    # ---------- 3) evaluation (s0 + horizons), chronometree -------------
    ev = dict(t_eval={}, n_outside={}, **stamp())
    s0 = market.spots[:d].reshape(1, -1)
    pts_by = {"s0": s0}
    pts_by.update({h: np.asarray(S_by_h[h])[:, :d] for h in hs})
    for s, P in pts_by.items():
        inside = inside_box(P, box)
        out = np.full(len(P), np.nan)
        t0 = perf_counter()
        if inside.any():
            out[inside] = eval_fn(P[inside])
        if (~inside).any():                      # jamais d'extrapolation
            out[~inside] = [pricer(p) for p in P[~inside]]
        ev["t_eval"][s] = perf_counter() - t0
        ev["n_outside"][s] = int((~inside).sum())
        save_npy(os.path.join(root, "eval_%s.npy" % s), out)
        if verbose:
            print("  eval %-4s : %6d points en %.3fs (%d hors boite)"
                  % (s, len(P), ev["t_eval"][s], ev["n_outside"][s]))
    save_json(f_eval, ev)
    return _load_result(root, d, method, box, hs)


def _load_result(root, d, method, box, hs):
    b = load_json(os.path.join(root, "build.json"))
    ev = load_json(os.path.join(root, "eval.json"))
    V = {s: load_npy(os.path.join(root, "eval_%s.npy" % s)) for s in ["s0"] + hs}
    return dict(d=d, method=method, box=np.asarray(box), root=root, status="ok",
                build=b, eval=ev, V0=float(V["s0"][0]),
                V={h: V[h] for h in hs},
                node_times=load_npy(os.path.join(root, "node_times.npy")))
