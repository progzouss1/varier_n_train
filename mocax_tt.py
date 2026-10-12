# -*- coding: utf-8 -*-
"""Enrobage de MoCaXExtend : domaine, build TT instrumente, garde-fous.

Encode TOUTES les mesures de notes.md, pour que le notebook n'ait jamais a s'en
souvenir :

  1. get_tensor_size renvoie n^d au lieu de (n+1)^d      -> patch_mocax(me)
  2. gen_train_val_data ne lit QUE la cle `percentage`   -> appel nomme + garde
  3. run_rank_adaptive_algo est stochastique (np.random) -> seed + restarts,
     selection sur l'erreur de VALIDATION (corr. log-log val/test = +0.999)
  4. subgrid_by_number tire AVEC remise                  -> n_train reel = len(pts)
  +  comportement hors domaine non tranche               -> eval_safe (filtre + repli)

Le build est en DEUX temps pour que le cache porte sur l'etape couteuse :
  make_subgrid -> pts ; pricing des noeuds (CRN, hors de ce module, cache) ;
  fit_tt(pts, vals) -> (obj, eval_fn, times, info). info contient les rangs
FINAUX du meilleur fit, ceux de chaque restart, les erreurs eventuelles de
chaque restart, et la trajectoire de rangs capturee sur stdout.
"""

import io
import re
from contextlib import redirect_stdout

import numpy as np

from monitoring import chrono


# =========================================================================
#  Patch de l'API (notes.md, point 1)
# =========================================================================
def patch_mocax(me):
    """Corrige get_tensor_size : (n+1)^d au lieu de n^d. Idempotent.

    Produit en ENTIERS PYTHON (precision arbitraire) : np.prod utiliserait
    int32 sous Windows et deborde des d=15 (7^15 > 2^31), rendant la taille
    negative et cassant subgrid_by_number.
    """
    if me is None:
        return me
    if not hasattr(me.MocaxExtend, "_get_tensor_size_orig"):
        me.MocaxExtend._get_tensor_size_orig = me.MocaxExtend.get_tensor_size

        def _fixed(self):
            out = 1
            for n in self.num_cheb_pts:
                out *= int(n) + 1
            return out
        me.MocaxExtend.get_tensor_size = _fixed
    return me


# =========================================================================
#  Introspection des rangs
# =========================================================================
def final_ranks(obj):
    """Sequence de rangs du tenseur entraine ([1, r1, ..., 1]) ou None.

    `trained_tt_tensor` est le nom documente (Listing 24.25) ; replis defensifs.
    """
    holders = (obj, getattr(obj, "trained_tt_tensor", None),
               getattr(obj, "cheb_tensor_tt", None))
    for h in holders:
        if h is None:
            continue
        for name in ("ranks", "get_ranks", "tt_ranks", "get_tt_ranks"):
            v = getattr(h, name, None)
            if v is None:
                continue
            if callable(v):
                try:
                    v = v()
                except Exception:
                    continue
            try:
                out = [int(x) for x in np.asarray(v).ravel()]
            except (TypeError, ValueError):
                continue
            if out:
                return out
        cores = getattr(h, "cores", None)
        if cores is not None:
            try:
                return [int(c.shape[0]) for c in cores] + [1]
            except Exception:
                pass
    return None


_INT_LIST = re.compile(r"\[(\s*\d+\s*(?:,\s*\d+\s*)+)\]")


def parse_rank_trajectory(log_text):
    """Trajectoire des rangs extraite du stdout de run_rank_adaptive_algo.

    Heuristique : toute liste d'entiers imprimee dont premier = dernier = 1 est
    interpretee comme une sequence de rangs TT. Renvoie la liste des sequences
    dans l'ordre d'apparition ([] si le format ne se laisse pas parser :
    le log brut reste disponible dans info["fit_log"]).
    """
    traj = []
    for m in _INT_LIST.finditer(log_text or ""):
        try:
            seq = [int(x) for x in m.group(1).split(",")]
        except ValueError:
            continue
        if len(seq) >= 3 and seq[0] == 1 and seq[-1] == 1:
            if not traj or traj[-1] != seq:
                traj.append(seq)
    return traj


# =========================================================================
#  Erreur de validation (selection des restarts, notes.md point 3)
# =========================================================================
def validation_error(obj):
    """Erreur max |pred - vraie| sur le jeu de VALIDATION interne de l'objet."""
    ent = np.atleast_2d(np.asarray(obj.subgrid_entries_val)).astype(int)
    nodes = [np.asarray(x, dtype=float) for x in obj.cheb_pts_per_dimension]
    coords = np.column_stack([nodes[j][ent[:, j]] for j in range(ent.shape[1])])
    pred = np.asarray(obj.cheb_tensor_evals(coords), dtype=float).ravel()
    true = np.asarray(obj.subgrid_vals_val, dtype=float).ravel()
    return float(np.abs(pred - true).max())


# =========================================================================
#  Build TT en deux temps : sous-grille (+ pricing externe, cache) puis fit
# =========================================================================
def make_subgrid(me, box, tt_cfg):
    """Tire la sous-grille (reproductible : np.random.seed(seed_subgrid)).

    -> (pts (n, d) float, info) ; les points sont des COORDONNEES (spots),
    c'est eux que l'on price puis que l'on met en cache.
    """
    if me is None:
        raise RuntimeError("MoCaXExtend indisponible.")
    patch_mocax(me)
    d = box.shape[0]
    ranges = [[float(a), float(b)] for a, b in box]
    proto = me.MocaxExtend(d, [int(tt_cfg.degree)] * d, ranges)
    k = min(int(tt_cfg.n_train), proto.get_tensor_size())
    np.random.seed(int(tt_cfg.seed_subgrid))
    pts = np.asarray(proto.subgrid_by_number(k), dtype=float)
    n_unique = len(np.unique(np.round(pts, 12), axis=0))
    return pts, dict(n_train_req=int(k), n_train=int(len(pts)),
                     n_train_unique=int(n_unique),
                     tensor_size=int(proto.get_tensor_size()))


def _set_subgrid(obj, pts):
    try:
        obj.set_subgrid(pts)
    except (AttributeError, TypeError, ValueError):
        obj.set_subgrid(np.asarray(pts).tolist())


def fit_tt(me, box, tt_cfg, pts, vals, verbose=True):
    """Fit ALS (restarts) sur des valeurs aux noeuds DEJA calculees.

    Chaque restart : objet neuf, MEMES noeuds (set_subgrid), MEMES valeurs,
    np.random.seed(seed_fit + 1000 i), run_rank_adaptive_algo. Selection sur
    l'erreur de VALIDATION. Un restart qui plante (ex. LinAlgError "SVD did
    not converge", vu en d=25) est ENREGISTRE (message) et ignore ; si tous
    plantent -> TTFitError portant le diagnostic complet.

    -> (obj, eval_fn, times, info) ; times = {"t_extension": total restarts}
    """
    patch_mocax(me)
    d = box.shape[0]
    ranges = [[float(a), float(b)] for a, b in box]
    degree = int(tt_cfg.degree)
    algo = tt_cfg.algo_params(print_progress=True)
    times, restarts, t_each = {}, [], []
    with chrono(times, "t_extension"):
        for i in range(int(tt_cfg.n_restarts)):
            buf = io.StringIO()
            t_run = {}
            rec = dict(obj=None, val_err=float("inf"), log="", ranks=None, error=None)
            try:
                obj = me.MocaxExtend(d, [degree] * d, ranges)
                _set_subgrid(obj, pts)
                _set_values(obj, vals)
                _split(obj, tt_cfg.train_frac)
                np.random.seed(int(tt_cfg.seed_fit) + 1000 * i)
                with chrono(t_run, "t"), redirect_stdout(buf):
                    obj.run_rank_adaptive_algo(**algo)
                rec["obj"] = obj
                rec["ranks"] = final_ranks(obj)
                try:
                    rec["val_err"] = validation_error(obj)
                except Exception as exc:
                    rec["error"] = "validation_error: %r" % (exc,)
            except Exception as exc:                  # LinAlgError, SystemError...
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
                if verbose:
                    print("  restart %d ECHEC : %s" % (i, rec["error"]))
            rec["log"] = buf.getvalue()
            restarts.append(rec)
            t_each.append(t_run.get("t", float("nan")))

    ok = [i for i, r in enumerate(restarts) if r["obj"] is not None
          and np.isfinite(r["val_err"])]
    diag = dict(restart_val_errs=[r["val_err"] for r in restarts],
                restart_ranks=[r["ranks"] for r in restarts],
                restart_errors=[r["error"] for r in restarts],
                t_extension_each=t_each, t_extension=times["t_extension"])
    if not ok:
        raise TTFitError("Tous les restarts ALS ont echoue.", diag)
    best_i = min(ok, key=lambda i: restarts[i]["val_err"])
    best = restarts[best_i]
    obj = best["obj"]
    ranks = best["ranks"]
    core = ranks[1:-1] if (ranks and len(ranks) > 2) else (ranks or [])
    rmax = max(core) if core else (max(ranks) if ranks else None)
    info = dict(d=d, n_train=int(len(pts)),
                tensor_size=int(obj.get_tensor_size()),
                ranks=ranks, rank_max=rmax,
                saturated=(None if rmax is None else bool(rmax >= tt_cfg.max_rank)),
                val_err=best["val_err"], best_restart=best_i,
                rank_trajectory=parse_rank_trajectory(best["log"]),
                fit_log=best["log"], **diag)

    def eval_fn(q):
        arr = np.atleast_2d(np.asarray(q, dtype=float))
        try:
            return np.asarray(obj.cheb_tensor_evals(arr), dtype=float).ravel()
        except (AttributeError, TypeError):
            return np.asarray(obj.cheb_tensor_evals(arr.tolist()), dtype=float).ravel()

    return obj, eval_fn, times, info


class TTFitError(RuntimeError):
    def __init__(self, msg, diag):
        RuntimeError.__init__(self, msg)
        self.diag = diag


def _set_values(obj, vals):
    try:
        obj.set_subgrid_values(vals)
    except (AttributeError, TypeError):
        obj.set_subgrid_values(np.asarray(vals).tolist())


def _split(obj, train_frac):
    """gen_train_val_data via la SEULE cle lue, `percentage`, + garde (point 2)."""
    obj.gen_train_val_data(percentage=float(train_frac))
    ntr, nva = obj.get_training_set_size(), obj.get_validating_set_size()
    got = ntr / float(ntr + nva)
    if abs(got - train_frac) > 0.06:
        raise RuntimeError("split demande %.2f, obtenu %.2f" % (train_frac, got))


# =========================================================================
#  Evaluation sure + erreurs
# =========================================================================
def eval_safe(obj, points, fallback=None):
    """(valeurs, masque_dedans). Hors boite : fallback(point) sinon NaN.

    JAMAIS d'extrapolation silencieuse ; et ne jamais SUPPRIMER les scenarios
    hors boite d'un calcul de VaR (ce sont les plus extremes) : repli MC.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    box = np.asarray(obj.intervals, dtype=float)
    inside = np.all((P >= box[:, 0] - 1e-12) & (P <= box[:, 1] + 1e-12), axis=1)
    out = np.full(len(P), np.nan)
    if inside.any():
        out[inside] = np.asarray(obj.cheb_tensor_evals(P[inside]),
                                 dtype=float).ravel()
    if fallback is not None and (~inside).any():
        out[~inside] = np.asarray([fallback(p) for p in P[~inside]], dtype=float)
    return out, inside
