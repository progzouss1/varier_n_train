# -*- coding: utf-8 -*-
"""Tableaux de resultats : full repricing vs proxies, VaR/ES, temps, speed-ups.

Rien n'est calcule ici qui ne soit deja en cache : on relit le full repricing
(FullRepricing) et les resultats de proxies (run_group_proxy), on compare, et
on ECRIT les tableaux (CSV + JSON) et figures dans cache/results/<empreinte>/.

Definitions (toutes les durees sont MESUREES, jamais projetees) :
  t_FR[h]      = somme des temps des pricings unitaires (100 produits ou ceux du
                 groupe) sur les N scenarios de l'horizon h (sequentiel equivalent)
  t_build      = t_init_crn + t_nodes (pricing CRN des noeuds) + t_fit
  t_eval[h]    = evaluation du proxy sur les N scenarios de h
  speedup_online[h] = t_FR[h] / t_eval[h]
  speedup_total     = (t_FR[s0] + sum_h t_FR[h]) / (t_build + t_eval[s0] + sum_h t_eval[h])
"""

import os

import numpy as np
import pandas as pd

from cache import ensure_dir, save_df, save_json, stamp
from config import fingerprint
from full_repricing import fr_key
from risk import measure_names, price_errors, risk_measures


def results_dir(cfg, proxy_results):
    fp = fingerprint(fr_key(cfg), sorted(r["root"] for r in proxy_results.values()))
    root = ensure_dir(os.path.join(cfg.run.cache_dir, "results", fp))
    save_json(os.path.join(cfg.run.cache_dir, "results", "LATEST.json"),
              dict(path=root, **stamp()))
    return root


# =========================================================================
#  Full repricing seul
# =========================================================================
def fr_risk_table(fr, cfg):
    rows = []
    perims = [("d=%d" % d, d) for d in cfg.portfolio.dimensions] + [("portefeuille", None)]
    for name, d in perims:
        V0 = fr.V0(d)
        for h in cfg.risk.horizons:
            L = V0 - fr.values(h, d)
            r = dict(perimetre=name, horizon=h, V0=V0)
            r.update(risk_measures(L, cfg.risk))
            r["t_FR_s"] = fr.time_sum(h, d)
            r["t_FR_par_scenario_s"] = fr.time_per_scenario(h, d)
            r["n_pricings"] = int(fr.times[h][:, fr._mask(d)].size)
            rows.append(r)
    return pd.DataFrame(rows).set_index(["perimetre", "horizon"])


def fr_group_table(fr, cfg):
    rows = []
    V0 = fr.V0()
    for d in cfg.portfolio.dimensions:
        r = dict(d=d, V0_d=fr.V0(d), part_V0_pct=100 * fr.V0(d) / V0)
        for s in ["s0"] + list(cfg.risk.horizons):
            r["t_FR_%s_s" % s] = fr.time_sum(s, d)
        r["t_1_pricing_moy_s"] = float(np.mean(np.concatenate(
            [fr.times[s][:, fr._mask(d)].ravel() for s in fr.times])))
        rows.append(r)
    r = dict(d="total", V0_d=V0, part_V0_pct=100.0)
    for s in ["s0"] + list(cfg.risk.horizons):
        r["t_FR_%s_s" % s] = fr.time_sum(s)
    r["t_1_pricing_moy_s"] = float(np.mean(np.concatenate(
        [fr.times[s].ravel() for s in fr.times])))
    rows.append(r)
    return pd.DataFrame(rows).set_index("d")


# =========================================================================
#  Proxies vs full repricing
# =========================================================================
def _compare_rows(name, method, V0_fr, V0_px, Vfr, Vpx, cfg, t_fr, t_ev, extra):
    rows = []
    for h in cfg.risk.horizons:
        Lfr, Lpx = V0_fr - Vfr[h], V0_px - Vpx[h]
        mfr, mpx = risk_measures(Lfr, cfg.risk), risk_measures(Lpx, cfg.risk)
        r = dict(perimetre=name, horizon=h, methode=method,
                 V0_FR=V0_fr, V0_proxy=V0_px, dV0=V0_px - V0_fr)
        for k in measure_names(cfg.risk):
            r["%s_FR" % k] = mfr[k]
            r["%s_proxy" % k] = mpx[k]
            r["d%s" % k] = mpx[k] - mfr[k]
            r["d%s_pct" % k] = 100 * (mpx[k] - mfr[k]) / abs(mfr[k]) if mfr[k] else np.nan
        e = price_errors(Vpx[h], Vfr[h])
        r.update({k: e[k] for k in ("erel_sc_max", "erel_sc_mean", "eabs_max",
                                    "eabs_mean", "bias_mean")})
        r["t_FR_s"] = t_fr[h]
        r["t_eval_s"] = t_ev[h]
        r["speedup_online"] = t_fr[h] / max(t_ev[h], 1e-12)
        r.update(extra)
        rows.append(r)
    return rows


def proxy_tables(fr, proxy_results, cfg):
    """-> (df_compare [perimetre x horizon], df_times [perimetre])."""
    hs = list(cfg.risk.horizons)
    stages = ["s0"] + hs
    cmp_rows, time_rows = [], []
    Vpf = {h: np.zeros(cfg.risk.n_scenarios) for h in hs}
    V0pf = 0.0
    tot = dict(t_build=0.0, t_nodes=0.0, t_fit=0.0, t_init_crn=0.0, n_nodes=0,
               t_eval={s: 0.0 for s in stages})
    fallback = []

    for d in cfg.portfolio.dimensions:
        res = proxy_results.get(d)
        t_fr = {s: fr.time_sum(s, d) for s in stages}
        if res is None or res.get("status") != "ok":
            # groupe sans proxy : on NE CACHE PAS l'echec -> repli full repricing
            fallback.append(d)
            b = (res or {}).get("build", {})
            time_rows.append(dict(perimetre="d=%d" % d,
                                  methode=(res or {}).get("method", "?"),
                                  statut="ECHEC : %s" % b.get("error", "non construit"),
                                  t_nodes_s=b.get("t_nodes", np.nan)))
            V0pf += fr.V0(d)
            for h in hs:
                Vpf[h] += fr.values(h, d)
            tot["t_eval"] = {s: tot["t_eval"][s] + t_fr[s] for s in stages}
            continue
        b, ev = res["build"], res["eval"]
        V0_px, Vpx = res["V0"], res["V"]
        V0pf += V0_px
        for h in hs:
            Vpf[h] += Vpx[h]
        t_ev = {s: ev["t_eval"][s] for s in stages}
        info = b.get("info", {})
        cmp_rows += _compare_rows(
            "d=%d" % d, res["method"], fr.V0(d), V0_px,
            {h: fr.values(h, d) for h in hs}, Vpx, cfg, t_fr, t_ev,
            dict(n_hors_boite=sum(ev["n_outside"].values())))
        t_all_fr = sum(t_fr.values())
        t_all_px = b["t_build"] + sum(t_ev.values())
        tr = dict(perimetre="d=%d" % d, methode=res["method"], statut="ok",
                  n_noeuds=b["n_nodes"], t_init_crn_s=b["t_init_crn"],
                  t_nodes_s=b["t_nodes"], t_noeud_moy_s=b["t_node_mean"],
                  t_fit_s=b["t_fit"], t_build_s=b["t_build"])
        for s in stages:
            tr["t_eval_%s_s" % s] = t_ev[s]
            tr["t_FR_%s_s" % s] = t_fr[s]
        for h in hs:
            tr["speedup_online_%s" % h] = t_fr[h] / max(t_ev[h], 1e-12)
        tr["t_FR_total_s"] = t_all_fr
        tr["t_proxy_total_s"] = t_all_px
        tr["speedup_total"] = t_all_fr / max(t_all_px, 1e-12)
        if res["method"] == "tt":
            tr.update(rangs=info.get("ranks"), rang_max=info.get("rank_max"),
                      sature=info.get("saturated"), val_err=info.get("val_err"),
                      n_train=info.get("n_train"),
                      n_train_uniques=info.get("n_train_unique"),
                      restart_retenu=info.get("best_restart"),
                      erreurs_restarts=info.get("restart_errors"))
        else:
            tr.update(err_ex_ante=info.get("err_ex_ante"))
        time_rows.append(tr)
        for k in ("t_build", "t_nodes", "t_fit", "t_init_crn"):
            tot[k] += b[k]
        tot["n_nodes"] += b["n_nodes"]
        tot["t_eval"] = {s: tot["t_eval"][s] + t_ev[s] for s in stages}

    # ---------------- portefeuille : SOMME des evaluations des proxies -----
    t_fr = {s: fr.time_sum(s) for s in stages}
    label = "portefeuille" + (" (repli FR : d=%s)" % fallback if fallback else "")
    cmp_rows += _compare_rows(
        label, "somme proxies", fr.V0(), V0pf, {h: fr.values(h) for h in hs}, Vpf,
        cfg, t_fr, tot["t_eval"], dict(groupes_en_repli_FR=str(fallback)))
    t_all_fr = sum(t_fr.values())
    t_all_px = tot["t_build"] + sum(tot["t_eval"].values())
    tr = dict(perimetre=label, methode="somme proxies",
              statut="ok" if not fallback else "PARTIEL",
              n_noeuds=tot["n_nodes"], t_init_crn_s=tot["t_init_crn"],
              t_nodes_s=tot["t_nodes"], t_fit_s=tot["t_fit"], t_build_s=tot["t_build"])
    for s in stages:
        tr["t_eval_%s_s" % s] = tot["t_eval"][s]
        tr["t_FR_%s_s" % s] = t_fr[s]
    for h in hs:
        tr["speedup_online_%s" % h] = t_fr[h] / max(tot["t_eval"][h], 1e-12)
    tr.update(t_FR_total_s=t_all_fr, t_proxy_total_s=t_all_px,
              speedup_total=t_all_fr / max(t_all_px, 1e-12))
    time_rows.append(tr)

    df_cmp = pd.DataFrame(cmp_rows).set_index(["perimetre", "horizon"])
    df_t = pd.DataFrame(time_rows).set_index("perimetre")
    return df_cmp, df_t, dict(V0=V0pf, V=Vpf, fallback=fallback)


def summary_table(df_cmp, cfg):
    """Vue compacte : mesures FR vs proxy et ecarts %, par perimetre et horizon."""
    cols = ["methode"]
    for k in measure_names(cfg.risk):
        cols += ["%s_FR" % k, "%s_proxy" % k, "d%s_pct" % k]
    cols += ["erel_sc_mean", "t_FR_s", "t_eval_s", "speedup_online"]
    return df_cmp[cols]


# =========================================================================
#  Sauvegarde + figures
# =========================================================================
def save_all(root, tables):
    for name, df in tables.items():
        save_df(df, os.path.join(root, name))
    return root


def plot_losses(fr, proxy_results, pf_proxy, cfg, root=None, show=True):
    import matplotlib.pyplot as plt
    hs = list(cfg.risk.horizons)
    perims = [(d, "d=%d" % d) for d in cfg.portfolio.dimensions] + [(None, "portefeuille")]
    figs = []
    for d, name in perims:
        if d is None:
            V0px, Vpx = pf_proxy["V0"], pf_proxy["V"]
        else:
            res = proxy_results.get(d)
            if not res or res.get("status") != "ok":
                continue
            V0px, Vpx = res["V0"], res["V"]
        fig, ax = plt.subplots(1, len(hs), figsize=(5.5 * len(hs), 3.8))
        ax = np.atleast_1d(ax)
        for a, h in zip(ax, hs):
            Lfr = fr.V0(d) - fr.values(h, d)
            Lpx = V0px - Vpx[h]
            a.hist(Lfr, bins=80, alpha=.5, label="Full repricing")
            a.hist(Lpx, bins=80, histtype="step", lw=1.5, label="Proxy")
            for lvl, ls in zip(cfg.risk.var_levels, ["--"]):
                a.axvline(np.quantile(Lfr, lvl), color="r", ls=ls, lw=1,
                          label="VaR %.0f%% FR" % (100 * lvl))
                a.axvline(np.quantile(Lpx, lvl), color="b", ls=":", lw=1,
                          label="VaR %.0f%% proxy" % (100 * lvl))
            a.set_title("%s, pertes L, horizon %s" % (name, h))
            a.set_xlabel("perte")
            a.legend(fontsize=7)
            a.grid(alpha=.3)
        fig.tight_layout()
        if root:
            fig.savefig(os.path.join(root, "pertes_%s.png"
                                     % name.replace("=", "").replace(" ", "_")), dpi=120)
        figs.append(fig)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figs
