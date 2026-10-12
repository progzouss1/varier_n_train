# -*- coding: utf-8 -*-
"""Genere experience_portefeuille.ipynb (python tests/build_notebook.py)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "experience_portefeuille.ipynb")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# VaR / ES d'un portefeuille de dérivés de corrélation — Full repricing vs proxies (Full Chebyshev, Tensor-Train)

**Règles de cette version** (détail et écarts avec la version précédente : `CHANGEMENTS.md`)

| Élément | Choix |
|---|---|
| Full repricing | **un seul**, **unitaire** (instrument par instrument, 100 pricings par scénario), régime **noCRN** : générateur **re-créé avec la même seed** à chaque pricing (coût d'un pricer front office) |
| Trajectoires | **100 000** pour **tous** les pricings (full repricing, V0, nœuds des proxies) |
| Scénarios | **10 000** chocs corrélés sur les 25 sous-jacents, **même** paquet pour 1j et 10j ; le groupe $d$ lit le préfixe $s_{1:d}$ |
| Mesures | pour **1j** et **10j** séparément : **VaR 99 %**, **ES 97,5 %**, **ES 99 %** |
| Boîte des proxies | enveloppe empirique des scénarios 1j + 10j : $\text{lo}_i = 0{,}99\min_{k,h}s^{(k)}_{h,i}$, $\text{hi}_i = 1{,}01\max_{k,h}s^{(k)}_{h,i}$ |
| Nœuds des proxies | pricer de groupe **CRN** (paquet figé), Full Chebyshev ($d=5$) et TT ($d=10\dots25$) |
| Cache | **tout** : scénarios, chaque pricing unitaire (valeur + temps), nœuds (valeur + temps), fits, évaluations, tableaux, figures. Reprise exacte après interruption. |

Le full repricing est long (ordre de grandeur mesuré : ~4 s par scénario pour le portefeuille, ~20 h en séquentiel). Il peut être lancé **ici** ou, mieux, en ligne de commande : `nohup python run_all.py --stage fr --workers 8 > fr.log 2>&1 &` ; ce notebook relit ensuite le même cache.
""")

md("## 0. Imports")
code(r"""
import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from config import ExperimentConfig
from model import OPTION_TYPES
from portfolio import build_market, build_portfolio, check_portfolio, basket_level
from risk import get_scenarios, empirical_box, inside_box, scenarios_dir
import full_repricing as FR
from proxies import run_group_proxy
import analysis as A

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
""")
code(r"""
# MoCaX : necessaire UNIQUEMENT pour construire un proxy absent du cache.
MOCAX_ROOT = os.path.join(os.path.expanduser("~"), "mocax")
mocaxpy = me = None
try:
    from mocax_loader import load_mocax
    from mocax_extend_loader import load_mocax_extend
    mocaxpy = load_mocax(search_roots=[MOCAX_ROOT])
    me = load_mocax_extend(search_roots=[MOCAX_ROOT])
except Exception as exc:
    print("MoCaX non charge (%s) -> proxies lus du cache uniquement." % exc)
""")

md("## 1. Configuration")
code(r"""
# True = mini-run de verification (cache separe, resultats NON exploitables)
SMOKE = False

cfg = ExperimentConfig.smoke() if SMOKE else ExperimentConfig().validate()
DIMS = list(cfg.portfolio.dimensions)
HS = list(cfg.risk.horizons)
print(cfg.summary())
""")

md("## 2. Marché, portefeuille")
code(r"""
market = build_market(cfg.market)
pf = build_portfolio(cfg.portfolio, market)
check_portfolio(cfg, market, pf)

display(Markdown("**Marché**"))
w = np.linalg.eigvalsh(market.corr_matrix)
display(pd.DataFrame([
    ("sous-jacents", cfg.market.n_underlyings), ("taux r", market.rate),
    ("corrélation rho", cfg.market.rho),
    ("spots", "%.2f .. %.2f" % (market.spots.min(), market.spots.max())),
    ("volatilités", "%.1f%% .. %.1f%%" % (100*market.vols.min(), 100*market.vols.max())),
    ("vp min / max / cond.", "%.3f / %.3f / %.1f" % (w.min(), w.max(), w.max()/w.min())),
], columns=["paramètre", "valeur"]).set_index("paramètre"))

rows = []
for i in range(cfg.market.n_underlyings):
    r = {"sous_jacent": "UL_%02d" % (i+1), "spot": market.spots[i], "vol": market.vols[i]}
    for d in DIMS:
        r["G%d" % d] = "X" if i < d else ""
    rows.append(r)
display(Markdown("**Sous-jacents et appartenance aux groupes emboîtés**"))
display(pd.DataFrame(rows).set_index("sous_jacent"))

rows = []
for d in DIMS:
    g = pf.group(d)
    r = {"d": d, "n_produits": len(g), "alpha_net": sum(x.nominal for x in g),
         "méthode proxy": cfg.proxies.methods[d]}
    for t in cfg.portfolio.option_types:
        r["B0_%s" % OPTION_TYPES[t]] = basket_level(market, d, t)
    p = cfg.proxies.params_for(d)
    if cfg.proxies.methods[d] == "full_cheb":
        r["noeuds"] = (p.degree + 1) ** d
    else:
        r["noeuds"] = p.n_train
    r["grille_pleine"] = "%.2e" % float((p.degree + 1) ** d)
    rows.append(r)
display(Markdown("**Groupes**"))
display(pd.DataFrame(rows).set_index("d"))

rows = []
for dv in sorted(pf, key=lambda x: x.derivative_id):
    b0 = basket_level(market, dv.dimension, dv.option_type)
    rows.append(dict(id=dv.derivative_id, d=dv.dimension, type=OPTION_TYPES[dv.option_type],
                     sens="Call" if dv.callput_flag == 0 else "Put", B_0=b0,
                     strike=dv.strike, moneyness=dv.strike/b0, maturite=dv.maturity,
                     alpha=dv.nominal))
display(Markdown("**Produits (100 lignes)**"))
display(pd.DataFrame(rows).set_index("id"))
""")

md(r"""
## 3. Scénarios (un seul jeu, caché) et boîtes d'approximation

Boîte du groupe $d$ = enveloppe des scénarios **1j et 10j** sur les $d$ premiers actifs, marge 1 %.
Par construction la boîte du groupe $d$ est le préfixe de celle du groupe 25.
""")
code(r"""
G_scen, S = get_scenarios(cfg, market)
BOX = {d: empirical_box(S, d, cfg.box) for d in DIMS}

rows = []
for i in range(cfg.market.n_underlyings):
    lo, hi = BOX[max(DIMS)][i]
    rows.append(dict(actif="UL_%02d" % (i+1), S0=market.spots[i], lo=lo, hi=hi,
                     lo_pct_S0=100*(lo/market.spots[i]-1), hi_pct_S0=100*(hi/market.spots[i]-1),
                     **{"min_%s" % h: S[h][:, i].min() for h in HS},
                     **{"max_%s" % h: S[h][:, i].max() for h in HS}))
display(Markdown("**Boîte (groupe 25 ; les autres groupes en sont les préfixes)**"))
display(pd.DataFrame(rows).set_index("actif"))

for d in DIMS:
    n_out = {h: int((~inside_box(S[h][:, :d], BOX[d])).sum()) for h in HS}
    print("d=%2d : scénarios hors boîte %s | s0 dans la boîte : %s"
          % (d, n_out, bool(inside_box(market.spots[:d], BOX[d]).all())))
""")

md(r"""
## 4. Convergence de $V(s_0)$ en $M$ (pricer unitaire, même seed) — cachée

Contrôle du budget imposé de 100 000 trajectoires (pas un choix : une vérification).
""")
code(r"""
RUN_CONVERGENCE = True
M_CONV = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
if SMOKE:
    M_CONV = [1_000, 2_000, 5_000]
if RUN_CONVERGENCE:
    df_conv = FR.convergence_study(cfg, market, pf, M_CONV)
    display(df_conv)
    plt.figure(figsize=(7, 3.5))
    plt.plot(df_conv.index, df_conv["V"], marker="o")
    plt.axvline(cfg.mc.n_paths, color="r", ls="--", lw=1, label="M retenu")
    plt.xscale("log"); plt.xlabel("M"); plt.ylabel("V(s0)"); plt.grid(alpha=.3); plt.legend()
    plt.title("Convergence de V(s0) (pricer unitaire noCRN, seed %d)" % cfg.mc.seed_pricing)
    plt.show()
""")

md(r"""
## 5. LE full repricing (unitaire, noCRN, même seed) — caché, reprenable

`benchmark` mesure un pricing unitaire par produit et projette le coût total.
`run_full_repricing` ne calcule que ce qui manque ; une interruption (Ctrl-C) est reprise
exactement au dernier checkpoint. `n_workers > 1` parallélise par (étape, produit).
""")
code(r"""
bench = FR.benchmark(cfg, market, pf)
""")
code(r"""
RUN_FR_HERE = True     # False : ne rien calculer ici (ex. calcul lancé via run_all.py)
N_WORKERS = 1          # ex. os.cpu_count() pour paralléliser (temps par pricing toujours mesurés)

if RUN_FR_HERE:
    FR.run_full_repricing(cfg, market, pf, n_workers=N_WORKERS)
st = FR.status(cfg, pf)
display(st[FR.stages(cfg)].apply(lambda c: c.value_counts()).fillna(0).astype(int))
print("Cache :", FR.fr_dir(cfg))
""")

md("### 5.1 Résultats du full repricing : valeurs, temps mesurés, VaR / ES")
code(r"""
fr = FR.FullRepricing(cfg, pf)

display(Markdown("**Par produit : prix en s0, contribution, temps mesurés (somme des pricings par étape)**"))
df_prod = fr.product_table()
display(df_prod)

display(Markdown("**Par groupe : V0, part, temps mesurés**"))
df_grp = A.fr_group_table(fr, cfg)
display(df_grp)

display(Markdown("**VaR 99 % / ES 97,5 % / ES 99 % — full repricing (référence), par groupe et portefeuille**"))
df_fr = A.fr_risk_table(fr, cfg)
display(df_fr)
""")
code(r"""
fig, ax = plt.subplots(1, len(HS), figsize=(5.5*len(HS), 3.8))
for a, h in zip(np.atleast_1d(ax), HS):
    L = fr.V0() - fr.values(h)
    a.hist(L, bins=80, alpha=.7)
    for k, ls, c in [("VaR_99", "--", "r"), ("ES_97_5", ":", "k"), ("ES_99", "-.", "g")]:
        v = df_fr.loc[("portefeuille", h), k]
        a.axvline(v, color=c, ls=ls, lw=1, label="%s = %.2f" % (k, v))
    a.set_title("Portefeuille, pertes L = V(s0) - V(s_h), %s" % h)
    a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Proxies par groupe — construction (nœuds CRN) et évaluation, cachées

Si un proxy est en cache, il est relu **sans** MoCaX. Un échec de fit (ex. `SVD did not converge`)
est enregistré et affiché ; `RETRY_FAILED = True` pour réessayer.
""")
code(r"""
RETRY_FAILED = False
PROXY = {}
for d in DIMS:
    try:
        PROXY[d] = run_group_proxy(cfg, market, pf, d, S, mocaxpy, me,
                                   retry_failed=RETRY_FAILED)
    except RuntimeError as exc:
        print("d=%d : %s" % (d, exc))
""")
code(r"""
rows = []
for d, r in PROXY.items():
    b = r["build"]
    row = dict(d=d, méthode=r["method"], statut=r["status"], cache=r["root"],
               n_noeuds=b.get("n_nodes"), t_init_crn_s=b.get("t_init_crn"),
               t_noeuds_s=b.get("t_nodes"), t_fit_s=b.get("t_fit"), t_build_s=b.get("t_build"),
               mesuré_le=b.get("timestamp"), machine=b.get("host"), erreur=b.get("error"))
    info = b.get("info", {}) or {}
    row.update({k: info.get(k) for k in ("ranks", "rank_max", "saturated", "val_err",
                                         "n_train_req", "n_train", "n_train_unique",
                                         "best_restart", "restart_val_errs",
                                         "restart_errors", "err_ex_ante")})
    row["params"] = b.get("params")
    rows.append(row)
display(Markdown("**Construction des proxies : temps mesurés, rangs, erreurs**"))
display(pd.DataFrame(rows).set_index("d"))
""")

md(r"""
## 7. Proxies vs full repricing : VaR / ES, erreurs, temps, speed-ups

* `t_FR` : somme des temps **mesurés** des pricings unitaires du périmètre sur l'horizon.
* `t_eval` : évaluation du proxy sur les mêmes 10 000 scénarios.
* `speedup_online = t_FR / t_eval` ; `speedup_total = t_FR(s0+1j+10j) / (t_build + t_eval(s0+1j+10j))`.
* Portefeuille par proxy = **somme des évaluations** des proxies de groupe (un groupe en échec est
  remplacé par son full repricing et **signalé** dans le libellé).
""")
code(r"""
df_cmp, df_t, PF_PROXY = A.proxy_tables(fr, PROXY, cfg)
df_syn = A.summary_table(df_cmp, cfg)

display(Markdown("**Synthèse : mesures de risque FR vs proxy (écarts en %)**"))
display(df_syn)
display(Markdown("**Détail complet (toutes colonnes)**"))
display(df_cmp)
display(Markdown("**Temps mesurés et speed-ups**"))
display(df_t)
""")
code(r"""
RES_DIR = A.results_dir(cfg, PROXY)
A.save_all(RES_DIR, dict(fr_risque=df_fr, fr_groupes=df_grp, fr_produits=df_prod,
                         proxy_vs_fr=df_cmp, proxy_temps=df_t, synthese=df_syn))
A.plot_losses(fr, PROXY, PF_PROXY, cfg, root=RES_DIR, show=True)
print("Tableaux (CSV + JSON) et figures écrits dans", RES_DIR)
""")

md("## 8. Inventaire du cache")
code(r"""
rows = []
for dirpath, _dirs, files in os.walk(cfg.run.cache_dir):
    if not files:
        continue
    size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in files)
    rows.append(dict(dossier=os.path.relpath(dirpath, cfg.run.cache_dir),
                     n_fichiers=len(files), taille_Mo=size / 1e6))
df_inv = pd.DataFrame(rows).set_index("dossier")
display(df_inv.groupby(lambda p: "/".join(p.split(os.sep)[:2])).sum())
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                "language": "python"}
nbf.write(nb, OUT)
print("ecrit :", OUT)
