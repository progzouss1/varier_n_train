# -*- coding: utf-8 -*-
"""Genere experience_sweep_ntrain.ipynb (python tests/build_sweep_notebook.py)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "experience_sweep_ntrain.ipynb")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Balayage de `n_train` des proxies Tensor-Train — R1 (noCRN même seed) vs R2 (noCRN seed variable)

**Protocole**

| Élément | Choix |
|---|---|
| Proxy | **un TT par groupe et par horizon** ($d = 5, 10, 15, 20, 25$ ; 1j et 10j), y compris $d=5$ |
| Boîte | enveloppe des scénarios **de l'horizon seul** : $\text{lo}_i = 0{,}99\min_k s^{(k)}_{h,i}$, $\text{hi}_i = 1{,}01\max_k s^{(k)}_{h,i}$ |
| Pricing des nœuds | **unitaire** (pricer front office : 20 pricings par nœud, un par instrument), **noCRN**, 100 000 trajectoires |
| Paramètre étudié | `n_train` seul (5 valeurs par groupe, en baisse depuis le maximum) ; tous les autres hyperparamètres fixés |
| Référence | **le** full repricing unitaire noCRN même seed, relu du cache (rien n'est recalculé) |

**Les deux régimes de pricing des nœuds** (deux parties distinctes du notebook)

* **Partie A — R1, noCRN même seed** : à chaque pricing, le générateur est recréé avec la seed fixe (comme le full repricing). Cas d'un pricer dont on contrôle la seed.
* **Partie B — R2, noCRN seed variable** : chaque pricing (nœud × produit) a sa propre seed, comme un pricer front office qui ne donne pas accès à la seed. Les valeurs aux nœuds portent donc chacune leur propre bruit Monte-Carlo. Les seeds sont dérivées de (d, horizon, n_train, nœud, produit) : différentes d'un pricing à l'autre, identiques d'une exécution à l'autre (sinon le cache n'aurait pas de sens).

Pour un même (d, horizon, n_train), R1 et R2 pricent **la même sous-grille de points** : seule la seed change. Chaque TT price **ses propres** nœuds, et le temps est mesuré.

**Garde-fous**

* *Plafond de temps du fit* (`FIT_TIMEOUT`) : l'ALS tourne dans un processus séparé. S'il dépasse le plafond, il est arrêté et le TT est enregistré avec le statut **timeout** et le temps atteint : la ligne reste dans les tableaux au lieu de bloquer le notebook pendant des heures. `FIT_TIMEOUT = None` désactive le plafond.
* *Reprise* : le pricing des nœuds est sauvegardé par paquets de 100 nœuds. Après une interruption, relancer la cellule reprend au dernier paquet ; un TT terminé n'est jamais recalculé.
""")

md("## 0. Imports, MoCaX")
code(r"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import replace
from IPython.display import display, Markdown

from config import ExperimentConfig
from portfolio import build_market, build_portfolio, check_portfolio
from risk import get_scenarios
from cache import save_df
import full_repricing as FR
import sweep as SW

pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
""")
code(r"""
MOCAX_ROOT = os.path.join(os.path.expanduser("~"), "mocax")
me = None
try:
    from mocax_extend_loader import load_mocax_extend
    me = load_mocax_extend(search_roots=[MOCAX_ROOT])
except Exception as exc:
    print("MoCaXExtend non chargé (%s) -> lecture du cache uniquement." % exc)
""")

md("## 1. Configuration")
code(r"""
SMOKE = False                 # True : mini-run de vérification (cache séparé)
N_WORKERS = 1                 # processus pour le pricing des noeuds (ex. os.cpu_count())
FIT_TIMEOUT = 7200.0          # plafond du fit ALS en secondes (None = aucun)

cfg = ExperimentConfig.smoke() if SMOKE else ExperimentConfig().validate()
sw = SW.SweepConfig.smoke() if SMOKE else SW.SweepConfig()
sw = replace(sw, n_workers=N_WORKERS, fit_timeout=FIT_TIMEOUT)
DIMS, HS = list(cfg.portfolio.dimensions), list(cfg.risk.horizons)

print(cfg.summary().split("Proxy d=")[0])
display(Markdown("**Hyperparamètres TT fixés (seul `n_train` varie)**"))
display(pd.DataFrame([dict(d=d, degre=t.degree, pts_par_dim=t.points_per_dim,
                           grille_pleine="%.2e" % t.full_grid_size(d),
                           n_train=list(sw.n_train[d]), max_rank=t.max_rank,
                           tolerance=t.tolerance, rel_tolerance=t.rel_tolerance,
                           max_iters=t.max_iters, train_frac=t.train_frac,
                           n_restarts=t.n_restarts, seed_fit=t.seed_fit,
                           seed_subgrid=t.seed_subgrid)
                      for d, t in sw.tt.items()]).set_index("d"))
""")

md("## 2. Marché, scénarios, full repricing de référence (cache), boîtes par horizon")
code(r"""
market = build_market(cfg.market)
pf = build_portfolio(cfg.portfolio, market); check_portfolio(cfg, market, pf)
G_scen, S = get_scenarios(cfg, market)
if not FR.is_complete(cfg, pf):
    raise RuntimeError("Full repricing absent/incomplet du cache %s : lancer d'abord "
                       "experience_portefeuille.ipynb ou run_all.py --stage fr" % FR.fr_dir(cfg))
fr = FR.FullRepricing(cfg, pf)
print("Full repricing relu :", fr.root, "| V0 = %.4f" % fr.V0())

rows = []
for d in DIMS:
    for h in HS:
        b = SW.horizon_box(cfg, S, d, h)
        rows.append(dict(d=d, horizon=h, largeur_rel_moy_pct=100*np.mean((b[:,1]-b[:,0])/market.spots[:d]),
                         lo_min_pct=100*np.min(b[:,0]/market.spots[:d]-1),
                         hi_max_pct=100*np.max(b[:,1]/market.spots[:d]-1),
                         t_FR_1scenario_s=fr.time_per_scenario(h, d)))
display(Markdown("**Boîtes par horizon (largeur relative au spot) et coût mesuré d'un scénario du full repricing**"))
display(pd.DataFrame(rows).set_index(["d", "horizon"]))
""")
code(r"""
# Estimation du coût du pricing des noeuds (1 noeud = 1 scénario FR du groupe, temps mesurés)
rows = []
for d in DIMS:
    for h in HS:
        n_tot = sum(sw.n_train[d])
        rows.append(dict(d=d, horizon=h, n_noeuds_total=n_tot,
                         t_estime_par_regime_h=n_tot * fr.time_per_scenario(h, d) / 3600))
df_cost = pd.DataFrame(rows).set_index(["d", "horizon"])
display(df_cost)
print("Pricing des noeuds estimé : %.1f h par régime (séquentiel), %.1f h pour R1 + R2 ; "
      "hors fits." % (df_cost.t_estime_par_regime_h.sum(), 2 * df_cost.t_estime_par_regime_h.sum()))
""")

def part(reg, letter, title, extra=""):
    md(r"""
---
# Partie %s — %s : %s
%s""" % (letter, reg, title, extra))
    md("## %s.1 Construction des TT (pricing des nœuds, fit, évaluation) — caché, reprenable" % letter)
    code(r"""
RUN_%s = True       # False : lecture du cache uniquement
if RUN_%s:
    SW.run_sweep(cfg, sw, market, pf, S, me, "%s")
""" % (reg, reg, reg))
    md("## %s.2 Résultats par TT : temps mesurés, rangs, VaR / ES vs full repricing" % letter)
    code(r"""
df_%s = SW.sweep_table(cfg, sw, fr, S, "%s")
RES_%s = SW.results_dir(cfg, "%s")
save_df(df_%s.set_index(["d", "horizon", "n_train"]), os.path.join(RES_%s, "par_tt"))
display(Markdown("**Synthèse**"))
cols = ["statut", "n_noeuds", "t_nodes_s", "t_noeud_moy_s", "t_fit_s", "t_eval_s",
        "rangs", "rang_max", "n_tours_rang", "val_err",
        "VaR_99_FR", "VaR_99_proxy", "dVaR_99_pct", "ES_97_5_FR", "ES_97_5_proxy",
        "dES_97_5_pct", "ES_99_FR", "ES_99_proxy", "dES_99_pct", "erel_sc_mean",
        "speedup_online", "speedup_total"]%s
display(df_%s.set_index(["d", "horizon", "n_train"])[[c for c in cols if c in df_%s]])
display(Markdown("**Tableau complet (toutes colonnes)**"))
display(df_%s.set_index(["d", "horizon", "n_train"]))
""" % (reg, reg, reg, reg, reg, reg,
       ' + ["bruit_noeuds_std", "bruit_noeuds_rel_std", "n_seeds_distinctes"]' if reg == "R2" else "",
       reg, reg, reg))
    md("## %s.3 Effet de `n_train` (courbes par groupe)" % letter)
    code('SW.plot_sweep(df_%s, cfg, "%s (%s)", root=RES_%s);' % (reg, reg, SW_LABEL[reg], reg))
    md("## %s.4 Portefeuille : somme des 5 TT au même niveau de la grille" % letter)
    code(r"""
pf_%s = SW.portfolio_table(cfg, sw, fr, S, "%s")
save_df(pf_%s.set_index(["horizon", "niveau"]), os.path.join(RES_%s, "portefeuille"))
display(pf_%s.set_index(["horizon", "niveau"]))
""" % (reg, reg, reg, reg, reg))
    md("## %s.5 Distributions des pertes au `n_train` maximal" % letter)
    code('SW.plot_losses_max(cfg, sw, fr, S, "%s", root=RES_%s);\nprint("Résultats %s écrits dans", RES_%s)'
         % (reg, reg, reg, reg))


SW_LABEL = {"R1": "noCRN même seed", "R2": "noCRN seed variable"}
part("R1", "A", "noCRN, même seed à chaque pricing des nœuds",
     "Valeurs aux nœuds = exactement la fonction du full repricing : l'écart TT / FR "
     "est l'erreur d'interpolation seule.")
part("R2", "B", "noCRN, seed différente à chaque pricing des nœuds (pricer front office)",
     "Valeurs aux nœuds = fonction du full repricing **+ bruit Monte-Carlo propre à chaque "
     "pricing**. Colonnes `bruit_noeuds_*` : écart R2 − R1 aux mêmes nœuds (bruit "
     "effectivement injecté dans le fit).")

md("""
---
# Partie C — Comparaison R1 / R2 à (d, horizon, n_train) égal
""")
code(r"""
df_cmp = SW.compare_regimes(df_R1, df_R2)
save_df(df_cmp, os.path.join(SW.results_dir(cfg, "comparaison"), "R1_vs_R2"))
display(df_cmp)
""")
code(r"""
fig, axes = plt.subplots(len(HS), 4, figsize=(18, 4 * len(HS)))
for row, h in zip(np.atleast_2d(axes), HS):
    for ax, (col, lab, log) in zip(row, [("t_fit_s", "t fit ALS (s)", True),
                                        ("dVaR_99_pct", "Écart VaR 99 % (%)", False),
                                        ("dES_99_pct", "Écart ES 99 % (%)", False),
                                        ("rang_max", "Rang max", False)]):
        for d in DIMS:
            for reg, df, ls in [("R1", df_R1, "-"), ("R2", df_R2, "--")]:
                g = df[(df.d == d) & (df.horizon == h)].sort_values("n_train")
                if col in g:
                    ax.plot(g.n_train, g[col], ls, marker="o", ms=3,
                            label="d=%d %s" % (d, reg))
        if log: ax.set_yscale("log")
        if col.endswith("_pct"): ax.axhline(0, color="k", lw=.6)
        ax.set_title("%s — %s (trait plein R1, tirets R2)" % (lab, h), fontsize=9)
        ax.set_xlabel("n_train"); ax.grid(alpha=.3)
np.atleast_2d(axes)[0, 0].legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(SW.results_dir(cfg, "comparaison"), "R1_vs_R2.png"), dpi=110)
plt.show()
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print("ecrit :", OUT)
