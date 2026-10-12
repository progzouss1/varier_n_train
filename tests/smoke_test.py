# -*- coding: utf-8 -*-
"""Test de bout en bout en mode reduit, avec les FAUSSES MoCaX.

  python tests/smoke_test.py

Verifie : cache des scenarios, full repricing reprenable (interruption
simulee), 2e passage = 0 recalcul, coherence CRN-groupe / somme unitaire,
boite = enveloppe, tableaux et fichiers de resultats.
"""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np

import analysis as A
import full_repricing as FR
from config import ExperimentConfig
from fake_mocax import fake_mocax_extend, fake_mocaxpy
from portfolio import build_market, build_portfolio, check_portfolio
from pricing import GroupPricer, UnitPricer
from proxies import run_group_proxy
from risk import empirical_box, get_scenarios, inside_box

CACHE = os.path.join(HERE, "_cache_smoke")
shutil.rmtree(CACHE, ignore_errors=True)
cfg = ExperimentConfig.smoke(cache_dir=CACHE)
m = build_market(cfg.market)
pf = build_portfolio(cfg.portfolio, m)
check_portfolio(cfg, m, pf)
G, S = get_scenarios(cfg, m)

# ---- 1) interruption simulee au milieu d'une tache, puis reprise ---------
root = FR._write_manifest(cfg, m, pf)
dv = pf[1]
calls = {"n": 0}
orig = UnitPricer.__call__


def boom(self, p, a=None):
    calls["n"] += 1
    if calls["n"] == 120:
        raise KeyboardInterrupt
    return orig(self, p, a)


UnitPricer.__call__ = boom
try:
    FR.run_full_repricing(cfg, m, pf, stages_to_run=["1j"], product_ids=[1])
except KeyboardInterrupt:
    print("\n-> interruption simulee")
UnitPricer.__call__ = orig
st = FR.status(cfg, pf)
assert st.loc[1, "1j"].startswith("partiel 100"), st.loc[1, "1j"]
FR.run_full_repricing(cfg, m, pf, stages_to_run=["1j"], product_ids=[1])
v = np.load(os.path.join(root, "1j", "p001.npy"))
ref = np.array([UnitPricer(dv, m, cfg.mc.n_paths, cfg.mc.seed_pricing)(s[:5])
                for s in S["1j"]])
assert np.allclose(v, ref, rtol=0, atol=1e-12), "reprise != calcul direct"
print("OK reprise exacte apres interruption")

# ---- 2) full repricing complet (2 workers) puis 2e passage = cache -------
FR.run_full_repricing(cfg, m, pf, n_workers=2)
assert FR.is_complete(cfg, pf)
t0 = time.time()
FR.run_full_repricing(cfg, m, pf)
assert time.time() - t0 < 5
fr = FR.FullRepricing(cfg, pf)
print("OK full repricing en cache, V0 = %.4f" % fr.V0())

# ---- 3) coherence CRN groupe (meme seed) = somme des unitaires -----------
for d in cfg.portfolio.dimensions:
    gp = GroupPricer(pf.group(d), m, cfg.mc.n_paths, cfg.mc.crn_seed)
    for k in (0, 7):
        assert abs(gp(S["10j"][k, :d]) - fr.values("10j", d)[k]) < 1e-9
print("OK noeud CRN == somme des pricings unitaires (seed identique)")

# ---- 4) boite = enveloppe ------------------------------------------------
for d in cfg.portfolio.dimensions:
    b = empirical_box(S, d, cfg.box)
    assert all(inside_box(S[h][:, :d], b).all() for h in S)
    assert inside_box(m.spots[:d], b).all()
print("OK tous les scenarios (et s0) dans la boite")

# ---- 5) proxies (fausses libs) + 2e passage sans libs --------------------
mp, me = fake_mocaxpy(), fake_mocax_extend()
res = {d: run_group_proxy(cfg, m, pf, d, S, mp, me) for d in cfg.portfolio.dimensions}
res2 = {d: run_group_proxy(cfg, m, pf, d, S, None, None) for d in cfg.portfolio.dimensions}
for d in res:
    assert np.array_equal(res[d]["V"]["1j"], res2[d]["V"]["1j"])
print("OK proxies relus du cache sans MoCaX")

# ---- 6) tableaux ---------------------------------------------------------
df_cmp, df_t, pfp = A.proxy_tables(fr, res2, cfg)
out = A.results_dir(cfg, res2)
A.save_all(out, dict(fr_risque=A.fr_risk_table(fr, cfg), proxy_vs_fr=df_cmp,
                     proxy_temps=df_t, synthese=A.summary_table(df_cmp, cfg)))
import matplotlib
matplotlib.use("Agg")
A.plot_losses(fr, res2, pfp, cfg, root=out, show=False)
print(A.summary_table(df_cmp, cfg).round(3).to_string())
print(sorted(os.listdir(out)))
# d=5 Full CT deg 3 sur vraie interpolation : erreur doit etre petite
r5 = df_cmp.loc[("d=5", "1j")]
print("d=5 CT (fausse lib mais vraie interpolation) : dVaR = %.3f %%" % r5["dVaR_99_pct"])
print("\nSMOKE TEST OK")
