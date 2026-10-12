# -*- coding: utf-8 -*-
"""Test du balayage R1 / R2 en mode reduit (FAUSSE MoCaX : resultats sans valeur).

  python tests/smoke_sweep.py
"""
import os
import shutil
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np

import full_repricing as FR
import sweep as SW
from config import ExperimentConfig
from fake_mocax import fake_mocax_extend
from portfolio import build_market, build_portfolio
from pricing import UnitPricer
from risk import get_scenarios, inside_box

CACHE = os.path.join(HERE, "_cache_sweep")
shutil.rmtree(CACHE, ignore_errors=True)
cfg = ExperimentConfig.smoke(cache_dir=CACHE)
sw = SW.SweepConfig.smoke()
m = build_market(cfg.market)
pf = build_portfolio(cfg.portfolio, m)
G, S = get_scenarios(cfg, m, verbose=False)
FR.run_full_repricing(cfg, m, pf, n_workers=2, verbose=False)
fr = FR.FullRepricing(cfg, pf)
me = fake_mocax_extend()

# ---- boites par horizon -------------------------------------------------
b1, b10 = SW.horizon_box(cfg, S, 25, "1j"), SW.horizon_box(cfg, S, 25, "10j")
assert np.all(np.ptp(b1, axis=1) < np.ptp(b10, axis=1))
assert inside_box(S["1j"][:, :25], b1).all() and inside_box(m.spots, b1).all()
print("OK boite 1j plus etroite que 10j, scenarios et s0 dedans")

# ---- R1 : interruption au milieu du pricing des noeuds, puis reprise -----
calls = {"n": 0}
orig = UnitPricer.__call__


def boom(self, p, a=None, seed=None):
    calls["n"] += 1
    if calls["n"] == 30 * 20:
        raise KeyboardInterrupt
    return orig(self, p, a, seed)


UnitPricer.__call__ = boom
try:
    SW.run_sweep(cfg, sw, m, pf, S, me, "R1", groups=[10], horizons=["1j"], levels=[1])
except KeyboardInterrupt:
    print("\n-> interruption simulee")
UnitPricer.__call__ = orig
SW.run_sweep(cfg, sw, m, pf, S, me, "R1")
print("OK R1 complet (avec reprise)")

# ---- R2 en parallele -----------------------------------------------------
SW.run_sweep(cfg, replace(sw, n_workers=2), m, pf, S, me, "R2")
print("OK R2 complet (2 processus)")

# ---- controles R1 / R2 ---------------------------------------------------
d, h, n = 10, "1j", sw.n_train[10][1]
r1 = SW.load_tt(cfg, sw, S, "R1", d, h, n)
r2 = SW.load_tt(cfg, sw, S, "R2", d, h, n)
pts = np.load(os.path.join(SW.subgrid_dir(cfg, d, h, sw.tt_for(d, n),
                                          SW.horizon_box(cfg, S, d, h)), "nodes.npy"))
ref = [sum(dv.nominal * UnitPricer(dv, m, cfg.mc.n_paths, cfg.mc.seed_pricing)(p)
           for dv in pf.group(d)) for p in pts[:5]]
assert np.allclose(r1["node_values"][:5], ref, atol=1e-10), "R1 != pricing meme seed"
assert r1["nodes"]["n_distinct_seeds"] == 1
assert r2["nodes"]["n_distinct_seeds"] == n * 20, r2["nodes"]["n_distinct_seeds"]
assert not np.allclose(r1["node_values"], r2["node_values"])
U2 = np.load(os.path.join(r2["root"], "node_unit_prices.npy"))
SD = np.load(os.path.join(r2["root"], "node_seeds.npy"))
dv = pf.group(d)[3]
assert abs(UnitPricer(dv, m, cfg.mc.n_paths, 0)(pts[7], seed=int(SD[7, 3])) - U2[7, 3]) < 1e-12
print("OK R1 = meme seed partout ; R2 = %d seeds distinctes, rejouables" % (n * 20))

# ---- 2e passage : tout en cache -----------------------------------------
t0 = time.time()
SW.run_sweep(cfg, sw, m, pf, S, None, "R1", verbose=False)
SW.run_sweep(cfg, sw, m, pf, S, None, "R2", verbose=False)
assert time.time() - t0 < 5
print("OK 2e passage sans calcul et sans MoCaX")

# ---- plafond de temps du fit --------------------------------------------
me_slow = fake_mocax_extend()
_o = me_slow.MocaxExtend.run_rank_adaptive_algo
me_slow.MocaxExtend.run_rank_adaptive_algo = lambda self, **kw: (time.sleep(30), _o(self, **kw))
sw_t = replace(sw, fit_timeout=3.0, n_train={**sw.n_train, 5: (55,)})
root = SW.run_one(cfg, sw_t, m, pf, S, me_slow, "R2", 5, "1j", 55)
f = SW.load_tt(cfg, sw_t, S, "R2", 5, "1j", 55)["fit"]
assert f["status"] == "timeout", f
print("OK fit trop long -> statut timeout enregistre")

# ---- tableaux ------------------------------------------------------------
import pandas as pd
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
t1 = SW.sweep_table(cfg, sw, fr, S, "R1")
t2 = SW.sweep_table(cfg, sw, fr, S, "R2")
assert len(t1) == len(t2) == 5 * 2 * 2
print(t2[["d", "horizon", "n_train", "statut", "t_nodes_s", "t_fit_s", "rang_max",
          "dVaR_99_pct", "dES_99_pct", "bruit_noeuds_rel_std"]].round(4).to_string())
print(SW.portfolio_table(cfg, sw, fr, S, "R1")[["horizon", "niveau", "dVaR_99_pct",
                                              "groupes_en_repli_FR"]])
print(SW.compare_regimes(t1, t2)[["ratio_t_fit_R2_R1", "rang_max_R1", "rang_max_R2"]].head())
import matplotlib
matplotlib.use("Agg")
SW.plot_sweep(t1, cfg, "R1", root=SW.results_dir(cfg, "R1"), show=False)
SW.plot_losses_max(cfg, sw, fr, S, "R2", root=SW.results_dir(cfg, "R2"), show=False)
print("\nSMOKE SWEEP OK")
