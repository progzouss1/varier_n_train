# -*- coding: utf-8 -*-
"""Balayage n_train R1 / R2 en ligne de commande (meme cache que le notebook).

  nohup python run_sweep.py --regime R1 --workers 8 > sweep_R1.log 2>&1 &
  nohup python run_sweep.py --regime R2 --workers 8 > sweep_R2.log 2>&1 &
  python run_sweep.py --regime R2 --groups 25 --horizons 10j --levels 0 1
  python run_sweep.py --regime R1 --retry-failed      # relance les timeout / echecs
  python run_sweep.py --regime R1 --fit-timeout 0     # 0 = pas de plafond

Relancer la meme commande reprend la ou elle s'est arretee.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=["R1", "R2"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fit-timeout", type=float, default=7200.0)
    ap.add_argument("--groups", type=int, nargs="*")
    ap.add_argument("--horizons", nargs="*")
    ap.add_argument("--levels", type=int, nargs="*", help="indices 0..4 dans la grille n_train")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mocax-root", default=os.path.join(os.path.expanduser("~"), "mocax"))
    a = ap.parse_args(argv)
    if a.workers > 1:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")

    from dataclasses import replace
    import full_repricing as FR
    import sweep as SW
    from config import ExperimentConfig
    from mocax_extend_loader import load_mocax_extend
    from portfolio import build_market, build_portfolio
    from risk import get_scenarios

    cfg = ExperimentConfig.smoke() if a.smoke else ExperimentConfig().validate()
    sw = SW.SweepConfig.smoke() if a.smoke else SW.SweepConfig()
    sw = replace(sw, n_workers=a.workers, fit_timeout=a.fit_timeout or None)
    market = build_market(cfg.market)
    pf = build_portfolio(cfg.portfolio, market)
    _, S = get_scenarios(cfg, market)
    if not FR.is_complete(cfg, pf):
        sys.exit("Full repricing incomplet dans %s" % FR.fr_dir(cfg))
    me = load_mocax_extend(search_roots=[a.mocax_root])
    SW.run_sweep(cfg, sw, market, pf, S, me, a.regime, groups=a.groups,
                 horizons=a.horizons, levels=a.levels, retry_failed=a.retry_failed)


if __name__ == "__main__":
    main()
