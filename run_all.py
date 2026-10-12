# -*- coding: utf-8 -*-
"""Pipeline complet en ligne de commande (pour les longs calculs, hors Jupyter).

Tout est cache et reprenable : relancer la meme commande reprend la ou elle
s'est arretee (Ctrl-C, coupure, reboot) sans jamais refaire un pricing termine.

  python run_all.py --stage status              # etat du cache
  python run_all.py --stage bench               # cout d'un pricing, projection
  python run_all.py --stage fr --workers 8      # LE full repricing (long)
  python run_all.py --stage proxies             # noeuds CRN + fits + evals
  python run_all.py --stage proxies --groups 10 15 --retry-failed
  python run_all.py --stage results             # tableaux + figures (CSV/JSON/PNG)
  python run_all.py --stage all
  python run_all.py --smoke --stage all         # mini-run de verification

Conseil pour le full repricing : lancer sous nohup / tmux / screen, ex.
  nohup python run_all.py --stage fr --workers 8 > fr.log 2>&1 &
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["status", "bench", "fr", "proxies", "results", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--groups", type=int, nargs="*", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--mocax-root", default=os.path.join(os.path.expanduser("~"), "mocax"))
    a = ap.parse_args(argv)

    if a.workers and a.workers > 1:                 # 1 thread BLAS par processus
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")

    from dataclasses import replace
    from config import ExperimentConfig
    from portfolio import build_market, build_portfolio, check_portfolio
    import full_repricing as FR
    from risk import get_scenarios

    cfg = ExperimentConfig.smoke() if a.smoke else ExperimentConfig().validate()
    if a.cache_dir:
        cfg = replace(cfg, run=replace(cfg.run, cache_dir=a.cache_dir))
    if a.workers:
        cfg = replace(cfg, run=replace(cfg.run, n_workers=a.workers))
    print(cfg.summary())

    market = build_market(cfg.market)
    pf = build_portfolio(cfg.portfolio, market)
    check_portfolio(cfg, market, pf)
    G, S = get_scenarios(cfg, market)
    groups = a.groups or list(cfg.portfolio.dimensions)

    if a.stage in ("status",):
        st = FR.status(cfg, pf)
        print(st[FR.stages(cfg)].apply(lambda c: c.value_counts()).fillna(0).astype(int))
        return

    if a.stage in ("bench", "all"):
        FR.benchmark(cfg, market, pf)

    if a.stage in ("fr", "all"):
        FR.run_full_repricing(cfg, market, pf)

    proxy_results = {}
    if a.stage in ("proxies", "all", "results"):
        from proxies import run_group_proxy
        mocaxpy = me = None
        if a.stage != "results":
            from mocax_loader import load_mocax
            from mocax_extend_loader import load_mocax_extend
            need = {cfg.proxies.methods[d] for d in groups}
            if "full_cheb" in need:
                mocaxpy = load_mocax(search_roots=[a.mocax_root])
            if "tt" in need:
                me = load_mocax_extend(search_roots=[a.mocax_root])
        for d in groups:
            try:
                proxy_results[d] = run_group_proxy(cfg, market, pf, d, S, mocaxpy,
                                                   me, retry_failed=a.retry_failed)
            except RuntimeError as exc:
                print("Proxy d=%d indisponible : %s" % (d, exc))

    if a.stage in ("results", "all"):
        import analysis as A
        if not FR.is_complete(cfg, pf):
            print("Full repricing incomplet -> pas de tableaux comparatifs.")
            return
        fr = FR.FullRepricing(cfg, pf)
        df_fr = A.fr_risk_table(fr, cfg)
        df_grp = A.fr_group_table(fr, cfg)
        df_cmp, df_t, pfp = A.proxy_tables(fr, proxy_results, cfg)
        root = A.results_dir(cfg, proxy_results)
        A.save_all(root, dict(fr_risque=df_fr, fr_groupes=df_grp,
                              fr_produits=fr.product_table(),
                              proxy_vs_fr=df_cmp, proxy_temps=df_t,
                              synthese=A.summary_table(df_cmp, cfg)))
        import matplotlib
        matplotlib.use("Agg")
        A.plot_losses(fr, proxy_results, pfp, cfg, root=root, show=False)
        import pandas as pd
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(A.summary_table(df_cmp, cfg).round(4))
            print(df_t.round(3))
        print("Resultats ecrits dans %s" % root)


if __name__ == "__main__":
    main()
