# Projet VaR / ES : full repricing vs proxies Chebyshev / TT

## Fichiers

| Fichier | Rôle |
|---|---|
| `config.py` | Tous les paramètres (marché, portefeuille, scénarios, M, seeds, boîte, proxies, cache) |
| `model.py`, `portfolio.py` | Produits, marché, construction du portefeuille (inchangés) |
| `pricing.py` | `UnitPricer` (FO, noCRN, même seed) et `GroupPricer` (CRN, nœuds des proxies) |
| `risk.py` | Scénarios (cachés), boîte enveloppe, VaR / ES, erreurs |
| `full_repricing.py` | LE full repricing : tâches (étape × produit), checkpoints, parallélisme, lecture, benchmark, convergence |
| `proxies.py` | Full Chebyshev / TT par groupe : nœuds CRN cachés, fit, évaluation cachée |
| `mocax_tt.py` | Enrobage MoCaXExtend (sous-grille, fit avec restarts protégés, eval sûre) |
| `analysis.py` | Tableaux de comparaison, temps, speed-ups, sauvegarde CSV/JSON, figures |
| `run_all.py` | Pipeline en ligne de commande (mêmes fonctions, même cache) |
| `experience_portefeuille.ipynb` | Notebook de l'expérience |
| `tests/` | `smoke_test.py` (bout en bout, fausse MoCaX), `fake_mocax.py`, `build_notebook.py` |
| `CHANGEMENTS.md` | Écarts avec la version précédente |

## Marche à suivre

```bash
python tests/smoke_test.py                         # 30 s : vérifie la chaîne (fausse MoCaX)
python run_all.py --stage bench                    # coût mesuré sur votre machine
nohup python run_all.py --stage fr --workers 8 > fr.log 2>&1 &   # LE full repricing (long)
python run_all.py --stage status                   # avancement (reprenable à tout moment)
python run_all.py --stage proxies                  # nœuds CRN + fits + évaluations
python run_all.py --stage results                  # tableaux + figures
```

Puis ouvrir le notebook : il relit tout depuis `cache/` (aucun recalcul).
Les tableaux finaux sont dans `cache/results/<empreinte>/` (le dernier est indiqué dans
`cache/results/LATEST.json`).

## Où sont les résultats

```
cache/
  scenarios/<fp>/                G.npy, S_1j.npy, S_10j.npy, S0.npy, manifest.json
  full_repricing/<fp>/           manifest.json, benchmark_n3.json
     s0/ 1j/ 10j/                pNNN.npy (prix unitaires), pNNN_times.npy (temps de chaque pricing), pNNN.json
  proxies/dNN_<methode>_<fp>/    nodes.npy, node_values.npy, node_times.npy, build.json,
                                 fit_log.txt, obj.pkl, eval_s0.npy, eval_1j.npy, eval_10j.npy, eval.json
  convergence/<fp>.json
  results/<fp>/                  synthese, proxy_vs_fr, proxy_temps, fr_risque, fr_groupes,
                                 fr_produits (.csv + .json), pertes_*.png
```

## Parallélisme

`--workers N` lance N processus (une tâche = un produit sur une étape). Le temps de **chaque**
pricing reste mesuré dans son processus ; `t_FR` = somme de ces temps (équivalent séquentiel).
Attention : sous forte charge, les processus se partagent la bande passante mémoire, donc le temps
unitaire mesuré peut être un peu plus élevé qu'en séquentiel. Pour des temps « FO » stricts,
utiliser `--workers 1`.

## Balayage `n_train` R1 / R2 (`experience_sweep_ntrain.ipynb`)

Un TT par groupe **et par horizon** (boîte = enveloppe des scénarios de l'horizon seul), d=5…25,
nœuds pricés en **unitaire noCRN** : R1 = même seed à chaque pricing, R2 = seed différente à
chaque pricing (pricer FO). Partie A = R1, partie B = R2, partie C = comparaison.

```bash
python tests/smoke_sweep.py                                          # 2 min, fausse MoCaX
nohup python run_sweep.py --regime R1 --workers 8 > sweep_R1.log 2>&1 &
nohup python run_sweep.py --regime R2 --workers 8 > sweep_R2.log 2>&1 &
```
Puis ouvrir le notebook (`RUN_R1 = RUN_R2 = False` pour ne faire que relire le cache).
Résultats : `cache/sweep/results/{R1,R2,comparaison}/` (CSV, JSON, PNG).
