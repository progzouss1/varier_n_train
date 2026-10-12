# Écarts entre le projet précédent et les nouvelles exigences

## A. Full repricing (référence)

| # | Projet précédent | Nouvelle version |
|---|---|---|
| A1 | Régime **CRN** : `GroupPricer(..., "CRN", seed)`, paquet de normales tiré **une fois** à l'init puis réutilisé. | Régime **noCRN** systématique : `UnitPricer` **re-crée** le générateur et **re-tire** les M×d normales (et refait le Cholesky) **à chaque pricing**. |
| A2 | Pricing **agrégé par groupe** : les 20 produits d'un groupe évalués sur les mêmes trajectoires (1 simulation pour 20 produits). | Pricing **unitaire** (comme un pricer FO) : 1 simulation par instrument, soit **100 pricings par scénario**. Le prix unitaire $\Pi_j$ est stocké ; la position $\alpha_j$ est appliquée à l'agrégation. |
| A3 | Seeds différentes selon la cellule : `seed_gt+100+d` (§4, §5), `98875+d` (groupes), `seed+d` (`portfolio_value_exact`). | **Une seule seed** (`mc.seed_pricing`) pour **tous** les pricings noCRN (full repricing, V0, convergence). |
| A4 | **Plusieurs** full repricings incohérents entre eux : portefeuille (§4, M=10 000), groupe d=5 (§5, M=100 000, **interrompu**), puis un MC de référence **refait dans chaque** expérience de groupe (d=10…25). | **Un seul** full repricing : 10 000 chocs × 2 horizons × 100 produits (+ s0). Les références de groupe **et** du portefeuille sont des sommes de ces pricings unitaires. |
| A5 | Scénarios différents d'une cellule à l'autre : le portefeuille tirait (N, 25) ; chaque groupe re-tirait (N, d) avec la **même** seed 3467, ce qui donne des chocs **différents** (le remplissage ligne par ligne diffère). Les VaR de groupe n'étaient donc pas des sous-ensembles cohérents de la VaR portefeuille. | **Un seul** paquet G (N, 25) ~ N(0, C₂₅), le **même** pour 1j et 10j ; le groupe d lit le préfixe `s[:, :d]`. Tout est comparable scénario par scénario. |
| A6 | Budget : **10 000** trajectoires (rapport §1.5 et cellules exécutées) ; `config.py` contenait 3 budgets non utilisés (`m_gt=1e6`, `m_build=2e5`, `m_speed=2e4`) ; l'en-tête du notebook annonçait `M_BUILD=M_SPEED=200 000`. | **100 000** trajectoires pour **tous** les pricings (`mc.n_paths`), budgets obsolètes supprimés. |
| A7 | Temps du MC : `t_mc_1` estimé sur 10 pricings puis **extrapolé** ; en cas de cache, la boucle était « projetée » (le vrai temps était perdu). | Le temps de **chaque** pricing unitaire est **mesuré et stocké** (`*_times.npy`, + horodatage et machine). Tous les temps affichés sont mesurés. |

## B. Mesures de risque

| # | Avant | Maintenant |
|---|---|---|
| B1 | VaR 99 % et ES 97,5 % uniquement. | **VaR 99 %, ES 97,5 %, ES 99 %**, calculées **séparément pour 1j et 10j**, pour chaque groupe **et** pour le portefeuille. |
| B2 | Pas de VaR du portefeuille par proxy (`assemble_portfolio_value` existait mais n'était pas utilisé). | VaR/ES du portefeuille par **somme des évaluations** des 5 proxies, comparée au full repricing. |

## C. Proxies

| # | Avant | Maintenant |
|---|---|---|
| C1 | Nœuds pricés en CRN à **10 000** trajectoires, avec **le même pricer** que la référence (CRN). | Nœuds pricés en **CRN** (`GroupPricer`, paquet figé) à **100 000** trajectoires ; la référence est désormais le full repricing **unitaire noCRN**. |
| C2 | Boîte : `config.BoxConfig` décrivait encore une boîte *vol-scaled* (z=3σ, h_dom) et l'en-tête du notebook des « points de test uniformes TEST25 » (obsolètes). Les cellules exécutées utilisaient l'enveloppe **en y ajoutant s0**, et **chaque groupe** avec ses propres scénarios (A5). | Boîte = enveloppe empirique **exacte** de la spécification : $0{,}99\min_{k,h}$ / $1{,}01\max_{k,h}$ sur les scénarios 1j+10j communs, sans ajouter s0. On vérifie que s0 et **tous** les scénarios sont dedans. `BoxConfig` vol-scaled supprimé. |
| C3 | Sous-grille TT tirée sans graine explicite (`subgrid_by_number`) : non reproductible d'un run à l'autre. | `np.random.seed(seed_subgrid)` avant le tirage ; les **points** sont mis en cache et réinjectés par `set_subgrid`. |
| C4 | Un plantage de l'ALS (d=25 : `SVD did not converge`) arrêtait le notebook. | Chaque restart est protégé ; l'erreur est **enregistrée** (`build.json`) et **affichée** ; le groupe apparaît « ECHEC » dans les tableaux et le portefeuille par proxy est alors **signalé** « repli FR : d=[…] ». `RETRY_FAILED=True` pour réessayer. |
| C5 | Hyperparamètres TT : défauts de `config.py` (degré 6, max_rank 7, tol 1e-3, 100 itérations, 3 restarts) **différents** de ceux réellement utilisés (degré 5, max_rank 10/10/8/8, tol 1e-4, 20 itérations, 1 restart), ceux-ci étant redéfinis dans chaque cellule. | Les valeurs **réellement utilisées** deviennent les défauts, **par groupe**, dans `ProxyConfig.tt`. |
| C6 | Speed-ups comparés à un MC CRN de groupe à M=10 000. | Speed-ups comparés au full repricing **unitaire noCRN** à M=100 000 (coût FO), à partir des temps **mesurés**. |

## D. Cache et traçabilité

| # | Avant | Maintenant |
|---|---|---|
| D1 | Cache limité aux valeurs MC (`.npy`), nommé par N, M et seed **seulement** : si l'on changeait le marché ou le portefeuille, le cache périmé était relu **sans avertissement**. | Chaque dossier est indexé par une **empreinte** (hash) de tout ce qui détermine le résultat (marché, produits, M, seeds, horizons, boîte, hyperparamètres). Un changement de paramètre donne un nouveau dossier ; rien de périmé n'est relu. |
| D2 | Rien d'autre n'était caché : les 3 125 nœuds du Full CT étaient **re-pricés à chaque exécution**, les fits TT, les évaluations, les temps et les tableaux VaR étaient perdus au redémarrage du noyau. | **Tout** est caché : scénarios, chaque pricing unitaire (valeur + temps), nœuds (valeur + temps), sous-grilles, logs d'ALS, objet proxy (s'il est picklable), évaluations, tableaux (CSV + JSON), figures (PNG). |
| D3 | Un calcul interrompu était perdu (le `KeyboardInterrupt` de §5 a tout perdu). | **Checkpoint** tous les 250 pricings (full repricing **et** nœuds) ; reprise exacte ; écritures atomiques (jamais de fichier à moitié écrit). |
| D4 | Tout se faisait dans le notebook. | `run_all.py` (ligne de commande, lançable sous `nohup`/`tmux`, parallélisable `--workers N`) remplit **le même** cache que le notebook. |

## E. Point à connaître (conséquence directe des exigences)

* **Même seed en noCRN + même seed pour le paquet CRN des nœuds.** Avec la même seed (réglage par
  défaut, `mc.seed_crn = None`), le paquet CRN du groupe d est **exactement** celui que re-tire le
  pricer unitaire. La valeur d'un nœud est donc **égale** (à 1e-14 près, vérifié) à la somme des
  pricings unitaires au même point. Conséquences :
  - l'écart proxy / full repricing mesure l'**erreur d'interpolation seule** (même logique que le
    notebook précédent, qui utilisait le même pricer des deux côtés) ;
  - numériquement, le full repricing « noCRN, même seed » donne les **mêmes valeurs** qu'un CRN
    par produit : seule la **charge de calcul** (re-tirage à chaque appel) diffère. C'est voulu
    (coût d'un pricer FO) ;
  - pour inclure aussi l'écart de bruit MC entre les deux estimateurs, il suffit de fixer
    `MCConfig(seed_crn=<autre seed>)`. Seuls les proxies sont alors recalculés, le full repricing
    reste en cache.
* **Coût** (mesuré ici, 2 cœurs) : un repricing complet du portefeuille ≈ **4,2 s** (100 pricings
  unitaires à M=100 000) ; full repricing complet ≈ 20 001 × 4,2 s ≈ **23 h en séquentiel**. Nœuds
  CRN : ≈ 0,10 s (d=5) à 0,19 s (d=25) par nœud, soit ≈ 5 min pour le Full CT et ≈ 35 min pour les
  4 TT (hors fit ALS).
* `max_rounds` reste écrasé par MoCaXExtend (5·d) : inchangé, c'est la bibliothèque.
* Le `rapport.md` décrit l'ancien protocole (M=10 000, référence CRN) : il faudra le mettre à jour
  une fois les nouveaux résultats produits.
