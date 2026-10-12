# -*- coding: utf-8 -*-
"""Configuration de l'experience (dataclasses). Un run = un ExperimentConfig.

  - DONNEES   : MarketConfig, PortfolioConfig
  - RISQUE    : RiskConfig (scenarios, horizons, niveaux VaR/ES), BoxConfig
  - NUMERIQUE : MCConfig (UN budget, UNE seed noCRN), CTConfig, TTConfig
  - PROXIES   : ProxyConfig (methode + hyperparametres par groupe)
  - EXECUTION : RunConfig (cache, checkpoints, parallelisme)

Regles imposees (voir CHANGEMENTS.md) :
  * full repricing = pricing UNITAIRE (produit par produit), regime noCRN,
    generateur RE-CREE avec la MEME seed a chaque appel (cout d'un pricer FO) ;
  * noeuds des proxies (Full Chebyshev et TT) = pricing CRN du groupe ;
  * n_paths = 100 000 trajectoires pour TOUS les pricings ;
  * UN SEUL full repricing : 10 000 chocs sur les 25 sous-jacents, 2 horizons ;
  * boite du proxy = enveloppe empirique des scenarios 1j+10j, marge 1 %.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------ marche
@dataclass
class MarketConfig:
    n_underlyings: int = 25
    spot_range: Tuple[float, float] = (100.0, 120.0)
    vol_range: Tuple[float, float] = (0.30, 0.60)
    rho: float = 0.50
    rate: float = 0.01
    deterministic: bool = True
    seed: int = 20240101

    def validate(self):
        if self.n_underlyings < 2:
            raise ValueError("n_underlyings >= 2 requis.")
        if self.spot_range[0] <= 0 or self.spot_range[1] < self.spot_range[0]:
            raise ValueError("spot_range invalide : %r" % (self.spot_range,))
        if self.vol_range[0] <= 0 or self.vol_range[1] < self.vol_range[0]:
            raise ValueError("vol_range invalide : %r" % (self.vol_range,))
        lower = -1.0 / (self.n_underlyings - 1)
        if not (lower < self.rho < 1.0):
            raise ValueError("rho=%.4f hors de ]%.4f, 1[." % (self.rho, lower))
        return self


# ------------------------------------------------------------- portefeuille
@dataclass
class ProductSpec:
    """kappa : moneyness (K = kappa * B_0) ; alpha : position (nominal)."""
    k: int
    kappa: float
    maturity: float
    callput_flag: int          # 0 = call, 1 = put
    alpha: float


DEFAULT_PRODUCT_GRID: List[ProductSpec] = [
    ProductSpec(k=1, kappa=0.90, maturity=1.0, callput_flag=0, alpha=+1.0),
    ProductSpec(k=2, kappa=0.95, maturity=2.0, callput_flag=0, alpha=+1.0),
    ProductSpec(k=3, kappa=1.00, maturity=3.0, callput_flag=0, alpha=+1.0),
    ProductSpec(k=4, kappa=1.05, maturity=4.0, callput_flag=1, alpha=-1.0),
    ProductSpec(k=5, kappa=1.10, maturity=5.0, callput_flag=1, alpha=-1.0),
]


@dataclass
class PortfolioConfig:
    dimensions: Tuple[int, ...] = (5, 10, 15, 20, 25)
    option_types: Tuple[int, ...] = (0, 1, 2, 3)      # geo, basket, BO, WO
    product_grid: List[ProductSpec] = field(
        default_factory=lambda: list(DEFAULT_PRODUCT_GRID))

    @property
    def n_products_per_group(self) -> int:
        return len(self.option_types) * len(self.product_grid)

    @property
    def n_products(self) -> int:
        return len(self.dimensions) * self.n_products_per_group

    def validate(self, n_underlyings: int):
        if not self.dimensions:
            raise ValueError("dimensions vide.")
        if max(self.dimensions) > n_underlyings:
            raise ValueError("dimension max > nombre de sous-jacents.")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions dupliquees.")
        for s in self.product_grid:
            if s.kappa <= 0 or s.maturity <= 0 or s.callput_flag not in (0, 1):
                raise ValueError("ProductSpec k=%d invalide." % s.k)
        return self


# ------------------------------------------------------------------ risque
@dataclass
class RiskConfig:
    """UN seul jeu de chocs : n_scenarios normales correlees N(0, C_25),
    LE MEME paquet sert aux deux horizons ; le groupe d lit le prefixe s[:d]."""
    n_scenarios: int = 10_000
    seed_scenarios: int = 3467
    horizons: Dict[str, float] = field(
        default_factory=lambda: {"1j": 1.0 / 252.0, "10j": 10.0 / 252.0})
    var_levels: Tuple[float, ...] = (0.99,)
    es_levels: Tuple[float, ...] = (0.975, 0.99)

    def validate(self):
        if self.n_scenarios < 100:
            raise ValueError("n_scenarios >= 100 requis.")
        for a in tuple(self.var_levels) + tuple(self.es_levels):
            if not (0.5 < a < 1.0):
                raise ValueError("niveau de risque hors ]0.5, 1[ : %r" % a)
        return self


@dataclass
class BoxConfig:
    """Enveloppe empirique : lo_i = 0.99 min_{k,h} s_hi^k, hi_i = 1.01 max_{k,h}."""
    factor_lo: float = 0.99
    factor_hi: float = 1.01

    def validate(self):
        if not (0 < self.factor_lo <= 1.0 <= self.factor_hi):
            raise ValueError("facteurs de marge invalides.")
        return self


# --------------------------------------------------------------- numerique
@dataclass
class MCConfig:
    """UN budget de trajectoires pour TOUS les pricings, UNE seed noCRN.

    seed_pricing : seed du generateur RE-CREE a chaque pricing unitaire noCRN
                   (full repricing, V0, etude de convergence).
    seed_crn     : seed du paquet CRN fige des pricers de groupe (noeuds des
                   proxies). None -> seed_pricing. ATTENTION : avec la meme
                   seed, le paquet CRN du groupe d est EXACTEMENT celui que tire
                   le pricer unitaire -> les valeurs aux noeuds coincident avec
                   la fonction du full repricing : l'ecart proxy/FR mesure
                   alors l'erreur d'interpolation SEULE.
    """
    n_paths: int = 100_000
    seed_pricing: int = 98765
    seed_crn: Optional[int] = None

    @property
    def crn_seed(self) -> int:
        return int(self.seed_pricing if self.seed_crn is None else self.seed_crn)

    def validate(self):
        if self.n_paths <= 0:
            raise ValueError("n_paths > 0 requis.")
        return self


@dataclass
class CTConfig:
    """Full Chebyshev Tensor (mocaxpy.Mocax) : grille pleine (degree+1)^d."""
    degree: int = 4

    def validate(self):
        if self.degree < 1:
            raise ValueError("degree >= 1 requis.")
        return self


@dataclass
class TTConfig:
    """Tensor-train (MoCaXExtend) d'UN groupe."""
    degree: int = 5                        # -> degree+1 points par dimension
    n_train: int = 1500                    # taille de sous-grille DEMANDEE
    max_rank: int = 10
    tolerance: float = 1e-4
    rel_tolerance: float = 1e-8
    max_iters: int = 20
    max_rounds: int = 20                   # NB : ecrase par la lib (5*d)
    train_frac: float = 0.8
    n_restarts: int = 1
    seed_fit: int = 0                      # np.random.seed avant chaque ALS
    seed_subgrid: int = 0                  # np.random.seed avant subgrid_by_number

    @property
    def points_per_dim(self) -> int:
        return self.degree + 1

    def full_grid_size(self, d: int) -> float:
        return float(self.points_per_dim) ** d

    def algo_params(self, print_progress: bool = False) -> dict:
        return dict(tolerance=self.tolerance, rel_tolerance=self.rel_tolerance,
                    max_iters=self.max_iters, max_rank=self.max_rank,
                    max_rounds=self.max_rounds, print_progress=print_progress)

    def validate(self):
        if self.degree < 1 or self.max_rank < 1 or self.n_restarts < 1:
            raise ValueError("degree, max_rank, n_restarts >= 1 requis.")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac dans ]0,1[ requis.")
        if self.n_train < 10:
            raise ValueError("n_train >= 10 requis.")
        return self


def _default_tt() -> Dict[int, TTConfig]:
    # hyperparametres effectivement utilises dans le notebook precedent
    return {10: TTConfig(n_train=1500, max_rank=10),
            15: TTConfig(n_train=2500, max_rank=10),
            20: TTConfig(n_train=3500, max_rank=8),
            25: TTConfig(n_train=5000, max_rank=8)}


@dataclass
class ProxyConfig:
    """methods[d] in {"full_cheb", "tt"} ; hyperparametres par methode."""
    methods: Dict[int, str] = field(default_factory=lambda: {
        5: "full_cheb", 10: "tt", 15: "tt", 20: "tt", 25: "tt"})
    ct: CTConfig = field(default_factory=CTConfig)
    tt: Dict[int, TTConfig] = field(default_factory=_default_tt)

    def params_for(self, d: int):
        m = self.methods[d]
        if m == "full_cheb":
            return self.ct
        if m == "tt":
            if d not in self.tt:
                raise KeyError("TTConfig manquant pour d=%d" % d)
            return self.tt[d]
        raise ValueError("methode inconnue %r" % m)

    def validate(self, dimensions):
        for d in dimensions:
            if d not in self.methods:
                raise KeyError("methode non definie pour d=%d" % d)
            self.params_for(d).validate()
        return self


# --------------------------------------------------------------- execution
@dataclass
class RunConfig:
    cache_dir: str = "cache"
    checkpoint_every: int = 250     # sauvegarde partielle tous les k pricings
    n_workers: int = 1              # >1 : processus paralleles (full repricing)
    progress_every: int = 500


# ---------------------------------------------------------- orchestration
@dataclass
class ExperimentConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    proxies: ProxyConfig = field(default_factory=ProxyConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self):
        self.market.validate()
        self.portfolio.validate(self.market.n_underlyings)
        self.risk.validate()
        self.box.validate()
        self.mc.validate()
        self.proxies.validate(self.portfolio.dimensions)
        return self

    @classmethod
    def smoke(cls, cache_dir="cache_smoke"):
        """Petit run pour verifier la chaine (resultats NON exploitables)."""
        tt = {d: TTConfig(n_train=n, max_rank=4, degree=3)
              for d, n in {10: 150, 15: 200, 20: 250, 25: 300}.items()}
        return cls(risk=RiskConfig(n_scenarios=200),
                   mc=MCConfig(n_paths=2_000),
                   proxies=ProxyConfig(ct=CTConfig(degree=3), tt=tt),
                   run=RunConfig(cache_dir=cache_dir, checkpoint_every=50,
                                 progress_every=100)).validate()

    def summary(self) -> str:
        lines = [
            "Marche       : %d sous-jacents, rho=%.2f, r=%.4f"
            % (self.market.n_underlyings, self.market.rho, self.market.rate),
            "Portefeuille : %d produits (%d groupes x %d types x %d moneyness)"
            % (self.portfolio.n_products, len(self.portfolio.dimensions),
               len(self.portfolio.option_types), len(self.portfolio.product_grid)),
            "Scenarios    : %s chocs (seed %d), horizons %s"
            % ("{:,}".format(self.risk.n_scenarios), self.risk.seed_scenarios,
               ", ".join("%s=%.5f" % kv for kv in self.risk.horizons.items())),
            "Mesures      : VaR %s | ES %s"
            % (", ".join("%.1f%%" % (100 * a) for a in self.risk.var_levels),
               ", ".join("%.1f%%" % (100 * a) for a in self.risk.es_levels)),
            "Monte-Carlo  : %s trajectoires / pricing ; seed noCRN = %d ; "
            "seed CRN noeuds = %d"
            % ("{:,}".format(self.mc.n_paths), self.mc.seed_pricing,
               self.mc.crn_seed),
            "Full repric. : UNITAIRE (produit par produit), noCRN, generateur "
            "re-cree a chaque appel",
            "Boite proxy  : enveloppe scenarios 1j+10j, [%.2f min, %.2f max]"
            % (self.box.factor_lo, self.box.factor_hi),
        ]
        for d in self.portfolio.dimensions:
            m = self.proxies.methods[d]
            p = self.proxies.params_for(d)
            if m == "full_cheb":
                lines.append("Proxy d=%-3d  : Full Chebyshev, degre %d (%d^%d = %d noeuds)"
                             % (d, p.degree, p.degree + 1, d, (p.degree + 1) ** d))
            else:
                lines.append("Proxy d=%-3d  : TT degre %d, n_train=%d, max_rank=%d, "
                             "tol=%.0e, iters=%d, restarts=%d"
                             % (d, p.degree, p.n_train, p.max_rank, p.tolerance,
                                p.max_iters, p.n_restarts))
        lines.append("Cache        : %s" % self.run.cache_dir)
        return "\n".join(lines)


# --------------------------------------------------------------- empreintes
def fingerprint(*objs) -> str:
    """Hash court et stable d'objets (dataclasses / dict / scalaires)."""
    def conv(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        return o
    payload = json.dumps([conv(o) for o in objs], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def replace_mc(cfg, **kw):
    return replace(cfg, mc=replace(cfg.mc, **kw))


def replace_risk(cfg, **kw):
    return replace(cfg, risk=replace(cfg.risk, **kw))
