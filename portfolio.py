# -*- coding: utf-8 -*-
"""Construction du marche synthetique et du portefeuille : fonctions libres.

    from config import ExperimentConfig
    from portfolio import build_market, build_portfolio, check_portfolio

    cfg = ExperimentConfig().validate()
    market = build_market(cfg.market)
    pf = build_portfolio(cfg.portfolio, market)
    check_portfolio(cfg, market, pf)

Groupes EMBOITES G_d = {0..d-1} : imposes par l'arithmetique (5+10+15+20+25 > 25).
La sous-matrice equicorrelee d'un groupe reste equicorrelee de meme rho, et le
Cholesky d'un bloc principal est le bloc principal du Cholesky : un scenario
complet s se lit par prefixe s[:d], et un paquet de normales correlees de
dimension 25 se restreint par troncature de colonnes (base de la variante
CRN-global de pricing.GroupPricer).

Strike relatif K = kappa * B_0(type, d) : sans cette convention, les worst-of
seraient tous profondement OTM et les best-of tous ITM.
"""

import numpy as np

from model import Derivative, MarketData, Portfolio, aggregate


def equicorrelation(n, rho):
    """C = rho*11' + (1-rho)*I. PD ssi -1/(n-1) < rho < 1."""
    C = np.full((n, n), float(rho))
    np.fill_diagonal(C, 1.0)
    return C


def build_market(market_cfg) -> MarketData:
    """Spots et vols en rampes (ou tirages uniformes), correlation equicorrelee."""
    n = market_cfg.n_underlyings
    smin, smax = market_cfg.spot_range
    vmin, vmax = market_cfg.vol_range
    if market_cfg.deterministic:
        t = np.arange(n, dtype=float) / max(n - 1, 1)
        spots = smin + (smax - smin) * t
        vols = vmin + (vmax - vmin) * t
    else:
        rng = np.random.default_rng(market_cfg.seed)
        spots = rng.uniform(smin, smax, size=n)
        vols = rng.uniform(vmin, vmax, size=n)
    return MarketData(spots=spots, vols=vols,
                      corr_matrix=equicorrelation(n, market_cfg.rho),
                      rate=float(market_cfg.rate))


def basket_level(market, d, option_type) -> float:
    """B_0 : niveau de l'agregat aux spots initiaux des d premiers actifs."""
    s = np.asarray(market.spots[:d], dtype=float).reshape(1, -1)
    return float(aggregate(s, option_type)[0])


def build_portfolio(portfolio_cfg, market) -> Portfolio:
    """dimensions x types x grille de moneyness -> Portfolio."""
    derivatives = {}
    next_id = 1
    for d in portfolio_cfg.dimensions:
        ids = tuple(range(d))
        for option_type in portfolio_cfg.option_types:
            b0 = basket_level(market, d, option_type)
            for spec in portfolio_cfg.product_grid:
                derivatives[next_id] = Derivative(
                    derivative_id=next_id,
                    option_type=int(option_type),
                    callput_flag=int(spec.callput_flag),
                    strike=float(spec.kappa * b0),
                    maturity=float(spec.maturity),
                    underlying_ids=ids,
                    nominal=float(spec.alpha))
                next_id += 1
    if len(derivatives) != portfolio_cfg.n_products:
        raise RuntimeError("%d produits construits, %d attendus."
                           % (len(derivatives), portfolio_cfg.n_products))
    return Portfolio(derivatives)


def check_portfolio(cfg, market, pf) -> bool:
    """Controles de coherence (spots, vols, correlation PD, groupes emboites)."""
    if np.any(market.spots <= 0) or np.any(market.vols <= 0):
        raise ValueError("Spots ou vols non strictement positifs.")
    C = market.corr_matrix
    if not np.allclose(C, C.T) or not np.allclose(np.diag(C), 1.0):
        raise ValueError("Matrice de correlation non symetrique ou diag != 1.")
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise ValueError("Correlation non definie positive (vp min = %.6g)."
                         % np.linalg.eigvalsh(C).min())
    for dv in pf:
        if dv.underlying_ids != tuple(range(dv.dimension)):
            raise ValueError("Produit %d : groupe non emboite." % dv.derivative_id)
        if dv.strike <= 0 or dv.maturity <= 0:
            raise ValueError("Produit %d : strike ou maturite <= 0." % dv.derivative_id)
    for d in cfg.portfolio.dimensions:
        got = len(pf.group(d))
        if got != cfg.portfolio.n_products_per_group:
            raise ValueError("Groupe d=%d : %d produits, %d attendus."
                             % (d, got, cfg.portfolio.n_products_per_group))
    return True