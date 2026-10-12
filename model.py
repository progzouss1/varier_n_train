# -*- coding: utf-8 -*-
"""Modele de portefeuille : les classes de base, sans hierarchie.

Les 4 types de panier ne different que par l'agregation R^d -> R ; le if/elif
de `aggregate` est exactement le bon niveau de complexite pour un PoC.
"""

from dataclasses import dataclass
import numpy as np

OPTION_TYPES = {0: "geo", 1: "basket", 2: "BO", 3: "WO"}


def aggregate(S, option_type):
    """Agrege (n_paths, d) -> (n_paths,) selon le type de panier.

    Sert aussi au calcul de B_0 (niveau de l'agregat aux spots initiaux),
    reference des strikes relatifs K = kappa * B_0.
    """
    S = np.asarray(S, dtype=float)
    t = int(option_type)
    if t == 0:                            # moyenne geometrique
        return np.exp(np.mean(np.log(S), axis=1))
    if t == 1:                            # moyenne arithmetique
        return np.mean(S, axis=1)
    if t == 2:                            # best-of
        return np.max(S, axis=1)
    if t == 3:                            # worst-of
        return np.min(S, axis=1)
    raise ValueError("option_type inconnu : %r" % option_type)


@dataclass
class Underlying:
    id: int
    spot: float
    vol: float


@dataclass
class Derivative:
    derivative_id: int
    option_type: int
    callput_flag: int          # 0 = call, 1 = put
    strike: float
    maturity: float
    underlying_ids: tuple
    nominal: float             # position alpha (facteur d'echelle du prix)

    @property
    def dimension(self) -> int:
        return len(self.underlying_ids)

    @property
    def type_name(self) -> str:
        return OPTION_TYPES[self.option_type]

    def payoff(self, terminal_spots):
        """Payoff europeen a maturite, vectorise : (n_paths, d) -> (n_paths,)."""
        S = np.asarray(terminal_spots, dtype=float)
        if S.ndim != 2 or S.shape[1] != self.dimension:
            raise ValueError("terminal_spots attendu (n_paths, %d), recu %s."
                             % (self.dimension, (S.shape,)))
        agg = aggregate(S, self.option_type)
        if self.callput_flag == 0:
            return np.maximum(agg - self.strike, 0.0)
        return np.maximum(self.strike - agg, 0.0)


@dataclass
class MarketData:
    """Etat de marche. Seuls les spots varient d'un scenario a l'autre."""
    spots: np.ndarray
    vols: np.ndarray
    corr_matrix: np.ndarray
    rate: float

    def sub(self, ids):
        idx = list(ids)
        return MarketData(self.spots[idx].copy(), self.vols[idx].copy(),
                          self.corr_matrix[np.ix_(idx, idx)].copy(), self.rate)


class Portfolio:
    """Conteneur iterable des derives, accessible par derivative_id."""

    def __init__(self, derivatives):
        self._derivatives = dict(derivatives)     # {id: Derivative}

    def __iter__(self):
        return iter(self._derivatives.values())

    def __getitem__(self, derivative_id):
        return self._derivatives[derivative_id]

    def __len__(self):
        return len(self._derivatives)

    @property
    def ids(self):
        return list(self._derivatives.keys())

    def group(self, d):
        """Derives du groupe de dimension d, tries par identifiant."""
        return [dv for dv in sorted(self, key=lambda x: x.derivative_id)
                if dv.dimension == d]
