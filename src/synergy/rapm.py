"""RAPM baseline: possession-weighted ridge regression, additive in players.

    y_s = h * home_s + sum_{i in A_s} beta_i - sum_{j in B_s} beta_j + eps_s

One beta per player, pooled across seasons. The intercept h is not penalized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lineup_index(st: pd.DataFrame, player_ix: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(n, 10) player column indices and signs (+1 home, -1 away).

    Players missing from player_ix (never seen in training) get index -1;
    they contribute beta = 0, the mean of the ridge prior.
    """
    def ix(col):
        return np.array([[player_ix.get(p, -1) for p in s.split("|")] for s in st[col]])
    idx = np.hstack([ix("home_players"), ix("away_players")])
    sign = np.hstack([np.ones((len(st), 5)), -np.ones((len(st), 5))])
    return idx, sign


def fit_rapm(st: pd.DataFrame, lam: float) -> dict:
    """Weighted ridge via the normal equations, built from the sparse design."""
    players = sorted(set("|".join(st.home_players).split("|"))
                     | set("|".join(st.away_players).split("|")))
    player_ix = {p: k for k, p in enumerate(players)}
    P = len(players)
    idx, sign = lineup_index(st, player_ix)
    w, y, home = st.n_s.to_numpy(), st.y_s.to_numpy(), st.home_s.to_numpy(float)

    # Gram matrix X'WX over [players | intercept]: each row has 10 nonzeros.
    G = np.zeros((P + 1, P + 1))
    a, b = np.repeat(idx, 10, axis=1), np.tile(idx, (1, 10))
    vals = np.repeat(sign, 10, axis=1) * np.tile(sign, (1, 10)) * w[:, None]
    np.add.at(G, (a.ravel(), b.ravel()), vals.ravel())
    xh = np.zeros(P)                                   # X'W home
    np.add.at(xh, idx.ravel(), (sign * (w * home)[:, None]).ravel())
    G[:P, P] = G[P, :P] = xh
    G[P, P] = (w * home * home).sum()

    rhs = np.zeros(P + 1)
    np.add.at(rhs, idx.ravel(), (sign * (w * y)[:, None]).ravel())
    rhs[P] = (w * home * y).sum()

    G[np.arange(P), np.arange(P)] += lam               # ridge on beta only
    sol = np.linalg.solve(G, rhs)
    return {"beta": sol[:P], "h": sol[P], "player_ix": player_ix}


def predict(model: dict, st: pd.DataFrame) -> np.ndarray:
    idx, sign = lineup_index(st, model["player_ix"])
    beta = np.append(model["beta"], 0.0)               # index -1 -> unseen player
    return model["h"] * st.home_s.to_numpy(float) + (beta[idx] * sign).sum(axis=1)


def weighted_rmse(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    """Possession-weighted RMSE, the project's headline metric."""
    return float(np.sqrt((w * (y - yhat) ** 2).sum() / w.sum()))
