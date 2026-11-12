"""Player-season node features (season t-1 stats for node (p, t)).

    python -m synergy.features

Reads data/processed/stints.parquet, writes data/processed/nodes.parquet with
one row per player-season appearing in any stint:
    node_id, player_id, season, no_prior_stats, raw prior stats,
    z-scored model features (prefix f_), lineup-type labels.

Rules (DECISIONS.md, "Features" and "Low-minute and missing prior seasons"):
  * no_prior_stats = 1 for rookies, players who missed t-1, or under 250
    minutes in t-1. Their stat features are imputed with the mean that such
    players actually produced in their season t, averaged over the training
    seasons only.
  * Bio features (height, age, position) are always the player's own values.
  * z-scores use means/stds from training-season nodes only.
  * Lineup labels come from raw prior stats; no_prior_stats players get none.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import fetch
from .config import load_config, prev_season

log = logging.getLogger("features")


def season_stats(season: str) -> pd.DataFrame:
    """One row per player for a season: per-100, efficiency, role, playing time."""
    base = fetch.player_stats(season, "Per100Possessions", "Base").set_index("PLAYER_ID")
    adv = fetch.player_stats(season, "PerGame", "Advanced").set_index("PLAYER_ID")
    tot = fetch.player_stats(season, "Totals", "Base").set_index("PLAYER_ID")
    df = base[["PTS", "FGA", "FG3A", "FTA", "AST", "TOV", "OREB", "DREB",
               "STL", "BLK", "PF", "FG3_PCT", "FT_PCT"]].copy()
    df[["TS_PCT", "EFG_PCT", "USG_PCT", "AST_PCT"]] = adv[["TS_PCT", "EFG_PCT",
                                                          "USG_PCT", "AST_PCT"]]
    fga = tot.FGA.replace(0, np.nan)
    df["FG3A_RATE"] = (tot.FG3A / fga).fillna(0)
    df["FTA_RATE"] = (tot.FTA / fga).fillna(0)
    df["GP"] = tot.GP
    df["MIN_TOTAL"] = tot.MIN
    df["MIN_PG"] = tot.MIN / tot.GP
    df.index = df.index.astype(str)
    return df


def bio_table(season: str, index: pd.DataFrame) -> pd.DataFrame:
    bio = fetch.player_bio(season).set_index("PLAYER_ID")
    df = pd.DataFrame({"HEIGHT_IN": bio.PLAYER_HEIGHT_INCHES, "AGE": bio.AGE})
    df.index = df.index.astype(str)
    pos = index.set_index("PERSON_ID").POSITION.reindex(df.index).fillna("")
    for code in ("G", "F", "C"):
        df[f"POS_{code}"] = pos.str.contains(code).astype(int)  # 'G-F' -> multi-hot
    return df


def build_nodes() -> pd.DataFrame:
    cfg = load_config()
    fc = cfg["features"]
    stints = pd.read_parquet(cfg["paths"]["processed"] / "stints.parquet")
    players = pd.concat([
        stints[["season", "home_players"]].rename(columns={"home_players": "p"}),
        stints[["season", "away_players"]].rename(columns={"away_players": "p"}),
    ])
    players["p"] = players.p.str.split("|")
    nodes = (players.explode("p").drop_duplicates()
             .rename(columns={"p": "player_id"}).reset_index(drop=True))

    stat_cols = fc["per100"] + fc["efficiency"] + fc["role"] + fc["playing_time"]
    index = fetch.player_index()
    rows = []
    for season, grp in nodes.groupby("season"):
        prior = season_stats(prev_season(season))
        same = season_stats(season)
        bio = bio_table(season, index)
        g = grp.set_index("player_id")
        p = prior.reindex(g.index)
        g["no_prior_stats"] = (p.MIN_TOTAL.isna()
                               | (p.MIN_TOTAL < fc["low_minutes_cutoff"])).astype(int)
        g[[f"prior_{c}" for c in stat_cols]] = p[stat_cols].to_numpy()
        g[[f"same_{c}" for c in stat_cols]] = same.reindex(g.index)[stat_cols].to_numpy()
        g[fc["bio"]] = bio.reindex(g.index)[fc["bio"]].to_numpy()
        rows.append(g.reset_index())
    nodes = pd.concat(rows, ignore_index=True)

    # Impute no-prior players with what such players produce in their season t,
    # averaged over training seasons only.
    train = nodes.season.isin(cfg["seasons"]["train"])
    impute = nodes.loc[train & (nodes.no_prior_stats == 1),
                       [f"same_{c}" for c in stat_cols]].mean()
    feats = pd.DataFrame(index=nodes.index)
    for c in stat_cols:
        x = nodes[f"prior_{c}"].where(nodes.no_prior_stats == 0, impute[f"same_{c}"])
        feats[c] = x.fillna(impute[f"same_{c}"])  # e.g. FG3_PCT NaN with 0 attempts
    for c in fc["bio"]:
        feats[c] = nodes[c].fillna(nodes.loc[train, c].mean())

    # z-score with training-season statistics; leave the 0/1 position flags raw.
    zcols = [c for c in feats.columns if not c.startswith("POS_")]
    mu, sd = feats.loc[train, zcols].mean(), feats.loc[train, zcols].std().replace(0, 1)
    feats[zcols] = (feats[zcols] - mu) / sd
    feats["no_prior_stats"] = nodes.no_prior_stats
    nodes = pd.concat([nodes, feats.add_prefix("f_")], axis=1)

    nodes = add_lineup_labels(nodes)
    nodes = nodes.drop(columns=[c for c in nodes.columns if c.startswith("same_")])
    nodes.insert(0, "node_id", np.arange(len(nodes)))
    return nodes


def add_lineup_labels(nodes: pd.DataFrame) -> pd.DataFrame:
    lc = load_config()["lineup_labels"]
    has = nodes.no_prior_stats == 0
    nodes["is_shooter"] = has & (nodes.prior_FG3A >= lc["shooter_min_3pa_per100"]) \
        & (nodes.prior_FG3_PCT >= lc["shooter_min_3p_pct"])
    nodes["is_nonshooter"] = has & (nodes.prior_FG3A < lc["nonshooter_max_3pa_per100"])
    nodes["is_ball_handler"] = has & (nodes.prior_AST_PCT >= lc["ball_handler_min_ast_pct"])
    return nodes


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    nodes = build_nodes()
    out = load_config()["paths"]["processed"] / "nodes.parquet"
    nodes.to_parquet(out, index=False)
    print(nodes.groupby("season").agg(nodes=("node_id", "size"),
                                      no_prior=("no_prior_stats", "sum"),
                                      shooters=("is_shooter", "sum"),
                                      nonshooters=("is_nonshooter", "sum"),
                                      handlers=("is_ball_handler", "sum")).to_string())
    missing = nodes[[c for c in nodes.columns if c.startswith("f_")]].isna().sum()
    print("NaN in model features:", int(missing.sum()))


if __name__ == "__main__":
    main()
