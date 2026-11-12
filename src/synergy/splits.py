"""Assign every stint to a split and flag unseen lineups.

    python -m synergy.splits

Reads data/interim/stints_<season>.parquet for every configured season and
writes data/processed/stints.parquet with columns:
    split           train | val | test_games | test_season
    home_unseen     home lineup's five players never shared a training stint
    away_unseen     same for away
    test_unseen     test row with at least one unseen lineup (test set 3)
    test_unseen_both  both lineups unseen

Splits are by game, seed 224: within each training season, 10% of games go
to validation and another 10% to held-out test; 2025-26 is entirely test.
Lineups are compared as player sets, ignoring season.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import load_config

log = logging.getLogger("splits")


def load_stints() -> pd.DataFrame:
    cfg = load_config()
    seasons = cfg["seasons"]["train"] + cfg["seasons"]["test"]
    frames = []
    for s in seasons:
        path = cfg["paths"]["interim"] / f"stints_{s}.parquet"
        if not path.exists():
            log.warning("missing %s; skipping season", path.name)
            continue
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def assign_splits(st: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    sp = cfg["splits"]
    rng = np.random.default_rng(sp["seed"])
    game_split = {}
    for season in cfg["seasons"]["train"]:
        games = np.array(sorted(st.loc[st.season == season, "game_id"].unique()))
        games = games[rng.permutation(len(games))]
        n_val = round(sp["val_frac"] * len(games))
        n_test = round(sp["test_frac"] * len(games))
        for g in games[:n_val]:
            game_split[g] = "val"
        for g in games[n_val:n_val + n_test]:
            game_split[g] = "test_games"
        for g in games[n_val + n_test:]:
            game_split[g] = "train"
    for season in cfg["seasons"]["test"]:
        for g in st.loc[st.season == season, "game_id"].unique():
            game_split[g] = "test_season"
    st["split"] = st.game_id.map(game_split)
    return st


def flag_unseen(st: pd.DataFrame) -> pd.DataFrame:
    train = st[st.split == "train"]
    seen = set(train.home_players) | set(train.away_players)  # sorted '|'-joined ids
    st["home_unseen"] = ~st.home_players.isin(seen)
    st["away_unseen"] = ~st.away_players.isin(seen)
    is_test = st.split.isin(["test_games", "test_season"])
    st["test_unseen"] = is_test & (st.home_unseen | st.away_unseen)
    st["test_unseen_both"] = is_test & st.home_unseen & st.away_unseen
    return st


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    st = load_stints()
    n0 = len(st)
    st = st[st.n_s >= cfg["stints"]["min_possessions"]].reset_index(drop=True)
    tiny = ((st.n_s > 0) & (st.n_s <= 2)).sum()
    log.info("dropped %d stints with n_s = 0; kept %d; %d (%.1f%%) have 1-2 possessions",
             n0 - len(st), len(st), tiny, 100 * tiny / len(st))

    st = flag_unseen(assign_splits(st))
    out = cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    st.to_parquet(out / "stints.parquet", index=False)

    summary = st.groupby("split").agg(games=("game_id", "nunique"), stints=("n_s", "size"),
                                      possessions=("n_s", "sum"))
    summary.loc["test_unseen"] = [st[st.test_unseen].game_id.nunique(),
                                  int(st.test_unseen.sum()), st[st.test_unseen].n_s.sum()]
    summary.loc["test_unseen_both"] = [st[st.test_unseen_both].game_id.nunique(),
                                       int(st.test_unseen_both.sum()),
                                       st[st.test_unseen_both].n_s.sum()]
    print(summary.to_string())


if __name__ == "__main__":
    main()
