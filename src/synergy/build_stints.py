"""Fetch and parse every game of a season into a stint table.

    python -m synergy.build_stints --season 2023-24 --limit 10   # validate first
    python -m synergy.build_stints --season all                  # full run

Outputs (data/interim/):
    stints_<season>.parquet       one row per stint
    flags_<season>.csv            validation failures to inspect
    possessions_<season>.csv      counted vs box-estimate possessions per game
"""
from __future__ import annotations

import argparse
import logging
import traceback

import pandas as pd

from . import fetch
from .config import load_config
from .stints import parse_game
from .validate import check_game, possession_summary

log = logging.getLogger("build_stints")


def season_games(season: str) -> pd.DataFrame:
    """One row per game: game_id, home/away team ids and box rows."""
    cfg = load_config()
    tg = fetch.season_game_log(season)
    tg = pd.concat([tg] + [box_score_rows(g) for g in
                           cfg.get("extra_game_ids", {}).get(season, [])], ignore_index=True)

    # Home = the "vs." row. Some neutral-site games (2024-25 on) list both
    # teams with "@"; for those, the box score's homeTeam is the listed home.
    is_home = tg.MATCHUP.str.contains(" vs. ")
    for gid in set(tg.GAME_ID) - set(tg.loc[is_home, "GAME_ID"]):
        home_id = str(fetch.box_score(gid)["boxScoreTraditional"]["homeTeamId"])
        is_home |= (tg.GAME_ID == gid) & (tg.TEAM_ID == home_id)

    home = tg[is_home].set_index("GAME_ID")
    away = tg[~is_home].set_index("GAME_ID")
    games = home.add_suffix("_H").join(away.add_suffix("_A"), how="inner")
    if len(games) != tg.GAME_ID.nunique():
        raise ValueError(f"{season}: {tg.GAME_ID.nunique() - len(games)} games lost pairing teams")
    return games.reset_index().sort_values(["GAME_DATE_H", "GAME_ID"])


def box_score_rows(game_id: str) -> pd.DataFrame:
    """Game-log-style team rows for a game outside the regular-season log (Cup final)."""
    b = fetch.box_score(game_id)["boxScoreTraditional"]
    rows = []
    for side, other in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
        t, st = b[side], b[side]["statistics"]
        sep = " vs. " if side == "homeTeam" else " @ "
        rows.append({
            "GAME_ID": game_id, "TEAM_ID": str(t["teamId"]),
            "TEAM_ABBREVIATION": t["teamTricode"],
            "MATCHUP": t["teamTricode"] + sep + b[other]["teamTricode"],
            "GAME_DATE": "", "PTS": st["points"], "FGA": st["fieldGoalsAttempted"],
            "OREB": st["reboundsOffensive"], "TOV": st["turnovers"],
            "FTA": st["freeThrowsAttempted"],
        })
    return pd.DataFrame(rows)


def box_row(g, suffix):
    cols = ["GAME_ID", "TEAM_ID", "PTS", "FGA", "OREB", "TOV", "FTA"]
    vals = {c: g[f"{c}_{suffix}"] if c != "GAME_ID" else g["GAME_ID"] for c in cols}
    return pd.Series(vals)


def build_season(season: str, limit: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    games = season_games(season)
    if limit:
        games = games.head(limit)
    neutral = set(cfg.get("neutral_site_game_ids") or [])

    all_stints, all_flags, poss = [], [], []
    for k, g in enumerate(games.itertuples(index=False), 1):
        g = pd.Series(g._asdict())
        gid = g["GAME_ID"]
        hb, ab = box_row(g, "H"), box_row(g, "A")
        try:
            pbp = fetch.play_by_play(gid)
            rot = fetch.game_rotation(gid)
            st, issues, team_oreb = parse_game(gid, hb.TEAM_ID, ab.TEAM_ID, pbp, rot)
        except Exception as exc:
            log.error("%s failed: %s", gid, exc)
            all_flags.append({"game_id": gid, "check": "exception",
                              "detail": traceback.format_exc(limit=2)})
            continue
        st.insert(1, "season", season)
        st["home_s"] = 0 if gid in neutral else 1
        all_flags += check_game(st, issues, hb, ab, team_oreb)
        poss.append(possession_summary(st, hb, ab, team_oreb))
        all_stints.append(st)
        if k % 50 == 0:
            log.info("%s: %d/%d games", season, k, len(games))

    stints = pd.concat(all_stints, ignore_index=True)
    stints = add_garbage_flag(stints)
    out = cfg["paths"]["interim"]
    out.mkdir(parents=True, exist_ok=True)
    suffix = season if not limit else f"{season}_first{limit}"
    stints.to_parquet(out / f"stints_{suffix}.parquet", index=False)
    pd.DataFrame(all_flags, columns=["game_id", "check", "detail"]).to_csv(
        out / f"flags_{suffix}.csv", index=False)
    pd.DataFrame(poss).to_csv(out / f"possessions_{suffix}.csv", index=False)

    n_games = stints.game_id.nunique()
    flagged = len({f["game_id"] for f in all_flags})
    log.info("%s: %d games, %d stints, %d games flagged (%.1f%%)",
             season, n_games, len(stints), flagged, 100 * flagged / max(n_games, 1))
    return stints


def add_garbage_flag(st: pd.DataFrame) -> pd.DataFrame:
    gt = load_config()["garbage_time"]
    st["garbage_time"] = (
        (st.start_margin.abs() >= gt["min_abs_margin"])
        & (st.period == gt["period"])
        & (st.start_clock <= gt["max_seconds_left"])
    ).astype(int)
    return st


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g. 2023-24, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="first N games only")
    args = ap.parse_args()
    cfg = load_config()
    seasons = (cfg["seasons"]["train"] + cfg["seasons"]["test"]
               if args.season == "all" else [args.season])
    for s in seasons:
        build_season(s, args.limit)


if __name__ == "__main__":
    main()
