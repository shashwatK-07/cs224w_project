"""Per-game checks from DECISIONS.md.

1. Exactly 5 players per side in every stint.
2. Stint margins sum to the final score.
3. Each team's counted possessions within tolerance of
   FGA - OREB + TOV + 0.44*FTA - team_OREB.
   Box-score OREB counts player rebounds only. Team offensive rebounds
   (e.g. a miss deflected out of bounds off the defense) also continue a
   possession, so the raw formula overcounts by ~4 per team-game. We subtract
   team OREBs counted from play-by-play.
Failures are returned as flag rows; a human inspects them and records the
fix in data/bad_data_log.csv (game_id, issue, fix).
"""
from __future__ import annotations

import pandas as pd

from .config import load_config


def box_possessions(row, team_oreb: int = 0) -> float:
    return row.FGA - row.OREB + row.TOV + 0.44 * row.FTA - team_oreb


def check_game(stints: pd.DataFrame, issues: list[str], home_box, away_box,
               team_oreb: dict) -> list[dict]:
    tol = load_config()["possession_check"]["tolerance"]
    gid = home_box.GAME_ID
    flags = [{"game_id": gid, "check": "parser", "detail": msg} for msg in issues]

    for side in ("home", "away"):
        sizes = stints[f"{side}_players"].str.split("|").str.len()
        bad = stints.loc[sizes != 5, "stint_idx"].tolist()
        if bad:
            flags.append({"game_id": gid, "check": "five_players",
                          "detail": f"{side} stints {bad} sizes {sizes[sizes != 5].tolist()}"})

    for side, box in (("home", home_box), ("away", away_box)):
        pts = int(stints[f"pts_{side}"].sum())
        if pts != int(box.PTS):
            flags.append({"game_id": gid, "check": "final_score",
                          "detail": f"{side} stints sum {pts} vs box {int(box.PTS)}"})
        counted = int(stints[f"poss_{side}"].sum())
        est = box_possessions(box, team_oreb[side])
        if abs(counted - est) > tol:
            flags.append({"game_id": gid, "check": "possessions",
                          "detail": f"{side} counted {counted} vs box est {est:.1f}"})
    return flags


def possession_summary(stints: pd.DataFrame, home_box, away_box, team_oreb: dict) -> dict:
    return {
        "game_id": home_box.GAME_ID,
        "home_counted": int(stints.poss_home.sum()),
        "home_est": box_possessions(home_box, team_oreb["home"]),
        "home_raw_box_est": box_possessions(home_box),
        "away_counted": int(stints.poss_away.sum()),
        "away_est": box_possessions(away_box, team_oreb["away"]),
        "away_raw_box_est": box_possessions(away_box),
        "n_stints": len(stints),
    }
