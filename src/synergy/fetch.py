"""Cached, rate-limited access to stats.nba.com via nba_api.

Every response is saved as raw JSON under data/raw/<endpoint>/<key>.json
before any parsing, so a parser bug never means re-scraping.
IDs are always handled as strings (leading zeros matter: 0022201086).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from nba_api.stats import endpoints as ep

from .config import load_config

log = logging.getLogger(__name__)

# nba_api's own STATS_HEADERS mimic a current Chrome; custom headers get
# silently stalled by stats.nba.com, so we use the library defaults.

_last_call = 0.0


def _cache_path(endpoint: str, key: str) -> Path:
    return load_config()["paths"]["raw"] / endpoint / f"{key}.json"


def cached_call(endpoint_cls, key: str, refresh: bool = False, **kwargs) -> dict:
    """Return the JSON dict for an endpoint call, from cache if present."""
    global _last_call
    cfg = load_config()["fetch"]
    path = _cache_path(endpoint_cls.__name__, key)
    if path.exists() and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    for attempt in range(cfg["max_retries"]):
        wait = cfg["sleep_seconds"] - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_call = time.monotonic()
            data = endpoint_cls(
                **kwargs, timeout=cfg["timeout_seconds"]
            ).get_dict()
            break
        except Exception as exc:  # network errors, timeouts, bad JSON
            backoff = cfg["backoff_base_seconds"] * 2**attempt
            log.warning("%s %s failed (%s); retry in %.0fs",
                        endpoint_cls.__name__, key, exc, backoff)
            time.sleep(backoff)
    else:
        raise RuntimeError(f"{endpoint_cls.__name__} {key} failed after retries")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)
    return data


def result_set(data: dict, name: str | None = None):
    """Turn a stats.nba.com resultSets payload into a DataFrame (all str IDs)."""
    import pandas as pd

    sets = data.get("resultSets") or data.get("resultSet")
    if isinstance(sets, dict):
        sets = [sets]
    rs = sets[0] if name is None else next(s for s in sets if s["name"] == name)
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    for col in df.columns:
        if col.endswith("_ID") or col == "GAME_ID":
            df[col] = df[col].astype(str)
    return df


# ---- endpoint wrappers -------------------------------------------------------

def season_game_log(season: str, season_type: str = "Regular Season"):
    """One row per team-game, with box totals (FGA, OREB, TOV, FTA, PTS)."""
    data = cached_call(
        ep.LeagueGameLog, f"{season}_{season_type.replace(' ', '')}",
        season=season, season_type_all_star=season_type,
        player_or_team_abbreviation="T",
    )
    df = result_set(data)
    df["GAME_ID"] = df["GAME_ID"].str.zfill(10)
    return df


def play_by_play(game_id: str) -> dict:
    return cached_call(ep.PlayByPlayV3, game_id, game_id=game_id)


def game_rotation(game_id: str) -> dict:
    return cached_call(ep.GameRotation, game_id, game_id=game_id)


def box_score(game_id: str) -> dict:
    """Only needed for games missing from the season game log (Cup final)."""
    return cached_call(ep.BoxScoreTraditionalV3, game_id, game_id=game_id)


def player_stats(season: str, per_mode: str, measure: str = "Base"):
    data = cached_call(
        ep.LeagueDashPlayerStats, f"{season}_{measure}_{per_mode}",
        season=season, per_mode_detailed=per_mode,
        measure_type_detailed_defense=measure,
        season_type_all_star="Regular Season",
    )
    return result_set(data)


def player_bio(season: str):
    data = cached_call(ep.LeagueDashPlayerBioStats, season, season=season,
                       season_type_all_star="Regular Season")
    return result_set(data)


def player_index(season: str = "2025-26"):
    """All players ever (historical=1) with POSITION and FROM_YEAR.

    Without historical=1 the endpoint returns only ~100 rows.
    """
    data = cached_call(ep.PlayerIndex, f"historical_{season}", season=season,
                       historical_nullable="1")
    return result_set(data)
