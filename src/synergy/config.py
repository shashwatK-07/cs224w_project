"""Load config.yaml. Every processing rule lives there, not in code."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = ROOT / rel
    return cfg


def season_start_year(season: str) -> int:
    """'2021-22' -> 2021."""
    return int(season[:4])


def prev_season(season: str) -> str:
    y = season_start_year(season) - 1
    return f"{y}-{str(y + 1)[-2:]}"
