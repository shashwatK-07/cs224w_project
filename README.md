# NBA lineup synergy (CS224W)

Data pipeline for predicting stint margins from two opposing five-player lineups.
All processing rules live in [config.yaml](config.yaml).

## Setup

```bash
py -3.14 -m venv .venv            # any CPython >= 3.10 (not the MSYS2 build)
.venv/Scripts/python -m pip install -e ".[dev]"
```

## Pipeline

```bash
# 1. Scrape + parse stints (cached; safe to interrupt and rerun)
python -m synergy.build_stints --season 2023-24 --limit 10   # validate on 10 games first
python -m synergy.build_stints --season all                  # ~6,150 games, ~4 h

# 2. Drop n_s = 0, assign train/val/test splits, flag unseen lineups
python -m synergy.splits

# 3. Player-season node features (season t-1 stats)
python -m synergy.features

# Tests
python -m pytest
```

| Output | Contents |
| --- | --- |
| `data/raw/<Endpoint>/<key>.json` | Raw API cache (not committed) |
| `data/interim/stints_<season>.parquet` | One row per stint |
| `data/interim/flags_<season>.csv` | Games failing a validation check, to inspect by hand |
| `data/interim/possessions_<season>.csv` | Counted vs. box-estimate possessions per game |
| `data/processed/stints.parquet` | All seasons with `split`, `*_unseen`, `garbage_time` |
| `data/processed/nodes.parquet` | One row per player-season; model features prefixed `f_` |
| `data/bad_data_log.csv` | Hand fixes: game_id, issue, fix (committed) |

## Modules

- [fetch.py](src/synergy/fetch.py): rate-limited, retrying, disk-cached nba_api calls.
- [stints.py](src/synergy/stints.py): PlayByPlayV3 + GameRotation into stints and possessions.
- [validate.py](src/synergy/validate.py): 5 players per side, score sums, possession check.
- [splits.py](src/synergy/splits.py): game-level splits (seed 224) and unseen-lineup flags.
- [features.py](src/synergy/features.py): prior-season features, no_prior_stats imputation, lineup labels.
