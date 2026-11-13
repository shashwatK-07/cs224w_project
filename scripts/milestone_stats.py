"""Preliminary numbers for the milestone report from games cached so far.

Parses every 2021-22 game whose PBP and rotation JSON are already on disk
(no network), runs the validation checks, and fits RAPM on a game-level
80/10/10 split (seed 224). Prints a JSON summary.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from synergy import rapm  # noqa: E402
from synergy.build_stints import add_garbage_flag, box_row, season_games  # noqa: E402
from synergy.config import load_config  # noqa: E402
from synergy.stints import parse_game  # noqa: E402
from synergy.validate import check_game, possession_summary  # noqa: E402

SEASON = "2021-22"
cfg = load_config()
raw = cfg["paths"]["raw"]
games = season_games(SEASON)

stints, flags, poss = [], [], []
for g in games.itertuples(index=False):
    g = pd.Series(g._asdict())
    gid = g["GAME_ID"]
    fp, fr = raw / "PlayByPlayV3" / f"{gid}.json", raw / "GameRotation" / f"{gid}.json"
    if not (fp.exists() and fr.exists()):
        continue
    hb, ab = box_row(g, "H"), box_row(g, "A")
    st, issues, toreb = parse_game(gid, hb.TEAM_ID, ab.TEAM_ID,
                                   json.load(open(fp)), json.load(open(fr)))
    st["season"], st["home_s"] = SEASON, 1
    flags += check_game(st, issues, hb, ab, toreb)
    poss.append(possession_summary(st, hb, ab, toreb))
    stints.append(st)

st = add_garbage_flag(pd.concat(stints, ignore_index=True))
poss, flags = pd.DataFrame(poss), pd.DataFrame(flags, columns=["game_id", "check", "detail"])
n_games = st.game_id.nunique()
out = {
    "games": n_games,
    "stints_raw": len(st),
    "stints_per_game": round(len(st) / n_games, 1),
    "n_s_zero": int((st.n_s == 0).sum()),
    "flagged_games": int(flags.game_id.nunique()),
    "flags_by_check": flags.groupby("check").game_id.nunique().to_dict(),
    "poss_diff_adj_mean": round(float(pd.concat([poss.home_counted - poss.home_est,
                                                 poss.away_counted - poss.away_est]).mean()), 2),
    "poss_diff_adj_absmean": round(float(pd.concat([poss.home_counted - poss.home_est,
                                                    poss.away_counted - poss.away_est]).abs().mean()), 2),
    "poss_diff_raw_mean": round(float(pd.concat([poss.home_counted - poss.home_raw_box_est,
                                                 poss.away_counted - poss.away_raw_box_est]).mean()), 2),
}
d = pd.concat([poss.home_counted - poss.home_est, poss.away_counted - poss.away_est])
out |= {"poss_diff_adj_sd": round(float(d.std()), 2),
        "team_games_over_tol": round(float((d.abs() > 3).mean()), 3),
        "team_games": len(d)}
# Games whose GameRotation is incomplete cannot be used until lineups are rebuilt from PBP.
bad = set(flags.loc[flags.check.isin(["parser", "five_players"]), "game_id"])
out["bad_rotation_games"] = len(bad)
st = st[~st.game_id.isin(bad)]
st = st[st.n_s >= cfg["stints"]["min_possessions"]].reset_index(drop=True)
out |= {
    "stints_kept": len(st),
    "tiny_1_2": int((st.n_s <= 2).sum()),
    "tiny_share": round(float((st.n_s <= 2).mean()), 3),
    "median_n_s": float(st.n_s.median()),
    "mean_n_s": round(float(st.n_s.mean()), 2),
    "garbage_stints": int(st.garbage_time.sum()),
    "players": len(set("|".join(st.home_players).split("|")) | set("|".join(st.away_players).split("|"))),
    "lineups": len(set(st.home_players) | set(st.away_players)),
    "home_adv": round(float((st.y_s * st.n_s).sum() / st.n_s.sum()), 2),
}

# game-level 80/10/10 split, seed 224
rng = np.random.default_rng(cfg["splits"]["seed"])
gids = np.array(sorted(st.game_id.unique()))
gids = gids[rng.permutation(len(gids))]
k = round(0.1 * len(gids))
split = {g: "val" for g in gids[:k]} | {g: "test" for g in gids[k:2 * k]}
st["split"] = st.game_id.map(split).fillna("train")
tr, va, te = (st[st.split == s] for s in ("train", "val", "test"))


def rmse(model, d):
    return rapm.weighted_rmse(d.y_s.to_numpy(), rapm.predict(model, d), d.n_s.to_numpy())


grid = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
val = {lam: rmse(rapm.fit_rapm(tr, lam), va) for lam in grid}
best = min(val, key=val.get)
m = rapm.fit_rapm(tr, best)
m0 = rapm.fit_rapm(tr, 1e12)   # beta ~ 0: intercept only
seen = set(tr.home_players) | set(tr.away_players)
te_unseen = te[~te.home_players.isin(seen) | ~te.away_players.isin(seen)]
out |= {
    "split_stints": st.split.value_counts().to_dict(),
    "val_rmse_by_lambda": {str(k_): round(v, 3) for k_, v in val.items()},
    "best_lambda": best,
    "h": round(float(m["h"]), 2),
    "beta_sd": round(float(m["beta"].std()), 2),
    "test_rmse_intercept": round(rmse(m0, te), 3),
    "test_rmse_rapm": round(rmse(m, te), 3),
    "val_rmse_intercept": round(rmse(m0, va), 3),
    "val_rmse_rapm": round(val[best], 3),
    "test_unseen_share": round(len(te_unseen) / len(te), 3),
    "test_unseen_rmse_intercept": round(rmse(m0, te_unseen), 3),
    "test_unseen_rmse_rapm": round(rmse(m, te_unseen), 3),
}
print(json.dumps(out, indent=1))
