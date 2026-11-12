"""Fetch one game and print the shape of its PBP and rotation payloads."""
import sys
from collections import Counter

sys.path.insert(0, "src")
from synergy import fetch  # noqa: E402

log = fetch.season_game_log("2023-24")
print(log.shape, log.columns.tolist())
gid = sys.argv[1] if len(sys.argv) > 1 else log["GAME_ID"].iloc[0]
print("game", gid)

pbp = fetch.play_by_play(gid)
actions = pbp["game"]["actions"]
print("pbp keys", list(pbp["game"].keys()), len(actions))
print("action keys", list(actions[0].keys()))
print(Counter((a["actionType"], a["subType"]) for a in actions).most_common(80))
for a in actions[:12]:
    print({k: a[k] for k in ("actionNumber", "clock", "period", "teamId", "personId",
                             "actionType", "subType", "description", "scoreHome", "scoreAway",
                             "location", "shotResult", "isFieldGoal")})

rot = fetch.game_rotation(gid)
print([s["name"] for s in rot["resultSets"]])
print(fetch.result_set(rot, "HomeTeam").head())
