"""Possession and lineup rules on hand-built play-by-play."""
from synergy.stints import parse_game

H, A = "1", "2"
HOME5 = ["h1", "h2", "h3", "h4", "h5"]
AWAY5 = ["a1", "a2", "a3", "a4", "a5"]
HEADERS = ["GAME_ID", "TEAM_ID", "TEAM_CITY", "TEAM_NAME", "PERSON_ID", "PLAYER_FIRST",
           "PLAYER_LAST", "IN_TIME_REAL", "OUT_TIME_REAL", "PLAYER_PTS", "PT_DIFF", "USG_PCT"]


def rotation(home_spans, away_spans):
    """spans: list of (person_id, in_tenths, out_tenths)."""
    def rows(team, spans):
        return [["g", int(team), "", "", p, "", "", i, o, 0, 0, 0] for p, i, o in spans]
    return {"resultSets": [
        {"name": "AwayTeam", "headers": HEADERS, "rowSet": rows(A, away_spans)},
        {"name": "HomeTeam", "headers": HEADERS, "rowSet": rows(H, home_spans)},
    ]}


def full(players, end=7200):
    return [(p, 0, end) for p in players]


class PBP:
    def __init__(self):
        self.actions, self.score = [], [0, 0]
        self.add("period", "start", team=0, clock="PT12M00.00S", score=True)

    def add(self, typ, sub="", team=H, desc="", clock="PT06M00.00S", person=None,
            pts=0, score=False):
        if pts:
            self.score[0 if team == H else 1] += pts
            score = True
        self.actions.append({
            "actionType": typ, "subType": sub, "period": 1, "clock": clock,
            "teamId": int(team) if team else 0,
            "personId": person if person else 0,
            "description": desc,
            "scoreHome": str(self.score[0]) if score else "",
            "scoreAway": str(self.score[1]) if score else "",
        })
        return self

    def end(self):
        return self.add("period", "end", team=0, clock="PT00M00.00S", score=True)

    def json(self):
        return {"game": {"actions": self.actions}}


def run(pbp, rot=None):
    rot = rot or rotation(full(HOME5), full(AWAY5))
    df, issues, team_oreb = parse_game("g", H, A, pbp.json(), rot)
    return df, issues, team_oreb


def test_made_fg_and_defensive_rebound_end_possessions():
    p = PBP()
    p.add("Made Shot", team=H, pts=2)                    # home poss 1 ends
    p.add("Missed Shot", team=A)
    p.add("Rebound", team=H)                             # away poss 1 ends (DREB)
    p.add("Turnover", "Bad Pass", team=H, clock="PT00M00.00S")  # home poss 2 ends, 0 s left
    df, issues, _ = run(p.end())
    assert issues == []
    assert (df.poss_home.sum(), df.poss_away.sum()) == (2, 1)


def test_team_offensive_rebound_continues_possession():
    p = PBP()
    p.add("Missed Shot", team=H)
    p.add("Rebound", team=0, person=H, desc="TEAM Rebound")  # team OREB: teamId 0
    p.add("Made Shot", team=H, pts=2)
    df, _, team_oreb = run(p.end())
    assert df.poss_home.sum() == 1
    assert team_oreb == {"home": 1, "away": 0}


def test_and_one_is_one_possession():
    p = PBP()
    p.add("Made Shot", team=H, pts=2)
    p.add("Foul", "Shooting", team=A)
    p.add("Free Throw", "Free Throw 1 of 1", team=H, desc="X Free Throw 1 of 1", pts=1)
    p.add("Made Shot", team=A, pts=2, clock="PT00M00.00S")
    df, _, _ = run(p.end())
    assert df.poss_home.sum() == 1
    assert df.pts_home.sum() == 3


def test_technical_free_throw_does_not_end_possession():
    p = PBP()
    p.add("Foul", "Technical", team=A)
    p.add("Free Throw", "Free Throw Technical", team=H, desc="X Free Throw Technical", pts=1)
    p.add("Made Shot", team=H, pts=2, clock="PT00M00.00S")
    df, _, _ = run(p.end())
    assert df.poss_home.sum() == 1


def test_sub_between_free_throws_credits_trip_to_foul_lineup():
    # h5 -> h6 between FT 1 and FT 2 at 6:00 (t = 3600 tenths).
    rot = rotation(full(HOME5[:4]) + [("h5", 0, 3600), ("h6", 3600, 7200)], full(AWAY5))
    p = PBP()
    p.add("Foul", "Shooting", team=A)
    p.add("Free Throw", "Free Throw 1 of 2", team=H, desc="X Free Throw 1 of 2", pts=1)
    p.add("Substitution", team=H, person="h5", desc="SUB: Six FOR Five")
    p.add("Free Throw", "Free Throw 2 of 2", team=H, desc="X Free Throw 2 of 2", pts=1)
    p.add("Made Shot", team=A, pts=2, clock="PT05M00.00S")
    df, issues, _ = run(p.end(), rot)
    assert issues == []
    assert len(df) == 2
    first, second = df.iloc[0], df.iloc[1]
    assert "h5" in first.home_players and "h6" in second.home_players
    assert (first.pts_home, first.poss_home) == (2, 1)   # whole trip -> foul lineup
    assert (second.pts_away, second.poss_away) == (2, 1)


def test_same_lineup_across_period_break_is_one_stint():
    rot = rotation(full(HOME5, 14400), full(AWAY5, 14400))
    p = PBP()
    p.add("Made Shot", team=H, pts=2, clock="PT00M00.00S")
    p.end()
    p.actions.append({**p.actions[0], "period": 2})     # start of period 2
    p.actions.append({**p.actions[-2], "period": 2})    # end of period 2
    df, _, _ = run(p, rot)
    assert len(df) == 1
