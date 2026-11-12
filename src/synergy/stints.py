"""Build stints (same ten players on the floor) from PlayByPlayV3 + GameRotation.

Lineups: PBP substitution events only carry the outgoing player, so the
on-court set after any substitution is read from GameRotation (in/out times
in tenths of a second from tip-off). PBP order still decides *when* a
lineup change takes effect, which is what the free-throw rule needs.

Rules implemented (see DECISIONS.md, "Possession and split rules"):
  * A possession ends on a made FG (or at the and-one FT that follows it),
    a defensive rebound, a turnover, a made last FT of a trip, or the end
    of a period if the team had the ball.
  * Technical, flagrant and clear-path FTs never end a possession.
  * Subs that happen during an FT trip take effect after the trip resolves,
    so all of the trip's points and possessions go to the foul lineup.
  * A new period does not start a new stint if the same ten players stay on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .fetch import result_set

PERIOD_LEN = {True: 7200, False: 3000}  # tenths: regulation / overtime

# Action types that can sit between a play and its follow-up without
# changing who has the ball.
PASSIVE = {"Substitution", "Timeout", "", "Instant Replay", "Stoppage",
           "Ejection", "Foul", "Violation"}

FT_RE = re.compile(r"(\d) of (\d)")


def period_start(period: int) -> int:
    return (period - 1) * 7200 if period <= 4 else 28800 + (period - 5) * 3000


def clock_seconds(clock: str) -> float:
    """'PT06M07.00S' -> 367.0 seconds remaining."""
    m = re.match(r"PT(\d+)M([\d.]+)S", clock)
    return int(m.group(1)) * 60 + float(m.group(2))


def elapsed_tenths(period: int, clock: str) -> int:
    length = PERIOD_LEN[period <= 4]
    return period_start(period) + length - round(clock_seconds(clock) * 10)


def ft_info(a: dict):
    """Return (k, n, ends_possession_eligible) for a free throw, or None."""
    sub = a["subType"] or ""
    if "Technical" in sub:
        return None
    m = FT_RE.search(sub)
    if not m:
        return None
    retains_ball = "Flagrant" in sub or "Clear Path" in sub
    return int(m.group(1)), int(m.group(2)), not retains_ball


def is_made_ft(a: dict) -> bool:
    return not a["description"].startswith("MISS")


# ---- rotation ---------------------------------------------------------------

class Rotation:
    def __init__(self, rot_json: dict):
        frames = [result_set(rot_json, n) for n in ("HomeTeam", "AwayTeam")]
        df = pd.concat(frames, ignore_index=True)
        df["PERSON_ID"] = df["PERSON_ID"].astype(str)
        df["TEAM_ID"] = df["TEAM_ID"].astype(str)
        self.df = df

    def on_court(self, team_id: str, t: int) -> frozenset[str]:
        """Players on the floor just after time t (tenths)."""
        d = self.df
        q = t + 1
        m = (d.TEAM_ID == team_id) & (d.IN_TIME_REAL <= q) & (d.OUT_TIME_REAL > q)
        return frozenset(d.loc[m, "PERSON_ID"])


# ---- stint accumulation -------------------------------------------------------

@dataclass
class Stint:
    home: frozenset
    away: frozenset
    period: int
    start_t: int
    start_clock: float
    start_margin: int          # home - away at stint start
    end_t: int = 0
    pts_home: int = 0
    pts_away: int = 0
    poss_home: int = 0
    poss_away: int = 0
    notes: list = field(default_factory=list)


class GameParser:
    def __init__(self, game_id, home_id, away_id, pbp_json, rot_json):
        self.game_id = game_id
        self.home, self.away = str(home_id), str(away_id)
        self.actions = pbp_json["game"]["actions"]
        self.rot = Rotation(rot_json)
        self.stints: list[Stint] = []
        self.issues: list[str] = []
        self.score = [0, 0]  # home, away
        # possession state
        self.pending_fg = None     # (team, stint) after a made FG, awaiting and-one
        self.pending_miss = None   # (team, stint) after a miss, awaiting rebound
        self.open_team = None      # team with the ball now
        self.open_clock = 0.0      # clock (s remaining) when it got the ball
        # sub deferral during FT trips
        self.deferred_t = None
        self.trip_open = False
        # team offensive rebounds: excluded from box-score OREB, needed by the check
        self.team_oreb = {"home": 0, "away": 0}

    # -- helpers --
    def side(self, team) -> str:
        return "home" if str(team) == self.home else "away"

    def other(self, team) -> str:
        return self.away if str(team) == self.home else self.home

    @property
    def cur(self) -> Stint | None:
        return self.stints[-1] if self.stints else None

    def end_poss(self, team, stint=None, clock=0.0):
        s = stint or self.cur
        setattr(s, f"poss_{self.side(team)}", getattr(s, f"poss_{self.side(team)}") + 1)
        self.open_team = self.other(team)
        self.open_clock = clock

    def set_lineups(self, t, period, clock):
        h = self.rot.on_court(self.home, t)
        a = self.rot.on_court(self.away, t)
        if len(h) != 5 or len(a) != 5:
            self.issues.append(f"lineup size home={len(h)} away={len(a)} at t={t}")
        c = self.cur
        if c and c.home == h and c.away == a:
            return
        if c:
            c.end_t = t
        self.stints.append(Stint(h, a, period, t, clock, self.score[0] - self.score[1]))

    def trip_follows(self, i) -> bool:
        """Is the next ball-relevant action after index i a (non-tech) FT trip shot?"""
        for a in self.actions[i + 1:]:
            if a["actionType"] == "Free Throw":
                if ft_info(a) is not None:
                    return True
                continue  # technical FT: keep looking
            if a["actionType"] in PASSIVE:
                continue
            return False
        return False

    def flush_deferred(self, period, clock):
        if self.deferred_t is not None and not self.trip_open:
            self.set_lineups(self.deferred_t, period, clock)
            self.deferred_t = None

    def resolve_pending_fg(self):
        if self.pending_fg:
            team, stint = self.pending_fg
            self.end_poss(team, stint, self.open_clock_hint)
            self.pending_fg = None

    # -- main loop --
    def parse(self) -> list[dict]:
        for i, a in enumerate(self.actions):
            typ, period = a["actionType"], a["period"]
            clock = clock_seconds(a["clock"])
            t = elapsed_tenths(period, a["clock"])
            team = str(a["teamId"]) if a["teamId"] else None
            if team is None and str(a["personId"]) in (self.home, self.away):
                team = str(a["personId"])  # team rebounds/turnovers: teamId 0, personId = team
            self.open_clock_hint = clock

            if typ == "period" and a["subType"] == "start":
                self.set_lineups(t, period, clock)
                self.open_team, self.pending_miss, self.pending_fg = None, None, None
                continue

            if typ == "Substitution":
                if self.trip_open or self.trip_follows(i):
                    self.deferred_t = t
                    self.trip_open = True
                else:
                    self.set_lineups(t, period, clock)
                continue

            # and-one check before resolving a pending made FG
            fi = ft_info(a) if typ == "Free Throw" else None
            if self.pending_fg and typ not in PASSIVE:
                and_one = (fi is not None and fi[:2] == (1, 1)
                           and team == self.pending_fg[0])
                if and_one:
                    self.pending_fg = None   # possession now ends at this FT
                elif not (typ == "Free Throw" and fi is None):  # skip technicals
                    self.resolve_pending_fg()

            # Live play after a trip (e.g. a missed last FT with no recorded
            # rebound) closes the trip; deferred subs take effect first.
            if self.trip_open and typ in ("Made Shot", "Missed Shot", "Turnover", "Jump Ball"):
                self.trip_open = False
                self.flush_deferred(period, clock)

            self._add_points(a)

            if typ == "Made Shot":
                self.pending_miss = None
                self.pending_fg = (team, self.cur)
                self.open_team = team
            elif typ == "Missed Shot":
                self.pending_miss = (team, self.cur)
                self.open_team = team
            elif typ == "Rebound":
                if self.pending_miss and team:
                    miss_team, stint = self.pending_miss
                    if team != miss_team:
                        self.end_poss(miss_team, stint, clock)
                    elif not a["teamId"]:
                        self.team_oreb[self.side(team)] += 1
                    self.open_team = team
                    self.pending_miss = None
                    if self.trip_open and self.deferred_t is not None:
                        self.trip_open = False
                        self.flush_deferred(period, clock)
            elif typ == "Turnover" and team and "No Turnover" not in (a["subType"] or ""):
                self.pending_miss = None
                self.end_poss(team, None, clock)
            elif typ == "Free Throw" and fi is not None:
                k, n, can_end = fi
                self.open_team = team
                if k < n:
                    self.trip_open = True
                elif can_end:  # last FT of a trip
                    if is_made_ft(a):
                        self.end_poss(team, None, clock)
                        self.trip_open = False
                    else:
                        self.pending_miss = (team, self.cur)  # trip resolves at rebound
                else:  # flagrant / clear path: ball retained, trip over
                    self.trip_open = False
                if not self.trip_open:
                    self.flush_deferred(period, clock)
            elif typ == "period" and a["subType"] == "end":
                self.resolve_pending_fg()
                if self.pending_miss:
                    self.end_poss(self.pending_miss[0], self.pending_miss[1], 0.0)
                elif self.open_team and self.open_clock > 0:
                    self.end_poss(self.open_team, None, 0.0)
                self.pending_miss = None
                self.trip_open = False
                self.flush_deferred(period, clock)
                self.cur.end_t = t

        return [self._row(k, s) for k, s in enumerate(self.stints)]

    def _add_points(self, a):
        if a["scoreHome"] == "" or a["scoreAway"] == "":
            return
        h, w = int(a["scoreHome"]), int(a["scoreAway"])
        dh, dw = h - self.score[0], w - self.score[1]
        if dh or dw:
            self.cur.pts_home += dh
            self.cur.pts_away += dw
            self.score = [h, w]

    def _row(self, k: int, s: Stint) -> dict:
        n_s = (s.poss_home + s.poss_away) / 2
        return {
            "game_id": self.game_id,
            "stint_idx": k,
            "home_team_id": self.home,
            "away_team_id": self.away,
            "home_players": "|".join(sorted(s.home)),
            "away_players": "|".join(sorted(s.away)),
            "period": s.period,
            "start_t": s.start_t,
            "end_t": s.end_t,
            "seconds": (s.end_t - s.start_t) / 10,
            "start_clock": s.start_clock,
            "start_margin": s.start_margin,
            "pts_home": s.pts_home,
            "pts_away": s.pts_away,
            "poss_home": s.poss_home,
            "poss_away": s.poss_away,
            "n_s": n_s,
            "y_s": 100 * (s.pts_home - s.pts_away) / n_s if n_s > 0 else float("nan"),
        }


def parse_game(game_id, home_id, away_id, pbp_json, rot_json):
    """Return (stints DataFrame, issue strings, team offensive rebounds by side)."""
    p = GameParser(game_id, home_id, away_id, pbp_json, rot_json)
    df = pd.DataFrame(p.parse())
    return df, p.issues, p.team_oreb
