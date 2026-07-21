"""Match-results data layer for the forecaster (competition-agnostic).

The unit of data here is a **match result** (teams, goals, date, neutral venue,
tournament), NOT a shot event — so this is a separate loader from the xG side.
The scoreline model is fit on a broad pool of international matches; a specific
*competition* (e.g. the 2026 World Cup) is then simulated on top of those fitted
strengths. Keeping team-name normalization in one place lets the model, the
simulator and the frontend all refer to teams identically.

Source
------
`martj42/international_results` — ~49k international matches since 1872, updated
continuously, no auth. Columns: date, home_team, away_team, home_score,
away_score, tournament, city, country, neutral. The `neutral` flag is what lets
us apply home advantage correctly (World Cup games are neutral except the
hosts'), and `tournament` gives a match-importance hierarchy for down-weighting
friendlies. We fetch it live (with a cache) and commit a recent snapshot so the
deployed app starts offline, mirroring how the xG model is committed.

The **as-of clock** (`as_of`) is the spine of the live forecaster: every query
uses only results dated on/before `as_of`, so as games finish the tournament
state — who qualified, who's still alive — advances and predictions update.

Run:  python -m forecaster.data
"""

from __future__ import annotations

import csv
import json
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SNAPSHOT_PATH = ARTIFACTS / "results_snapshot.csv"  # committed offline fallback


def _resolve_cache_dir() -> Path:
    """Repo data dir when writable (local dev), else a temp dir (e.g. a read-only
    container filesystem on Hugging Face). Either way the committed snapshot is
    the offline fallback, so a non-writable cache never breaks serving."""
    primary = _ROOT / "data" / "forecaster"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        (primary / ".wtest").write_text("")
        (primary / ".wtest").unlink()
        return primary
    except OSError:
        d = Path(tempfile.gettempdir()) / "soccerboard-forecaster"
        d.mkdir(parents=True, exist_ok=True)
        return d


CACHE_DIR = _resolve_cache_dir()
CACHE_PATH = CACHE_DIR / "results.csv"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
SNAPSHOT_SINCE = "2000-01-01"  # committed snapshot is trimmed to keep the repo lean

# 2026 World Cup kicked off 2026-06-11; the pre-tournament strength fit uses only
# matches strictly before this, so the knockout forecast never sees tournament
# results (a team's group games decide *who* advances, never its strength).
WC2026_KICKOFF = "2026-06-11"


# --- Team-name normalization -------------------------------------------------
# Map historical / variant names onto the current national side so a team's
# whole history feeds one set of strengths. Kept deliberately small: with time
# decay, pre-2000 matches barely move the fit, so only continuity cases that
# matter are mapped. The martj42 names are otherwise treated as canonical.
TEAM_ALIASES: dict[str, str] = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Zaïre": "DR Congo",
    "Zaire": "DR Congo",
    "Congo DR": "DR Congo",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Chinese Taipei": "Taiwan",
}


def normalize_team(name: str) -> str:
    name = (name or "").strip()
    return TEAM_ALIASES.get(name, name)


# --- Match-importance weights ------------------------------------------------
# Friendlies are noisy (rotated squads, low stakes); competitive matches carry
# more signal. These multiply the time-decay weight at fit time. Documented and
# configurable rather than hidden constants.
def importance_weight(tournament: str) -> float:
    t = (tournament or "").lower()
    if "friendly" in t:
        return 0.5
    if "qualification" in t or "qualifier" in t:
        return 0.9
    if "nations league" in t:
        return 0.9
    # World Cup, Euro, Copa America, AFCON, Asian Cup, Gold Cup finals, etc.
    return 1.0


@dataclass(frozen=True)
class Match:
    date: str
    home: str
    away: str
    home_goals: int | None
    away_goals: int | None
    neutral: bool
    tournament: str

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


# --- Loading -----------------------------------------------------------------
def _csv_stats(path: Path) -> tuple[int, str]:
    """(dated-row count, newest date) of a results CSV — the signature used to
    reject a download that would regress the cache. Raises if the file isn't a
    usable results feed (unparseable, no `date` column, or no dated rows), so a
    garbage response (e.g. an HTML error page) never passes validation."""
    n, max_date = 0, ""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            raise ValueError("results CSV missing 'date' column")
        for row in reader:
            d = row["date"]
            if d:
                n += 1
                if d > max_date:
                    max_date = d
    if n == 0:
        raise ValueError("results CSV has no dated rows")
    return n, max_date


def fetch_results(force: bool = False, ttl_seconds: float = 3600.0) -> Path:
    """Download the latest results CSV into the cache. Returns the path actually
    used. Falls back to the committed snapshot if the network is unavailable so
    the app always has data offline. `ttl_seconds` skips re-download if the cache
    is fresh — this is the knob the live serving layer leans on.

    The download is validated and swapped in atomically. The feed is date-sorted
    ascending, so a truncated or rolled-back response silently drops its newest
    rows — the current tournament — which used to blank live results until the
    next clean fetch. Such a response (smaller row count or an older newest date
    than the current cache) is now rejected in favour of the existing cache, and
    a good download replaces the cache via an atomic rename, so a bad fetch can
    never corrupt or partially overwrite it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fresh = (
        CACHE_PATH.exists()
        and (time.time() - CACHE_PATH.stat().st_mtime) < ttl_seconds
    )
    if fresh and not force:
        return CACHE_PATH
    try:
        with urllib.request.urlopen(RESULTS_URL, timeout=20) as resp:
            body = resp.read()
        fd, tmp_name = tempfile.mkstemp(dir=CACHE_DIR, prefix="results.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with open(fd, "wb") as fh:
                fh.write(body)
            new_n, new_max = _csv_stats(tmp)  # rejects a garbage/empty download
            try:
                cur_n, cur_max = _csv_stats(CACHE_PATH) if CACHE_PATH.exists() else (0, "")
            except Exception:
                cur_n, cur_max = 0, ""  # unreadable cache: let the fresh download replace it
            if new_max < cur_max or new_n < cur_n * 0.9:
                raise ValueError(
                    f"rejecting feed regression: rows {new_n} vs {cur_n}, "
                    f"newest {new_max!r} vs {cur_max!r}"
                )
            tmp.replace(CACHE_PATH)  # atomic on the same filesystem
        finally:
            tmp.unlink(missing_ok=True)
        return CACHE_PATH
    except Exception:
        if CACHE_PATH.exists():
            return CACHE_PATH
        if SNAPSHOT_PATH.exists():
            return SNAPSHOT_PATH
        raise


def _source_path(prefer_live: bool, ttl_seconds: float) -> Path:
    if prefer_live:
        return fetch_results(ttl_seconds=ttl_seconds)
    for p in (CACHE_PATH, SNAPSHOT_PATH):
        if p.exists():
            return p
    return fetch_results(ttl_seconds=ttl_seconds)


def load_matches(
    as_of: str | None = None,
    since: str | None = None,
    played_only: bool = True,
    prefer_live: bool = False,
    ttl_seconds: float = 3600.0,
    path: Path | None = None,
) -> list[Match]:
    """Normalized international matches.

    as_of       : keep only matches dated <= this ISO date (the live clock).
    since       : keep only matches dated >= this ISO date.
    played_only : drop scheduled fixtures whose score isn't in yet.
    """
    src = path or _source_path(prefer_live, ttl_seconds)
    out: list[Match] = []
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["date"]
            if as_of and d > as_of:
                continue
            if since and d < since:
                continue
            hg, ag = row["home_score"], row["away_score"]
            has_score = hg not in ("", "NA", None) and ag not in ("", "NA", None)
            if played_only and not has_score:
                continue
            home, away = normalize_team(row["home_team"]), normalize_team(row["away_team"])
            # The feed schedules a knockout fixture before both sides are known,
            # naming the undetermined side with the same "NA" null it uses for an
            # unplayed score. That row isn't a fixture between two teams yet, and
            # the phantom "NA" team corrupts anything clustering teams by fixture.
            if home in ("", "NA") or away in ("", "NA"):
                continue
            out.append(
                Match(
                    date=d,
                    home=home,
                    away=away,
                    home_goals=int(hg) if has_score else None,
                    away_goals=int(ag) if has_score else None,
                    neutral=str(row.get("neutral", "")).upper() == "TRUE",
                    tournament=row.get("tournament", ""),
                )
            )
    return out


def tournament_matches(
    tournament: str, season_year: str, as_of: str | None = None, **kw
) -> list[Match]:
    """All matches of one tournament edition (e.g. the 2026 World Cup), including
    unplayed scheduled fixtures. Used to read the live bracket state."""
    matches = load_matches(
        as_of=as_of, since=f"{season_year}-01-01", played_only=False, **kw
    )
    return [m for m in matches if m.tournament == tournament]


# --- Competition registry ----------------------------------------------------
# A "competition" is a format config (groups, bracket, hosts) the simulator runs
# on top of the shared scoreline model. Only the World Cup is populated; the
# registry is what keeps the API and frontend competition-parameterized so
# leagues / Champions League are additive later.
def list_competitions() -> list[dict]:
    out = []
    for cfg_path in sorted(ARTIFACTS.glob("*.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("kind") == "competition":
            out.append({"id": cfg["id"], "name": cfg["name"], "format": cfg["format"]})
    return out


def get_competition(competition_id: str) -> dict:
    path = ARTIFACTS / f"{competition_id}.json"
    if not path.exists():
        raise KeyError(f"Unknown competition: {competition_id}")
    cfg = json.loads(path.read_text())
    if cfg.get("kind") != "competition":
        raise KeyError(f"{competition_id} is not a competition config")
    return cfg


def teams_for(competition_id: str) -> list[str]:
    cfg = get_competition(competition_id)
    teams: list[str] = []
    for members in cfg["groups"].values():
        teams.extend(members)
    return sorted(teams)


def main() -> None:
    matches = load_matches(since="2018-01-01")
    print(f"Loaded {len(matches)} played matches since 2018")
    teams = {m.home for m in matches} | {m.away for m in matches}
    print(f"Distinct teams: {len(teams)}")
    last = sorted(matches, key=lambda m: m.date)[-5:]
    print("Most recent results:")
    for m in last:
        venue = "(N)" if m.neutral else ""
        print(
            f"  {m.date}  {m.home} {m.home_goals}-{m.away_goals} {m.away} "
            f"{venue}  [{m.tournament}]"
        )
    try:
        comps = list_competitions()
        print(f"\nCompetitions registered: {[c['id'] for c in comps]}")
        for c in comps:
            print(f"  {c['id']}: {len(teams_for(c['id']))} teams ({c['format']})")
    except Exception as e:
        print(f"\n(no competition configs yet: {e})")


if __name__ == "__main__":
    main()
