"""Player-level contribution model: per-player attack/defense deltas fitted on
top of the Dixon-Coles team base parameters.

Stage-2 fit: given already-fitted DC team params, estimate how much each player
moves their team's expected goals above or below the team average. Uses lineup
data from StatsBomb open-data for major international tournaments (WC 2018 &
2022, Euro 2020 & 2024, Copa America 2024, AFCON 2023).

At serving time, when a player is listed in news_items.json as absent, their
learned delta is subtracted from their team's effective attack/defense strength.
No rubric tiers needed — the magnitude comes directly from the fit.

Fetch and fit:
  python -m forecaster.player_model   # downloads lineups, fits deltas, prints top players

Model definition
----------------
For a match where lineups are known:
  log(lambda) = att_base[H] + SUM(att_delta[p] for p in H_starters)
              - def_base[A] - SUM(def_delta[p] for p in A_starters)
              + home_adv * (not neutral)
  log(mu)     = att_base[A] + SUM(att_delta[p] for p in A_starters)
              - def_base[H] - SUM(def_delta[p] for p in H_starters)

att_delta[p] > 0: player helps their team score more (in log-goal space).
def_delta[p] > 0: player helps their team concede less.

Both are regularised toward zero: an absent player with no historical lineup
data has delta≈0, meaning no adjustment — which is the right default.

When player p is absent at prediction time:
  att_adj[team] -= att_delta[p]
  def_adj[team] -= def_delta[p]
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from forecaster.data import normalize_team, CACHE_DIR
from forecaster.dixon_coles import Params

# StatsBomb open-data raw GitHub base URL
_SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# (competition_id, season_id) tuples to use for lineup data.
# Covers all major international confederations' best recent players.
LINEUP_COMPETITIONS: list[tuple[int, int]] = [
    (43, 106),    # FIFA World Cup 2022
    (43, 3),      # FIFA World Cup 2018
    (55, 282),    # UEFA Euro 2024
    (55, 43),     # UEFA Euro 2020
    (223, 282),   # Copa America 2024
    (1267, 107),  # African Cup of Nations 2023
]

# Map StatsBomb team names -> canonical martj42 names where they differ.
_SB_TEAM_MAP: dict[str, str] = {
    "United States": "United States",
    "Iran": "IR Iran",
    "Côte D'Ivoire": "Ivory Coast",
    "Cote D'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland",
    "Türkiye": "Turkey",
    "North Macedonia": "North Macedonia",
    "Slovak Republic": "Slovakia",
    "Czech Republic": "Czech Republic",
    "DR Congo": "DR Congo",
    "Cabo Verde": "Cape Verde",
}


def _sb_team(name: str) -> str:
    mapped = _SB_TEAM_MAP.get(name, name)
    return normalize_team(mapped)


def normalize_player_name(name: str) -> str:
    """Accent-strip + lowercase, for fuzzy lookup from news_items.json."""
    nfkd = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.lower().strip()


# --- Data layer ---------------------------------------------------------------

LINEUPS_CACHE = CACHE_DIR / "statsbomb_lineups.json"
_FETCH_DELAY = 1.2   # seconds between StatsBomb requests (polite crawl)


def _fetch_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


@dataclass(frozen=True)
class LineupMatch:
    date: str          # ISO date
    home: str          # canonical team name
    away: str
    home_goals: int
    away_goals: int
    neutral: bool
    home_starters: tuple[str, ...]  # StatsBomb full player names (Starting XI)
    away_starters: tuple[str, ...]
    home_nicknames: tuple[str | None, ...]  # StatsBomb player_nickname (may be None)
    away_nicknames: tuple[str | None, ...]

    @staticmethod
    def _compat(d: dict) -> "LineupMatch":
        """Load from JSON, back-filling nickname tuples if absent (old cache)."""
        n = len(d.get("home_starters", []))
        m = len(d.get("away_starters", []))
        return LineupMatch(
            date=d["date"], home=d["home"], away=d["away"],
            home_goals=d["home_goals"], away_goals=d["away_goals"],
            neutral=d["neutral"],
            home_starters=tuple(d["home_starters"]),
            away_starters=tuple(d["away_starters"]),
            home_nicknames=tuple(d.get("home_nicknames") or [None] * n),
            away_nicknames=tuple(d.get("away_nicknames") or [None] * m),
        )


def _parse_lineup(team_obj: dict) -> tuple[list[str], list[str | None]]:
    """Extract starting-XI player names and nicknames from a StatsBomb lineup."""
    names, nicks = [], []
    for player in team_obj.get("lineup", []):
        positions = player.get("positions", [])
        if any(pos.get("start_reason") == "Starting XI" for pos in positions):
            names.append(player["player_name"])
            nicks.append(player.get("player_nickname") or None)
    return names, nicks


def fetch_lineup_matches(
    competitions: list[tuple[int, int]] = LINEUP_COMPETITIONS,
    cache_path: Path = LINEUPS_CACHE,
    force: bool = False,
) -> list[LineupMatch]:
    """Download StatsBomb lineup data for all competitions and return parsed
    LineupMatch objects. Caches to disk; skips download if cache is fresh."""
    if cache_path.exists() and not force:
        raw = json.loads(cache_path.read_text())
        return [LineupMatch._compat(m) for m in raw]

    all_matches: list[LineupMatch] = []
    for comp_id, season_id in competitions:
        label = f"comp={comp_id} season={season_id}"
        print(f"  Fetching matches for {label}...")
        matches_url = f"{_SB_BASE}/matches/{comp_id}/{season_id}.json"
        try:
            match_list = _fetch_json(matches_url)
        except Exception as exc:
            print(f"    WARNING: could not fetch {matches_url}: {exc}")
            continue

        for m in match_list:
            mid = m["match_id"]
            home_name = _sb_team(m["home_team"]["home_team_name"])
            away_name = _sb_team(m["away_team"]["away_team_name"])
            home_goals = m.get("home_score", 0) or 0
            away_goals = m.get("away_score", 0) or 0
            date_str = m["match_date"]

            time.sleep(_FETCH_DELAY)
            lineup_url = f"{_SB_BASE}/lineups/{mid}.json"
            try:
                lineup_data = _fetch_json(lineup_url)
            except Exception as exc:
                print(f"    WARNING: lineup {mid} failed: {exc}")
                continue

            home_starters, home_nicks = [], []
            away_starters, away_nicks = [], []
            for team_obj in lineup_data:
                tname = _sb_team(team_obj["team_name"])
                names, nicks = _parse_lineup(team_obj)
                if tname == home_name:
                    home_starters, home_nicks = names, nicks
                elif tname == away_name:
                    away_starters, away_nicks = names, nicks

            if not home_starters or not away_starters:
                continue  # malformed lineup; skip

            all_matches.append(LineupMatch(
                date=date_str,
                home=home_name,
                away=away_name,
                home_goals=home_goals,
                away_goals=away_goals,
                neutral=True,  # all StatsBomb intl comps are neutral-venue tournaments
                home_starters=tuple(home_starters),
                away_starters=tuple(away_starters),
                home_nicknames=tuple(home_nicks),
                away_nicknames=tuple(away_nicks),
            ))

        print(f"    {len(all_matches)} total matches so far")

    # Persist cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([asdict(m) for m in all_matches], indent=2))
    print(f"  Cached {len(all_matches)} lineup matches to {cache_path}")
    return all_matches


# --- Model layer --------------------------------------------------------------

@dataclass
class PlayerDeltas:
    """Per-player attack/defense deltas in log-goal space.

    att_delta[i] > 0 means player i improves their team's expected goals scored.
    def_delta[i] > 0 means player i improves their team's expected goals kept out.

    When the player is absent, subtract their deltas from the team's effective
    attack and defense strengths.
    """
    players: list[str]                 # canonical player names (StatsBomb full names)
    att_delta: list[float]
    def_delta: list[float]
    n_matches: list[int]               # lineup appearances used in fit
    reg: float
    n_lineup_matches: int              # total lineup matches used
    nicknames: list[str | None] | None = None  # StatsBomb player_nickname per player

    def index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.players)}

    def normalized_index(self) -> dict[str, int]:
        """Accent-stripped lowercase full name -> index."""
        return {normalize_player_name(p): i for i, p in enumerate(self.players)}

    def nickname_index(self) -> dict[str, int]:
        """Accent-stripped lowercase nickname -> index (skips None entries)."""
        if not self.nicknames:
            return {}
        return {
            normalize_player_name(n): i
            for i, n in enumerate(self.nicknames)
            if n is not None
        }

    def find_player(self, name: str) -> int | None:
        """Look up a player by common name, with multiple fallback strategies:
        1. Exact match on full name (accent-stripped, lowercase)
        2. Match on StatsBomb nickname
        3. Token subset: all tokens in `name` appear as prefixes of tokens in full name
        Returns the player index or None if not found."""
        q = normalize_player_name(name)
        # Strategy 1: exact normalized match
        pi = self.normalized_index().get(q)
        if pi is not None:
            return pi
        # Strategy 2: nickname match
        pi = self.nickname_index().get(q)
        if pi is not None:
            return pi
        # Strategy 3: token subset — all query tokens appear as a prefix of some
        # token in the full name (handles "Rodrygo" -> "rodrygo silva de goes")
        q_tokens = q.split()
        for i, p in enumerate(self.players):
            p_tokens = normalize_player_name(p).split()
            if all(any(pt.startswith(qt) for pt in p_tokens) for qt in q_tokens):
                return i
        return None

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path) -> "PlayerDeltas":
        d = json.loads(Path(path).read_text())
        d.setdefault("nicknames", None)
        return cls(**d)


def fit_player_deltas(
    dc_params: Params,
    lineup_matches: list[LineupMatch],
    reg: float = 1.0,
) -> PlayerDeltas:
    """Fit per-player attack/defense deltas (Stage 2 fit).

    Holds DC team params fixed and optimises player delta parameters to
    maximise the Dixon-Coles log-likelihood on the lineup-covered matches,
    with L2 regularisation toward zero (no effect).

    reg controls regularisation strength. Higher values → player deltas closer
    to zero (more conservative). Default 1.0 is appropriate for the sparse
    international-lineup dataset (~5–25 appearances per player).
    """
    from scipy.optimize import minimize
    from scipy.special import gammaln

    team_idx = dc_params.index()
    att_base = np.array(dc_params.attack)
    def_base = np.array(dc_params.defense)
    gamma = dc_params.home_adv
    rho = dc_params.rho

    # Build player index from all starters across all lineup matches
    all_players = sorted({
        p for m in lineup_matches
        for p in list(m.home_starters) + list(m.away_starters)
    })
    pidx = {p: i for i, p in enumerate(all_players)}
    n_p = len(all_players)

    # Filter to matches where both teams are in DC params
    valid = [
        m for m in lineup_matches
        if m.home in team_idx and m.away in team_idx
        and len(m.home_starters) >= 10 and len(m.away_starters) >= 10
    ]
    M = len(valid)
    if M == 0:
        raise ValueError("no valid lineup matches (teams not in DC params?)")

    # Per-match arrays
    h_idx = np.array([team_idx[m.home] for m in valid])
    a_idx = np.array([team_idx[m.away] for m in valid])
    x = np.array([m.home_goals for m in valid], dtype=float)
    y = np.array([m.away_goals for m in valid], dtype=float)
    hf = np.zeros(M)   # all StatsBomb intl matches are neutral venues

    lgx = gammaln(x + 1.0)
    lgy = gammaln(y + 1.0)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    # Starter indicator matrices (M × n_players)
    home_mat = np.zeros((M, n_p))
    away_mat = np.zeros((M, n_p))
    for i, m in enumerate(valid):
        for p in m.home_starters:
            if p in pidx:
                home_mat[i, pidx[p]] = 1.0
        for p in m.away_starters:
            if p in pidx:
                away_mat[i, pidx[p]] = 1.0

    # Base expected goals from team params (fixed throughout Stage 2)
    base_lam = np.exp(att_base[h_idx] - def_base[a_idx] + gamma * hf)
    base_mu  = np.exp(att_base[a_idx] - def_base[h_idx])

    def objective(params: np.ndarray):
        att_d = params[:n_p]
        def_d = params[n_p:]

        # Player adjustments stacked onto the team-level base
        lam = base_lam * np.exp(home_mat @ att_d - away_mat @ def_d)
        mu  = base_mu  * np.exp(away_mat @ att_d - home_mat @ def_d)

        # Dixon-Coles low-score correction
        tau = np.ones(M)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.maximum(tau, 1e-10)

        ll = (
            x * np.log(lam) - lam - lgx
            + y * np.log(mu)  - mu  - lgy
            + np.log(tau)
        )
        penalty = reg * (np.dot(att_d, att_d) + np.dot(def_d, def_d))
        nll = -np.sum(ll) + penalty

        # Gradient of NLL w.r.t. player deltas
        dt_dl = np.zeros(M)
        dt_dl[m00] = -mu[m00] * rho
        dt_dl[m01] = rho
        dt_dm = np.zeros(M)
        dt_dm[m00] = -lam[m00] * rho
        dt_dm[m10] = rho

        # u = dLL/d(log lam), v = dLL/d(log mu)
        u = x - lam + lam / tau * dt_dl
        v = y - mu  + mu  / tau * dt_dm

        # dLL/d(att_d[p]) = home_mat[:,p]·u + away_mat[:,p]·v
        # dNLL/d(att_d) = -(home_mat.T @ u + away_mat.T @ v) + 2*reg*att_d
        g_att = -(home_mat.T @ u + away_mat.T @ v) + 2.0 * reg * att_d

        # dLL/d(def_d[p]) = -(away_mat[:,p]·u + home_mat[:,p]·v)
        # dNLL/d(def_d) = (away_mat.T @ u + home_mat.T @ v) + 2*reg*def_d
        g_def = (away_mat.T @ u + home_mat.T @ v) + 2.0 * reg * def_d

        return nll, np.concatenate([g_att, g_def])

    res = minimize(
        objective, np.zeros(2 * n_p), method="L-BFGS-B", jac=True,
        options={"maxiter": 1000, "maxfun": 500000},
    )

    att_d = res.x[:n_p]
    def_d = res.x[n_p:]

    appearances = (home_mat + away_mat).sum(axis=0).astype(int)

    # Collect one nickname per player (from any match they appeared in)
    nick_map: dict[str, str | None] = {}
    for m in valid:
        for p, n in zip(m.home_starters, m.home_nicknames):
            if p not in nick_map and n:
                nick_map[p] = n
        for p, n in zip(m.away_starters, m.away_nicknames):
            if p not in nick_map and n:
                nick_map[p] = n
    nicks = [nick_map.get(p) for p in all_players]

    return PlayerDeltas(
        players=all_players,
        att_delta=att_d.tolist(),
        def_delta=def_d.tolist(),
        n_matches=appearances.tolist(),
        reg=reg,
        n_lineup_matches=M,
        nicknames=nicks,
    )


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    from forecaster import dixon_coles as dc
    from forecaster.data import load_matches, WC2026_KICKOFF, ARTIFACTS
    from pathlib import Path

    print("Fetching StatsBomb lineup data...")
    lineup_matches = fetch_lineup_matches(force=False)
    print(f"  {len(lineup_matches)} lineup matches loaded\n")

    print("Loading DC team params...")
    params_path = ARTIFACTS / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            "No DC params found. Run: python -m forecaster.build_artifacts first."
        )
    dc_params = dc.Params.from_json(params_path)
    print(f"  {len(dc_params.teams)} teams\n")

    print("Fitting player deltas...")
    deltas = fit_player_deltas(dc_params, lineup_matches)
    print(f"  {len(deltas.players)} players, {deltas.n_lineup_matches} valid matches\n")

    out_path = ARTIFACTS / "player_deltas.json"
    deltas.to_json(out_path)
    print(f"Player deltas written to {out_path}\n")

    # Print top players by attack delta
    pidx = sorted(range(len(deltas.players)), key=lambda i: -deltas.att_delta[i])
    print("Top 20 attack contributors (att_delta in log-goal space):")
    for i in pidx[:20]:
        p = deltas.players[i]
        ad = deltas.att_delta[i]
        dd = deltas.def_delta[i]
        n = deltas.n_matches[i]
        pct = round((math.exp(ad) - 1) * 100, 1)
        print(f"  {p:30s}  att={ad:+.3f} ({pct:+.1f}%)  def={dd:+.3f}  n={n}")

    print("\nTop 20 defense contributors:")
    didx = sorted(range(len(deltas.players)), key=lambda i: -deltas.def_delta[i])
    for i in didx[:20]:
        p = deltas.players[i]
        dd = deltas.def_delta[i]
        n = deltas.n_matches[i]
        pct = round((math.exp(dd) - 1) * 100, 1)
        print(f"  {p:30s}  def={dd:+.3f} ({pct:+.1f}%)  n={n}")


if __name__ == "__main__":
    main()
