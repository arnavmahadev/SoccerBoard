"""Pull StatsBomb shots + freeze frames and convert them to our locked schema.

Each StatsBomb `Shot` event embeds a `shot_freeze_frame`: the (x, y) of every
player visible at the moment of the shot, tagged teammate/opponent, with a
position name we use to spot the goalkeeper. That maps 1:1 onto `GameState`.

Output: data/processed/shots.parquet — one row per shot, carrying the
serialized GameState plus the label and useful metadata (including StatsBomb's
own xG as a benchmark).

Run:  python -m xg.data.load
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from xg.data.schema import GameState, Player, PITCH_LENGTH, PITCH_WIDTH

warnings.filterwarnings("ignore")  # silence statsbombpy open-data credential notice

# 360-enabled tournaments with full, modern freeze-frame coverage.
COMPETITIONS: list[tuple[int, int, str]] = [
    (43, 106, "World Cup 2022"),
    (55, 43, "Euro 2020"),
]

PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"
OUT_PATH = PROCESSED / "shots.parquet"


def _clamp(v: float, hi: float) -> float:
    """Pull a coordinate back onto the pitch. A handful of freeze-frame points
    sit a hair outside [0, hi]; clamp them so the strict schema accepts them."""
    return float(min(max(v, 0.0), hi))


def _freeze_frame_to_players(freeze_frame: object) -> list[Player]:
    if not isinstance(freeze_frame, list):
        return []
    players: list[Player] = []
    for entry in freeze_frame:
        loc = entry.get("location")
        if not loc or len(loc) < 2:
            continue
        position = (entry.get("position") or {}).get("name", "")
        players.append(
            Player(
                xy=(_clamp(loc[0], PITCH_LENGTH), _clamp(loc[1], PITCH_WIDTH)),
                team="att" if entry.get("teammate") else "def",
                is_gk=(position == "Goalkeeper"),
            )
        )
    return players


def shot_to_record(shot: pd.Series, competition: str) -> dict | None:
    """Convert one StatsBomb shot row into a flat, parquet-friendly record.
    Returns None if the shot lacks a usable location."""
    loc = shot.get("location")
    if not isinstance(loc, list) or len(loc) < 2:
        return None

    state = GameState(
        shot_xy=(_clamp(loc[0], PITCH_LENGTH), _clamp(loc[1], PITCH_WIDTH)),
        players=_freeze_frame_to_players(shot.get("shot_freeze_frame")),
    )

    return {
        "competition": competition,
        "match_id": shot.get("match_id"),
        "minute": shot.get("minute"),
        "team": shot.get("team"),
        "player": shot.get("player"),
        "shot_type": shot.get("shot_type"),          # Open Play / Penalty / Free Kick
        "body_part": shot.get("shot_body_part"),
        "outcome": shot.get("shot_outcome"),
        "is_goal": int(shot.get("shot_outcome") == "Goal"),
        "statsbomb_xg": shot.get("shot_statsbomb_xg"),  # benchmark to beat/match
        "n_players_visible": len(state.players),
        "game_state": state.model_dump_json(),          # the locked input, serialized
    }


def load_shots(competitions=COMPETITIONS) -> pd.DataFrame:
    from statsbombpy import sb

    records: list[dict] = []
    for comp_id, season_id, name in competitions:
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        print(f"{name}: {len(matches)} matches")
        for i, match_id in enumerate(matches["match_id"], 1):
            events = sb.events(int(match_id))
            events["match_id"] = int(match_id)
            shots = events[events["type"] == "Shot"]
            for _, shot in shots.iterrows():
                rec = shot_to_record(shot, name)
                if rec is not None:
                    records.append(rec)
            print(f"  [{i}/{len(matches)}] match {match_id}: "
                  f"{len(shots)} shots (total {len(records)})", end="\r")
        print()
    return pd.DataFrame(records)


def main() -> None:
    df = load_shots()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    goals = df["is_goal"].sum()
    print(f"\nSaved {len(df)} shots -> {OUT_PATH}")
    print(f"Goals: {goals} ({goals / len(df):.1%} conversion rate)")
    print(f"Shot types:\n{df['shot_type'].value_counts()}")
    print(f"Median players visible per freeze frame: {df['n_players_visible'].median():.0f}")


if __name__ == "__main__":
    main()
