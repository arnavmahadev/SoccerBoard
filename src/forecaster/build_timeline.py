"""Build the committed title-odds timeline (artifacts/title_odds_timeline.json).

The timeline is how the champion odds moved as the tournament actually unfolded:
a snapshot of every team's title probability after each matchday and each knockout
round, computed by locking in the games decided so far and simulating the rest on
the frozen pre-tournament ratings (bent by in-tournament form). Now the tournament
is over this is static, so — like the group forecast — it's committed rather than
recomputed per request, and the deployed app serves it instantly and offline.

Reads the frozen fit (params.json), the competition config and the committed
results snapshot; writes the timeline. Regenerate whenever a result override or
the snapshot changes.

Run:  python -m forecaster.build_timeline
"""

from __future__ import annotations

import json

from forecaster import predictor
from forecaster.data import ARTIFACTS, SNAPSHOT_PATH, get_competition, load_matches
from forecaster.dixon_coles import Params

N_SIMS = 20000  # committed, so simulate generously for stable odds


def build(competition: str = "world_cup_2026", n: int = N_SIMS) -> dict:
    cfg = get_competition(competition)
    params = Params.from_json(ARTIFACTS / "params.json")
    matches = [
        m for m in load_matches(
            path=SNAPSHOT_PATH, since=f"{cfg['season_year']}-01-01", played_only=False
        )
        if m.tournament == cfg["tournament_label"]
    ]
    return predictor.compute_timeline(cfg, params, matches, n=n)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = build()
    out = ARTIFACTS / "title_odds_timeline.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    print(f"  champion: {result['champion']}, {len(result['checkpoints'])} checkpoints "
          f"({result['n']} sims each)")
    for c in result["checkpoints"]:
        lead = c["teams"][0]
        print(f"  {c['label']:<22} lead {lead['team']} {lead['champion']*100:4.1f}%  "
              f"({c['alive']} alive)")


if __name__ == "__main__":
    main()
