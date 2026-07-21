"""Build every committed forecaster artifact, so the deployed app starts fast and
offline (mirroring the committed xG model).

Produces under forecaster/artifacts/:
  - world_cup_2026.json   competition config (groups, fixtures, knockout bracket)
  - params.json           frozen pre-tournament Dixon-Coles fit
  - group_forecast.json   pre-tournament P(win group)/P(advance) per team
  - title_odds_timeline.json  champion odds by tournament stage (slider replay)
  - metrics.json          backtest log-loss / Brier / calibration / baseline
  - results_snapshot.csv  trimmed offline fallback for the match-results feed

The title-odds timeline is committed because the tournament is over, so it's
static (like the group forecast). During the tournament the live knockout
simulation was recomputed from the latest results on each request instead.

Run:  python -m forecaster.build_artifacts
"""

from __future__ import annotations

import csv
import json

from forecaster import build_timeline, build_wc2026, dixon_coles as dc, evaluate
from forecaster.data import (
    ARTIFACTS,
    SNAPSHOT_SINCE,
    WC2026_KICKOFF,
    fetch_results,
    get_competition,
    load_matches,
)
from forecaster.formats.base import MatchSampler
from forecaster.formats.world_cup import WorldCupFormat
from forecaster.player_model import fetch_lineup_matches, fit_player_deltas

FIT_SINCE = "2010-01-01"   # history window for the production fit (time-decayed)
# xi (time-decay) and reg (ridge) chosen by an out-of-sample grid search on the
# backtest split (min log-loss): xi=0.25 (longer memory than 0.35) generalises
# slightly better and leans less on a handful of recent results per team.
XI = 0.25
REG = 0.02
N_SIMS = 20000             # group forecast is committed, so simulate generously


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("Fetching latest results feed...")
    src = fetch_results(force=True)

    print("Building competition config (world_cup_2026)...")
    cfg = build_wc2026.build_config()
    (ARTIFACTS / "world_cup_2026.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False)
    )

    print(f"Fitting Dixon-Coles (pre-tournament, since {FIT_SINCE})...")
    train = load_matches(as_of=WC2026_KICKOFF, since=FIT_SINCE)
    train = [m for m in train if m.date < WC2026_KICKOFF]
    params = dc.fit(train, xi=XI, reg=REG, ref_date=WC2026_KICKOFF)
    params.to_json(ARTIFACTS / "params.json")
    print(f"  {len(params.teams)} teams, {params.n_matches} matches, "
          f"home_adv={params.home_adv:.3f}, rho={params.rho:.3f}")

    print(f"Group-stage forecast ({N_SIMS} sims)...")
    sampler = MatchSampler(params)
    fmt = WorldCupFormat(cfg)
    forecast = fmt.group_forecast(sampler, n=N_SIMS, seed=7)
    (ARTIFACTS / "group_forecast.json").write_text(
        json.dumps({cfg["id"]: forecast}, indent=2, ensure_ascii=False)
    )

    print("Fetching StatsBomb lineup data + fitting player deltas...")
    lineup_matches = fetch_lineup_matches(force=False)
    print(f"  {len(lineup_matches)} lineup matches across international tournaments")
    player_deltas = fit_player_deltas(params, lineup_matches)
    player_deltas.to_json(ARTIFACTS / "player_deltas.json")
    print(f"  {len(player_deltas.players)} players fitted "
          f"({player_deltas.n_lineup_matches} valid matches)")

    print("Backtest + calibration...")
    report = evaluate.backtest(xi=XI, reg=REG)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2))
    evaluate._print_summary(report)

    print(f"\nWriting trimmed results snapshot (since {SNAPSHOT_SINCE})...")
    with open(src, newline="", encoding="utf-8") as f, \
            open(ARTIFACTS / "results_snapshot.csv", "w", newline="", encoding="utf-8") as out:
        reader = csv.reader(f)
        writer = csv.writer(out)
        header = next(reader)
        writer.writerow(header)
        kept = 0
        for row in reader:
            if row and row[0] >= SNAPSHOT_SINCE:
                writer.writerow(row)
                kept += 1
    print(f"  snapshot rows: {kept}")

    print("\nBuilding title-odds timeline (champion odds by stage)...")
    timeline = build_timeline.build(cfg["id"])
    (ARTIFACTS / "title_odds_timeline.json").write_text(
        json.dumps(timeline, indent=2, ensure_ascii=False)
    )
    print(f"  champion: {timeline['champion']}, {len(timeline['checkpoints'])} checkpoints")

    print("\nDone. Artifacts written to", ARTIFACTS)


if __name__ == "__main__":
    main()
