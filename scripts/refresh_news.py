#!/usr/bin/env python3
"""Regenerate src/forecaster/news_items.json from a free public injury tracker.

Picks the current top-N teams by title odds (from the forecaster's own live
simulation), scrapes their injuries from Soccer26Live's WC-2026 injury table,
and rewrites the news overlay the serving layer reads. No LLM in the loop, no API
key — run it on a cron or before a deploy.

    python scripts/refresh_news.py                 # refresh top-10, write the file
    python scripts/refresh_news.py --top 12 --dry-run   # preview, write nothing
    python scripts/refresh_news.py --self-test          # offline transform check

Exits non-zero on any fetch/parse error so a scheduler can alert on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from forecaster import news_fetch as nf  # noqa: E402


# Captured-shape Soccer26Live injury records for the offline self-test: exercises
# team filtering, name mapping (Türkiye -> Turkey), the recency cut, dedup, and
# issue text with no network. Same {player, team, injury, status,
# expected_return, updated} shape parse_injuries emits. Dates are relative to the
# fixed _AS_OF below so the test is deterministic.
_AS_OF = "2026-07-06"
_SELF_TEST_RECORDS = [
    {"player": "Leonardo Balerdi", "team": "Argentina", "injury": "Right calf injury",
     "status": "injured", "expected_return": "Ruled out for WC2026", "updated": "2026-07-05"},
    {"player": "Leonardo Balerdi", "team": "Argentina", "injury": "Right calf injury",
     "status": "injured", "expected_return": "Ruled out for WC2026", "updated": "2026-07-05"},  # dup -> one
    {"player": "Kenan Yildiz", "team": "Türkiye", "injury": "Fitness Check",
     "status": "doubtful", "expected_return": "TBC", "updated": "2026-07-04"},  # recent doubt -> kept
    {"player": "Nayef Aguerd", "team": "Morocco", "injury": "Injury (specifics not detailed)",
     "status": "injured", "expected_return": "Ruled out for WC2026", "updated": "2026-07-01"},  # generic reason
    {"player": "Lionel Messi", "team": "Argentina", "injury": "Fitness Check",
     "status": "doubtful", "expected_return": "TBC", "updated": "2026-05-26"},  # stale -> dropped
    {"player": "Ghost Player", "team": "Narnia", "injury": "Knee",
     "status": "injured", "expected_return": "Unknown", "updated": "2026-07-06"},  # team not in top-N -> dropped
]


def _self_test() -> int:
    wanted = ["Argentina", "Turkey", "Morocco"]
    teams_news = nf.build_teams_news(_SELF_TEST_RECORDS, wanted, as_of=_AS_OF, max_age_days=21)

    assert "Narnia" not in teams_news, "team outside top-N leaked through"
    assert set(teams_news) == {"Argentina", "Turkey", "Morocco"}, teams_news
    assert len(teams_news["Argentina"]) == 1, "duplicate not deduped / stale Messi not dropped"
    arg = teams_news["Argentina"][0]
    assert arg["player"] == "Leonardo Balerdi", arg
    assert all(i["player"] != "Lionel Messi" for i in teams_news["Argentina"]), "stale row leaked"
    assert arg["issue"] == "Right calf injury; ruled out of the tournament", arg["issue"]
    assert arg["source"] == "Soccer26Live" and arg["url"], arg
    yildiz = teams_news["Turkey"][0]  # Türkiye normalized; recent doubt survives the cut
    assert yildiz["issue"] == "a fitness doubt", yildiz["issue"]
    aguerd = teams_news["Morocco"][0]  # generic reason -> return-based text
    assert aguerd["issue"] == "ruled out of the tournament", aguerd["issue"]

    print("self-test OK — transform, team filter, recency cut, dedup, and issue text all pass.")
    print(json.dumps(teams_news, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--competition", default="world_cup_2026")
    ap.add_argument("--top", type=int, default=10, help="number of title-odds leaders to cover")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--n", type=int, default=20000, help="Monte-Carlo sims for the odds ranking")
    ap.add_argument("--max-age-days", type=int, default=nf.DEFAULT_MAX_AGE_DAYS,
                    help="drop injuries the source hasn't updated within this many days")
    ap.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    ap.add_argument("--self-test", action="store_true", help="offline transform check, no network")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    try:
        summary = nf.refresh(
            args.competition, top=args.top, season=args.season, n=args.n,
            max_age_days=args.max_age_days, dry_run=args.dry_run,
        )
    except nf.ApiError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    teams_news = summary["teams_with_news"]
    print(f"top-{args.top} by title odds: {', '.join(summary['top_teams'])}", file=sys.stderr)
    print(f"fetched {summary['records_fetched']} injury records "
          f"(<= {args.max_age_days}d old); "
          f"kept {summary['injuries_kept']} across {len(teams_news)} team(s)", file=sys.stderr)
    for team, items in teams_news.items():
        print(f"  {team}: {', '.join(i['player'] for i in items)}", file=sys.stderr)
    if args.dry_run:
        print(json.dumps({args.competition: {"teams": teams_news}}, indent=2, ensure_ascii=False))
    else:
        print(f"wrote {nf.NEWS_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
