"""FastAPI service for xG predictions.

The request body IS a `GameState` — the same locked schema used in training — so
pydantic validation (on-pitch coordinates, valid teams) comes for free and a
malformed request can never reach the model. The model loads once at startup.

The `shot_type` serving hint rides as a query parameter, keeping the request body
a pure GameState (positions only), faithful to the tracking-data interface.

Run:  uvicorn xg.serve.app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

from xg.data.schema import GameState
from xg.models.predictor import load, model_info, predict
from forecaster import predictor as forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()  # warm the xG model cache / fail fast if it isn't exported
    try:
        forecaster.load()  # warm the forecaster artifacts (params, config, metrics)
        # Refresh the injury overlay in-process on an interval (default daily), so
        # everyone hitting this server gets fresh injuries without a manual
        # scripts/refresh_news.py run or a redeploy. Disable with
        # FORECASTER_NEWS_AUTOREFRESH=0; tune cadence with FORECASTER_NEWS_REFRESH_HOURS.
        if os.environ.get("FORECASTER_NEWS_AUTOREFRESH", "1") != "0":
            # Surface the refresher's logs through uvicorn's handlers (the root
            # logger stays at WARNING under uvicorn, which would swallow them).
            # The "uvicorn" logger owns the default handler; "uvicorn.error" has
            # none of its own (it propagates), so copy from "uvicorn".
            import logging
            news_log = logging.getLogger("forecaster.news")
            uvicorn_handlers = logging.getLogger("uvicorn").handlers
            if uvicorn_handlers:
                news_log.handlers = uvicorn_handlers
                news_log.propagate = False
            news_log.setLevel(logging.INFO)
            hours = float(os.environ.get("FORECASTER_NEWS_REFRESH_HOURS", "24"))
            forecaster.start_news_autorefresh(interval_seconds=hours * 3600.0)
    except FileNotFoundError:
        pass  # forecaster artifacts not built yet; xG mode still serves
    yield


app = FastAPI(title="SoccerBoard", version="0.1.0", lifespan=lifespan)

# Open CORS so the static frontend (Phase 7) can call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    xg: float
    model: str
    shot_type: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **model_info()}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(state: GameState, shot_type: str = "open_play") -> PredictResponse:
    """xG for a shot situation. `shot_type=penalty` returns the canonical value."""
    xg = predict(state, shot_type=shot_type)
    return PredictResponse(xg=xg, model=model_info()["name"], shot_type=shot_type)


# --- Forecaster mode ---------------------------------------------------------
# Competition-parameterized so leagues / Champions League slot in without
# reshaping the API. Only the World Cup is populated for now.
class ForecastMatchRequest(BaseModel):
    competition: str = "world_cup_2026"
    home: str
    away: str
    neutral: bool = True


@app.get("/forecaster/competitions")
def forecaster_competitions() -> list[dict]:
    return forecaster.list_competitions()


@app.get("/forecaster/teams")
def forecaster_teams(competition: str = "world_cup_2026") -> dict:
    try:
        return {"competition": competition, "teams": forecaster.teams(competition)}
    except KeyError:
        raise HTTPException(404, f"Unknown competition: {competition}")


@app.post("/forecaster/match")
def forecaster_match(req: ForecastMatchRequest) -> dict:
    """Goal matrix + W/D/L + expected scoreline for any two teams, on the same
    live ratings (base + news + in-tournament form) as the title odds and
    bracket, so the head-to-head can't contradict them."""
    return forecaster.live_match(
        req.home, req.away, neutral=req.neutral, competition=req.competition
    )


@app.get("/forecaster/adjustments")
def forecaster_adjustments(competition: str = "world_cup_2026") -> dict:
    """The manual news overlay (injury/suspension rating tweaks) active on top of
    the base model, with reasons and sources so the UI can flag adjusted teams."""
    return forecaster.adjustments(competition)


@app.get("/forecaster/simulation")
def forecaster_simulation(
    competition: str = "world_cup_2026",
    as_of: str | None = None,
    n: int = Query(10000, ge=1000, le=50000),
    mode: str = "live",
) -> dict:
    """Per-team stage probabilities.

    mode: 'live' (default), 'pretournament', or 'reality'."""
    try:
        return forecaster.simulation(competition, as_of=as_of, n=n, mode=mode)
    except KeyError:
        raise HTTPException(404, f"Unknown competition: {competition}")


@app.get("/forecaster/bracket")
def forecaster_bracket(
    competition: str = "world_cup_2026", as_of: str | None = None
) -> dict:
    """Predicted knockout bracket — each tie advances the side likelier to win the
    tournament, so the predicted champion equals the title-odds leader; the actual
    view uses real results."""
    try:
        return forecaster.bracket(competition, as_of=as_of)
    except KeyError:
        raise HTTPException(404, f"Unknown competition: {competition}")


@app.get("/forecaster/groups")
def forecaster_groups(
    competition: str = "world_cup_2026", as_of: str | None = None
) -> dict:
    """Per-group actual table alongside the pre-tournament advancement forecast."""
    try:
        return forecaster.group_view(competition, as_of=as_of)
    except KeyError:
        raise HTTPException(404, f"Unknown competition: {competition}")


@app.get("/forecaster/group-matches")
def forecaster_group_matches(
    competition: str = "world_cup_2026", as_of: str | None = None
) -> dict:
    """Per-match group-stage predictions vs. the actual result each game got."""
    try:
        return forecaster.group_matches(competition, as_of=as_of)
    except KeyError:
        raise HTTPException(404, f"Unknown competition: {competition}")


@app.get("/forecaster/metrics")
def forecaster_metrics(competition: str = "world_cup_2026") -> dict:
    """Backtest log-loss / Brier / calibration curve + baseline."""
    return forecaster.metrics(competition)


# Serve the interactive frontend at "/" (mounted last so the API routes above
# and the auto docs take precedence). html=True serves index.html (the
# Forecaster landing page) at the root; the xG board lives at /xg.html.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
