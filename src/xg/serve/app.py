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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

from xg.data.schema import GameState
from xg.models.predictor import load, model_info, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()  # warm the cache / fail fast if the model isn't exported
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


# Serve the interactive frontend at "/" (mounted last so the API routes above
# and the auto docs take precedence). html=True serves index.html at the root.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
