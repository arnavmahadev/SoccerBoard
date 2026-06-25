---
title: Soccer xG
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<!-- The YAML block above is Hugging Face Spaces config; it must be the very
     first thing in this file. Harmless on GitHub. -->

# Soccer xG

**▶ Live demo: https://arnavmahadev-soccer-xg.hf.space**

An expected-goals (xG) model trained on **StatsBomb 360 freeze frames** — the
positions of every visible player at the moment of a shot — and served as an
interactive web app where you drag players around a 2D pitch and watch the xG
prediction update live.

![demo](reports/demo.png)

## Highlights

- **End-to-end ML system**: data ingestion → feature engineering → model
  comparison → calibration analysis → REST API → interactive UI → deployed
  container, all in one repo.
- **Positions-only model** (XGBoost) reaches **0.263 test log loss**, within
  ~0.02 of StatsBomb's professional xG (0.244) on the same shots — using only
  player coordinates, no proprietary event features.
- **Well calibrated** (ECE 0.027): a predicted 0.3 really converts ~30% of the
  time, which is what xG is *for*.
- **One locked input schema** flows through every layer, so a future
  video→2D-tracking pipeline plugs in with zero rework.

## How it works

```mermaid
flowchart LR
    A["StatsBomb 360<br/>freeze frames"] --> B(("GameState<br/>locked schema"))
    B --> C["9 position-only<br/>features"]
    C --> D["XGBoost<br/>(served model)"]
    D --> E["FastAPI<br/>/predict"]
    B -. "same schema is the<br/>API request body" .-> E
    E --> F["SVG draggable pitch<br/>live xG"]
    F -. "builds a GameState" .-> E
```

The hub is [`GameState`](src/xg/data/schema.py): `shot_xy` plus a list of
players (each `xy`, `team`, `is_gk`). It is the model input, the API request
body, and what the frontend constructs from marker positions — one definition,
validated at runtime, used everywhere. The SVG `viewBox` is set to pitch units,
so a dragged marker's position *is* a model coordinate (no scaling math).

### Coordinate system (StatsBomb convention)

- Pitch is **120 (length) × 80 (width)**; attack shoots toward **x = 120**.
- Goal mouth spans **y = 36 .. 44**, centered at **y = 40**.

## Models & results

Held-out test set: 540 open-play shots, **split by match** (no leakage).
StatsBomb's own xG is the professional benchmark on the identical shots.

| model | log loss | Brier | ECE | notes |
|---|---|---|---|---|
| Logistic regression | 0.270 | 0.076 | — | interpretable floor |
| **XGBoost** (served) | **0.263** | **0.071** | **0.027** | best; trees win on small tabular data |
| MLP (PyTorch) | 0.274 | 0.076 | — | behind XGBoost, as expected |
| DeepSets (PyTorch) | 0.287 | 0.080 | — | over raw player sets; robust (below) |
| StatsBomb (benchmark) | 0.244 | 0.068 | 0.019 | uses features we don't have |

![calibration](reports/calibration.png)

**A finding worth the interview:** a contrived wide-open chance (no nearby
defenders) breaks the plain MLP — it predicts **0.00**, because the "nearest
defender distance" feature hits an out-of-distribution sentinel that saturates
the network. XGBoost handles it (**0.66**) by extrapolating flat, and the
**DeepSets** model handles it too (**0.48**) — by consuming the raw player set,
"no defender" is just a smaller set with nothing to saturate on. Tree robustness
and set-based design, demonstrated on one concrete case.

Penalties are special-cased to the canonical **0.76** at serve time (an
out-of-band `shot_type` hint), keeping the `GameState` contract positions-only.
Details in [reports/evaluation.md](reports/evaluation.md).

## What I'd do next

- **More data**: StatsBomb 360 covers more competitions now; more shots would
  likely let the neural models close the gap with (or beat) XGBoost.
- **Richer geometry**: passing/assist context, shot trajectory, defender
  velocity — all derivable from full tracking data, none from the freeze frame.
- **Validate on Metrica continuous tracking** to prove the same model eats real
  25 fps tracking, not just freeze frames.
- **Per-shot explanations** (SHAP) surfaced in the UI: *why* this xG?

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # schema + sanity + metrics + API checks

python -m xg.data.load                  # build data/processed/shots.parquet
python -m xg.models.baseline            # train + export the served model
uvicorn xg.serve.app:app --reload       # http://127.0.0.1:8000  (docs at /docs)
```

Other entry points: `python -m xg.features.build` (feature/split summary),
`python -m xg.models.nn` / `python -m xg.models.deepsets` (neural models),
`python -m xg.eval.calibrate` (calibration report). Deployment:
[DEPLOY.md](DEPLOY.md).

## Project layout

```
src/xg/
  data/schema.py     # the locked GameState contract (pydantic)
  data/load.py       # StatsBomb shots + freeze frames -> GameState rows
  features/build.py  # GameState -> 9-feature vector (single train/serve source)
  models/baseline.py # logreg + XGBoost; exports the served booster
  models/nn.py       # PyTorch MLP
  models/deepsets.py # permutation-invariant net over raw player sets
  models/predictor.py# lightweight serving (xgboost + numpy only)
  eval/metrics.py    # log loss, Brier, calibration / ECE
  serve/app.py       # FastAPI: /predict, /health + static frontend
frontend/            # SVG draggable pitch (vanilla JS)
notebooks/           # EDA only
tests/               # schema, features, metrics, scenarios, API
```

## Status

Phases 0–8 complete (data → features → baseline → NN → eval → API → frontend →
deploy), plus the DeepSets stretch. Built as an incremental, one-phase-at-a-time
project; each phase is a separate commit.
