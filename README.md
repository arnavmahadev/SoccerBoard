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

> Predict the probability a shot becomes a goal — from **player positions alone** —
> and explore it by dragging players around a live 2D pitch.

**▶ Live demo: https://arnavmahadev-soccer-xg.hf.space**

![demo](reports/demo.png)

Expected goals (**xG**) is the standard way to measure shot quality in football:
the chance a given shot is scored. This project trains an xG model on
**StatsBomb 360 freeze frames** — the position of every visible player at the
instant of the shot — and serves it as an interactive web app. There are no
proprietary event features: just where the players are.

The whole thing is one repo, end to end:

```
StatsBomb 360  →  GameState  →  9 geometric features  →  model  →  FastAPI  →  draggable pitch UI
  (data)          (schema)       (train = serve)         (XGBoost)   (/predict)    (vanilla JS)
```

## Try it

Open the [live demo](https://arnavmahadev-soccer-xg.hf.space) and:

- **Drag the ball** closer to goal or out to the wing — watch xG rise and fall
  with distance and angle.
- **Move defenders** into the shooting lane, or pull them away, to see how a
  crowded box kills a chance.
- **Drag the keeper** off his line to open up the net.

The number updates on every move because the SVG pitch coordinates *are* the
model's coordinates — the marker positions get sent straight to `/predict`.

## Why it's interesting

- **End-to-end ML system** in one repo: data ingestion → feature engineering →
  model comparison → calibration analysis → REST API → interactive UI →
  deployed container.
- **Positions-only model** (XGBoost) reaches **0.263 test log loss**, within
  ~0.02 of StatsBomb's professional xG (**0.244**) on the same shots — using only
  player coordinates, none of the proprietary event features the benchmark has.
- **Well calibrated** (ECE **0.027**): a predicted 0.3 really does convert ~30%
  of the time, which is the whole point of xG.
- **One locked input schema** flows through every layer, so a future
  video→2D-tracking pipeline could plug in with no rework.

## How it works

![how it works](reports/how-it-works.png)

The hub is [`GameState`](src/xg/data/schema.py): `shot_xy` plus a list of players
(each with `xy`, `team`, `is_gk`). The same object is the model input, the API
request body, and what the frontend builds from marker positions — one
definition, validated at runtime, used everywhere. The SVG `viewBox` is set to
pitch units, so a dragged marker's position *is* a model coordinate (no scaling
math).

### The features

A `GameState` is reduced to **9 geometric features** — the single
train-and-serve source in [`features/build.py`](src/xg/features/build.py):

| feature | meaning |
|---|---|
| `distance` | shot distance to goal centre |
| `angle` | width of goal mouth subtended at the shooter |
| `abs_y_offset` | how far off-centre the shot is |
| `n_defenders` | defenders visible in the freeze frame |
| `defenders_in_cone` | defenders inside the shooter→posts triangle |
| `nearest_def_dist` | distance to the closest outfield defender |
| `gk_visible` | whether a defending keeper is in view |
| `gk_dist_to_goal` | how far the keeper is off his line |
| `gk_dist_to_shot` | keeper's distance from the shooter |

XGBoost is trained with **monotone constraints** on these (e.g. further out →
never higher xG; more open net → never lower), so the model can't learn
physically nonsensical wiggles from a small dataset.

### Coordinate system (StatsBomb convention)

- Pitch is **120 (length) × 80 (width)**; the attack shoots toward **x = 120**.
- Goal mouth spans **y = 36 .. 44**, centred at **y = 40**.

## Models & results

Held-out test set: **540 open-play shots, split by match** (no leakage between
train and test). StatsBomb's own xG is the professional benchmark on the
identical shots.

| model | log loss | Brier | ECE | notes |
|---|---|---|---|---|
| Logistic regression | 0.270 | 0.076 | — | interpretable floor |
| **XGBoost** (served) | **0.263** | **0.071** | **0.027** | best; trees win on small tabular data |
| MLP (PyTorch) | 0.274 | 0.076 | — | behind XGBoost, as expected at this data size |
| DeepSets (PyTorch) | 0.287 | 0.080 | — | over raw player sets; robust (see below) |
| StatsBomb (benchmark) | 0.244 | 0.068 | 0.019 | uses event features this model doesn't have |

![calibration](reports/calibration.png)

**A concrete robustness case.** A contrived wide-open chance (no nearby
defenders) breaks the plain MLP — it predicts **0.00**, because the
`nearest_def_dist` feature hits an out-of-distribution sentinel that saturates
the network. XGBoost handles it (**0.66**) by extrapolating flat, and the
**DeepSets** model handles it too (**0.48**) — by consuming the raw player set,
"no defender" is simply a smaller set with nothing to saturate on. Tree
robustness and set-based design, shown on one reproducible example.

Penalties are special-cased to the canonical **0.76** at serve time (an
out-of-band `shot_type` hint), which keeps the `GameState` contract
positions-only. Full numbers in [reports/evaluation.md](reports/evaluation.md).

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn xg.serve.app:app --reload   # serves the committed model + UI
# → http://127.0.0.1:8000   (API docs at /docs)
```

The trained model (`models/baseline.joblib`) is committed, so this runs without
any data download or training. To run it as the deployed container does, see
[DEPLOY.md](DEPLOY.md).

## Reproduce / retrain

```bash
pytest                              # schema + features + metrics + API checks

python -m xg.data.load             # build data/processed/shots.parquet
python -m xg.models.baseline       # train + export the served XGBoost model
```

Other entry points:

- `python -m xg.features.build` — feature / train-test split summary
- `python -m xg.models.nn` — PyTorch MLP
- `python -m xg.models.deepsets` — permutation-invariant net over raw player sets
- `python -m xg.eval.calibrate` — calibration report

## Project layout

```
src/xg/
  data/schema.py      # the locked GameState contract (pydantic)
  data/load.py        # StatsBomb shots + freeze frames -> GameState rows
  features/build.py   # GameState -> 9-feature vector (single train/serve source)
  models/baseline.py  # logreg + XGBoost; exports the served booster
  models/nn.py        # PyTorch MLP
  models/deepsets.py  # permutation-invariant net over raw player sets
  models/predictor.py # lightweight serving (xgboost + numpy only)
  eval/metrics.py     # log loss, Brier, calibration / ECE
  serve/app.py        # FastAPI: /predict, /health + static frontend
frontend/             # SVG draggable pitch (vanilla JS)
notebooks/            # EDA only
tests/                # schema, features, metrics, scenarios, API
```

## What's next

- **More data**: StatsBomb 360 now covers more competitions; more shots would
  likely let the neural models close the gap with — or beat — XGBoost.
- **Richer geometry**: passing/assist context, shot trajectory, defender
  velocity — all derivable from full tracking, none from a single freeze frame.
- **Validate on Metrica continuous tracking** to confirm the same model handles
  real 25 fps tracking, not just freeze frames.
- **Per-shot explanations** (SHAP) surfaced in the UI: *why* this xG?
