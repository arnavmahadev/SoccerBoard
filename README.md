# Soccer xG

An expected-goals (xG) model trained on **StatsBomb 360 freeze frames** — the
positions of every visible player at the moment of a shot — and served as an
interactive web app where you drag players around a 2D pitch and watch the xG
prediction update live.

![demo](reports/demo.png)

## Architecture principle

The model's input is **player (x, y) coordinates in pitch space + the shot
location** — the same format soccer tracking data provides. This is locked in
[`src/xg/data/schema.py`](src/xg/data/schema.py) and used everywhere (features,
model, API, frontend). A future video → 2D-tracking pipeline can feed the same
model with zero rework.

### Coordinate system (StatsBomb convention)

- Pitch is **120 (length) × 80 (width)**.
- The attacking side shoots toward the goal at **x = 120**.
- Goal mouth spans **y = 36 .. 44**, centered at **y = 40**.

## Status

- [x] **Phase 0** — setup + locked input schema + sanity-test harness
- [x] **Phase 1** — data acquisition & exploration (2,783 shots)
- [x] **Phase 2** — feature engineering (9 position-only features)
- [x] **Phase 3** — baseline model: XGBoost, test log loss 0.263 / Brier 0.071
- [x] **Phase 4** — MLP (0.274); XGBoost wins on small tabular data → served
- [x] **Phase 5** — evaluation: ECE 0.027, well calibrated ([report](reports/evaluation.md))
- [x] **Phase 6** — FastAPI `/predict` + `/health`, schema-validated requests
- [x] **Phase 7** — draggable SVG pitch, live xG (served from the API)
- [ ] Phase 8 — deployment (Hugging Face Spaces)
- [ ] Stretch (post-ship only) — DeepSets net over raw player sets

## Develop

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # schema + sanity + metrics + API checks

python -m xg.data.load                  # build data/processed/shots.parquet
python -m xg.models.baseline            # train + save the served model
uvicorn xg.serve.app:app --reload       # serve the API at http://127.0.0.1:8000/docs
```
