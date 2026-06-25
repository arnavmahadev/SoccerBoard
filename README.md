# Soccer xG

An expected-goals (xG) model trained on **StatsBomb 360 freeze frames** — the
positions of every visible player at the moment of a shot — and served as an
interactive web app where you drag players around a 2D pitch and watch the xG
prediction update live.

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
- [ ] Phase 1 — data acquisition & exploration
- [ ] Phase 2 — feature engineering
- [ ] Phase 3 — baseline model (logreg / XGBoost)
- [ ] Phase 4 — neural net (MLP)
- [ ] Phase 5 — evaluation (log loss, Brier, calibration)
- [ ] Phase 6 — FastAPI prediction endpoint
- [ ] Phase 7 — interactive draggable-pitch frontend
- [ ] Phase 8 — deployment (Hugging Face Spaces)
- [ ] Stretch (post-ship only) — DeepSets net over raw player sets

## Develop

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest            # runs schema + known-scenario sanity checks
```
