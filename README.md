# PathSnap: Map-Matching & Route Reconstruction

PathSnap matches noisy GPS tracks to a road network and rebuilds the route the vehicle actually drove. For every GPS point we take the nearby road segments as candidates, a LightGBM classifier scores how likely each one is the true road, and HMM/Viterbi decoding turns those scores into one connected route. A FastAPI backend (Laptop 1) serves predictions to a React UI (Laptop 2).

## Results (10 student-test trips)

| | Top-1 accuracy (per GPS point) | Route F1 |
|---|---|---|
| Nearest road (no ML) | 0.56 | – |
| Tuned LightGBM only | 0.882 | 0.778 |
| **Tuned LightGBM + HMM/Viterbi (final)** | **0.935** | **0.983** |

Model selection (5-fold CV grouped by trip, macro F1): LightGBM 0.914, Random Forest 0.914, MLP 0.910, KNN 0.901, Logistic Regression 0.890. LightGBM was selected for the best log loss (0.078), speed and size, then tuned with Optuna (CV macro F1 0.9161, half the trees, ~40% faster inference).

## Pipeline

```
.track + .nodes + .arcs
   │
   ├─ project lon/lat to local metres
   ├─ candidates: 10 nearest arcs within 200 m of each GPS point (KD-tree)
   ├─ 14 features per (point, arc): distance, heading difference, speed, rank, connectivity, ...
   ├─ tuned LightGBM → P(arc is the true road)
   ├─ HMM/Viterbi: emissions = log P, transitions = |road distance − GPS distance| / β
   └─ shortest-path gap filling → one continuous route
```

Training labels: the reference `.route` lists arcs in driving order but not which GPS point is on which arc, so each point is aligned to the route with a forward-only dynamic programme (`pathsnap/labels.py`). The route is used only for labels and scoring, never as a prediction input.

## Setup

```bash
git clone https://github.com/golden-Puffle/pathsnap.git
cd pathsnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data (not committed)

Unzip the dataset so this layout exists inside the repo (`data` can also be a symlink):

```
data/
├── train/          # 80 trips
└── student_test/   # 10 trips
```

Each trip folder `<id>/` holds `<id>.track` (lon, lat, t), `<id>.nodes` (lon, lat), `<id>.arcs` (from_node, to_node) and `<id>.route` (reference arc ids).

The evaluator trips are kept **outside** the training data and are only served by the API during the demo (see below).

## Reproducing the experiments

Run from the repo root with the virtualenv active.

| Step | Command | Output |
|---|---|---|
| 1. EDA | `python training/run_eda.py --data data --out results/eda` | `results/eda/` figures, tables, Folium maps |
| 2. Feature table | `python -m training.build_dataset` | `artifacts/candidates.parquet`, `results/candidate_stats.csv` |
| 3. Train & compare 5 models | `python -m training.train_models` | `results/model_comparison.csv`, `results/cv_folds.csv`, `results/mlp_training_history.png` |
| 4. Report graphs | `python -m training.plot_model_graphs` | `results/graphs/` |
| 5. Bayesian tuning (Optuna) | `python -m training.tune_optuna --trials 15` | `artifacts/final_model.joblib`, `results/tuning/` |
| 6. HMM parameter search (train trips) | `python -m training.evaluate_hmm --split train --grid --max-trips 12` | `results/hmm/hmm_grid_train.csv` |
| 7. Classifier-only vs HMM | `python -m training.evaluate_hmm --beta 2 --emission-weight 1.0` | `results/hmm/hmm_summary_student_test.csv` |
| 8. Viterbi demo + map | `python -m training.demo_viterbi --trip data/student_test/00000091 --map results/maps/viterbi_00000091.html` | dot-by-dot changes, Folium map |

Notes:
- Trips `00000033` and `00000054` are excluded from training: their GPS is offset 120–140 m from the reference route for the whole trip (see `results/candidate_stats.csv`).
- All models use the same seed (42), the same trip-grouped folds and the same 400k-row training sample (KNN: 100k, because its prediction cost grows with training size).
- Step 2 takes ~80 s, step 3 ~15 min (KNN is the slowest), step 5 ~7 min.

## Deployment (two laptops)

### Laptop 1: API (FastAPI)

```bash
source .venv/bin/activate
sudo firewall-cmd --add-port=8000/tcp          # Fedora; open port 8000 once
ip -4 addr | grep inet                         # note this laptop's IP

# student-test trips (default)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# evaluator trips (demo)
PATHSNAP_TRIPS_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API loads `artifacts/final_model.joblib` and `artifacts/hmm_config.json` once at startup and logs every request (trip, points, timings, metrics) in the terminal.

| Endpoint | Description |
|---|---|
| `GET /health` | Status, loaded model, HMM settings, trips folder |
| `GET /trips` | Available trip ids |
| `POST /predict` | Body `{"trip_id": "00000052", "use_hmm": true}` → GPS points, predicted route, metrics, timings |
| `GET /docs` | Interactive API page (`/` redirects here) |

Environment variables: `PATHSNAP_TRIPS_DIR` (trip folders), `PATHSNAP_MODEL` (model bundle), `PATHSNAP_HMM_CONFIG` (HMM settings).

**Ground truth stays server-side:** prediction loads the trip with `with_route=False`; the reference route is read only after predicting, to compute metrics, and its geometry is never returned to the UI.

### Laptop 2: UI (React + Vite)

Requires Node.js and npm.

```bash
cd ui
npm install
npm run dev          # open the printed URL, usually http://localhost:5173
```

Set the API address in `ui/src/App.jsx` to Laptop 1's IP before starting:

```js
const API = "http://<laptop-1-ip>:8000";
```

The header shows **API Connected** when the UI can reach the backend. Both laptops must be on the same network (a phone hotspot works if the venue Wi-Fi isolates devices).

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/io.py` | Load trip folders |
| `pathsnap/geo.py` | lon/lat → metres, point-to-segment distance, bearings |
| `pathsnap/labels.py` | Forward-only alignment of GPS points to the reference route (training labels) |
| `pathsnap/candidates.py` | Candidate generation and the 14 features |
| `pathsnap/hmm.py` | Road graph, bounded Dijkstra, HMM transitions, Viterbi, route reconstruction |
| `pathsnap/predict.py` | Full prediction pipeline for one trip (used by the API) |
| `pathsnap/metrics.py` | Macro F1, log loss, Top-1, continuity, route precision/recall/F1 |
| `pathsnap/eda.py`, `pathsnap/maps.py` | EDA statistics and Folium maps |
| `training/` | Scripts for every step above |
| `artifacts/` | `final_model.joblib`, `lgbm_best_params.json`, `hmm_config.json` |
| `results/` | EDA, model comparison, tuning, HMM results, graphs, maps |
| `api/main.py` | FastAPI backend |
| `ui/` | React evaluator interface |
