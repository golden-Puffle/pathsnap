# PathSnap: Map-Matching & Route Reconstruction

PathSnap matches noisy GPS tracks to a road network and rebuilds the route the vehicle actually drove. A classifier scores the candidate road segments near each GPS point, then HMM/Viterbi decoding turns those scores into one connected route. A FastAPI backend serves the result to a web UI.

## Setup

```bash
git clone https://github.com/golden-Puffle/pathsnap.git
cd pathsnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

The raw dataset is not committed. Unzip `Student folder.zip` so this layout exists:

```
data/
├── train/          # 80 trips
└── student_test/   # 10 trips
```

(`data` can also be a symlink to the unzipped `Student folder`.)

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Shared library: data loading, geometry, labels, candidates, HMM, metrics, prediction |
| `notebooks/` | EDA, model experiments, HMM analysis |
| `training/` | Dataset building, model training, Bayesian optimisation scripts |
| `artifacts/` | Final model, preprocessing objects, HMM configuration |
| `results/` | Experiment logs, CSVs, plots, Folium maps |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator interface (Laptop 2) |
| `report/` | Technical report and figures |

## Running

_To be completed: training, API and UI commands._
