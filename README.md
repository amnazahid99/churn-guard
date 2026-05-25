# ChurnGuard AI

ChurnGuard AI is a churn prediction platform with three parts:
- a Python training pipeline that builds the churn model and SHAP explainer
- a FastAPI backend that serves predictions and explanations
- a Next.js + Tailwind dashboard that visualizes risk and batch scoring

## Project Layout
```text
churn-guard/
├── pyproject.toml
├── uv.lock
├── README.md
└── churnguard-ai/
    ├── backend/
    │   └── main.py
    ├── data/
    │   ├── telco_churn.csv
    │   ├── telco_churn_train.csv
    │   └── telco_churn_test.csv
    ├── frontend/
    │   ├── package.json
    │   └── src/app/
    │       ├── layout.tsx
    │       ├── globals.css
    │       └── page.tsx
    ├── model/
    │   ├── churn_model.joblib
    │   ├── scaler.joblib
    │   ├── feature_names.joblib
    │   └── shap_explainer.joblib
    ├── train.py
    ├── pyproject.toml
    └── uv.lock
```

## What It Does
- Trains a 4-class churn severity model from the Telco Customer Churn schema.
- Scores customers with `No Risk`, `Low Risk`, `Medium Risk`, and `High Risk / Churn` labels.
- Serves single-record, batch, and explanation endpoints from FastAPI.
- Renders a dashboard with customer inputs, SHAP drivers, history, and batch leaderboards.

## Data
The project uses the Kaggle Telco Customer Churn dataset:
- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

If you do not have the original CSV, the repository includes generated sample files in `churnguard-ai/data/` with the same schema:
- `telco_churn.csv`
- `telco_churn_train.csv`
- `telco_churn_test.csv`

## Setup
Install `uv` if needed:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Create the Python environment and install dependencies from the root project:
```powershell
uv venv
uv sync
```

For the frontend:
```powershell
cd churnguard-ai/frontend
npm install
```

## Train the Model
Run training from the `churnguard-ai` folder:
```powershell
cd churnguard-ai
uv run python train.py
```

This generates the model artifacts in `churnguard-ai/model/`.

## Run the Backend
```powershell
cd churnguard-ai
uv run uvicorn backend.main:app --reload --port 8000
```

API docs:
- http://localhost:8000/docs

## Run the Frontend
```powershell
cd churnguard-ai/frontend
npm run dev
```

Dashboard:
- http://localhost:3000

## Backend Endpoints
- `GET /health` - service and model status
- `GET /model/info` - model metadata
- `GET /model/feature-importance` - top features
- `POST /predict` - single customer prediction
- `POST /predict/explain` - single prediction plus SHAP waterfall
- `POST /predict/batch` - batch CSV scoring
- `GET /cohort-analysis` - last batch cohort summary

## Notes
- The frontend uses Next.js 14 App Router, TypeScript, Recharts, Framer Motion, and Tailwind CSS.
- The backend reloads trained artifacts on demand, so you can retrain and keep the same server running.
- The generated sample CSVs are useful for local development, but the Kaggle dataset is the best source for real training.