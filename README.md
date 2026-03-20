# AI Health Tracker for Rural Areas

AI-powered health symptom checker built for rural and low-resource settings.
It combines a modern React frontend with a Python machine learning backend to provide fast, practical guidance from symptom inputs.

## What This Project Does

- Collects symptom descriptions from users
- Converts symptoms into model-ready features
- Predicts likely conditions using a trained ML model
- Shows an easy-to-read report with:
   - detected symptoms
   - possible conditions
   - recommended precautions
   - when to see a doctor

## Core Highlights

- One-command local startup (frontend + backend together)
- One-link app usage in browser (frontend proxies backend under /api)
- Unified Python backend entrypoint for training, API, and Streamlit mode
- Reproducible training workflow with model comparison
- Built for practical UX and deployability

## Tech Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS, Motion
- Backend/API: Python, FastAPI, Uvicorn
- ML: scikit-learn, pandas, joblib, optional XGBoost
- Optional UI mode: Streamlit

## Project Structure

```text
src/                       React frontend
main.py                    Unified Python entrypoint
requirements.txt           Python dependencies
ml/artifacts/              Trained model and metrics output
scripts/dev-all.js         Starts frontend + backend together
package.json               Node scripts and frontend deps
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+

### 1) Install frontend dependencies

```bash
npm install
```

### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the full app

```bash
npm run dev
```

This command starts both services:

- Frontend: http://localhost:3000 (or next free port)
- Backend API: http://localhost:8000

Open the frontend URL only. Backend calls are routed through /api, so it behaves like one integrated app link.

## Run Modes

### Full stack

```bash
npm run dev
```

### Frontend only

```bash
npm run dev:web
```

### Backend only

```bash
npm run dev:api
```

## ML Training

All backend modes are provided by main.py.

### Train on built-in sample dataset

```bash
python main.py train
```

### Train on your CSV

```bash
python main.py train --dataset path/to/your_dataset.csv --target label_column
```

### Compare models and keep best

```bash
python main.py train --dataset data/symptom-based-disease-prediction-dataset/disease_prediction.csv --target label --model compare
```

### Useful training options

- --model random_forest|log_reg|xgboost|compare
- --test-size 0.2
- --random-state 42
- --output-dir ml/artifacts

Optional local dependency for advanced model experiments:

```bash
pip install xgboost
```

### Training output files

- ml/artifacts/best_model.joblib
- ml/artifacts/metrics.json
- ml/artifacts/label_classes.json (when labels are string-based)

## Backend API

### Main endpoints

- GET /health
- POST /predict

### Through frontend proxy

- GET /api/health
- POST /api/predict

## Streamlit Mode (Optional)

You can run a Streamlit-based prediction UI from the same unified backend file:

```bash
python main.py streamlit
```

If needed, set a specific Streamlit port:

```bash
python -m streamlit run main.py --server.port 8765
```

For hosted deployment environments, use the repository root entrypoint:

```bash
streamlit run main.py
```

This root file is deployment-compatible and also exposes `main:api` for platforms that auto-detect ASGI apps.

Recommended Streamlit Cloud settings:

- Main file path: `main.py`
- Python version: from `runtime.txt` (3.11)
- Requirements file: root `requirements.txt`

## Configuration

Optional backend target override for Vite proxy:

```bash
VITE_BACKEND_TARGET=http://localhost:8000
```

## Troubleshooting

- Frontend starts on a different port:
   - if 3000 is busy, Vite auto-selects 3001, 3002, etc.
- Backend port conflict:
   - change backend port in run command or stop old process
- TypeScript command not found:
   - run npm install first
- Backend health test:

```bash
curl http://localhost:8000/health
```

- Streamlit deploy error: `Attribute "api" not found in module "main"`
   - Ensure deployment command is `streamlit run main.py`

## Disclaimer

This application provides informational guidance only and is not a substitute for licensed medical diagnosis, treatment, or emergency care.
