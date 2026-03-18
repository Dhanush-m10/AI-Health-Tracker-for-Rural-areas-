<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/69915254-4542-4dae-9d28-59be466d3585

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

`npm run dev` now starts both:

- Frontend: `http://localhost:3000`
- ML backend API: `http://localhost:8000`

The frontend uses a proxy (`/api`) to call the backend, so you only need to open the frontend URL.

Optional individual commands:

- `npm run dev:web` (frontend only)
- `npm run dev:api` (backend only)

Optional backend target override:

- `VITE_BACKEND_TARGET=http://localhost:8000`

## Python ML Training

This project now includes a unified Python entrypoint in `ml/main.py`.

### 1) Install ML dependencies

```bash
pip install -r ml/requirements.txt
```

### 2) Train with built-in dataset (quick start)

```bash
python ml/main.py train
```

This uses the `sklearn` breast cancer dataset to verify the training pipeline and produce metrics.

### 3) Train with your own dataset

Use a CSV where one column is the prediction target (label).

```bash
python ml/main.py train --dataset path/to/your_dataset.csv --target target_column_name
```

Optional arguments:

- `--model random_forest` (default) or `--model log_reg`
- `--model xgboost` (if installed)
- `--model compare` to auto-try all available models and keep the best
- `--test-size 0.2`
- `--random-state 42`
- `--output-dir ml/artifacts`

### Outputs

After training, artifacts are saved in `ml/artifacts`:

- `best_model.joblib`: trained model
- `metrics.json`: accuracy, F1, best params, and classification details
- `label_classes.json`: class names (only for string labels)

## Streamlit Inference App

After training, run the local prediction UI with Streamlit:

```bash
streamlit run ml/main.py
```

or

```bash
python ml/main.py streamlit
```

This app loads:

- `ml/artifacts/best_model.joblib`
- `ml/artifacts/metrics.json`

Then lets you enter symptom values and get predicted disease plus confidence scores.

## Integrate Model With Existing React UI

The existing symptom checker UI in `src/` is now wired to the trained ML model through a local Python API.

### 1) Start ML inference API

```bash
python ml/main.py api --host 0.0.0.0 --port 8000
```

or

```bash
python -m uvicorn ml.main:app --host 0.0.0.0 --port 8000
```

### 2) Start frontend

```bash
npm run dev
```

By default, frontend calls `http://localhost:8000`.

Optional: set a custom endpoint in `.env.local`:

```bash
VITE_ML_API_URL=http://localhost:8000
```
