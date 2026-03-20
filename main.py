import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import streamlit as st
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "best_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACTS_DIR / "metrics.json"
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "dist"


def load_dataset(dataset_path: Optional[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if dataset_path:
        csv_path = Path(dataset_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset. "
                f"Available columns: {', '.join(df.columns)}"
            )

        y = df[target_col]
        x = df.drop(columns=[target_col])
        return x, y

    breast = load_breast_cancer(as_frame=True)
    x = breast.data.copy()
    y = breast.target.copy()
    return x, y


def build_pipeline(x: pd.DataFrame, model_name: str, random_state: int) -> tuple[Pipeline, dict]:
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    if model_name == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
        }
    elif model_name == "log_reg":
        model = LogisticRegression(max_iter=3000, random_state=random_state)
        param_grid = {
            "model__C": [0.1, 0.2, 0.5, 1.0, 5.0],
            "model__solver": ["lbfgs"],
        }
    elif model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "model__n_estimators": [300, 600],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.05],
            "model__subsample": [0.9],
            "model__colsample_bytree": [0.9],
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe, param_grid


def train_single_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    random_state: int,
) -> tuple[dict, Pipeline]:
    pipeline, param_grid = build_pipeline(x_train, model_name, random_state)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_weighted = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions, output_dict=True)

    payload = {
        "model_family": model_name,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "test_accuracy": float(accuracy),
        "test_f1_weighted": float(f1_weighted),
        "classification_report": report,
    }
    return payload, best_model


def run_training(
    dataset: Optional[str],
    target: str,
    model_name: str,
    test_size: float,
    random_state: int,
    output_dir: str,
) -> None:
    x, y = load_dataset(dataset, target)

    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    stratify = y if y.nunique() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model_candidates = ["log_reg", "random_forest"]
    if XGBOOST_AVAILABLE:
        model_candidates.append("xgboost")

    if model_name == "compare":
        candidate_results: list[tuple[dict, Pipeline]] = []
        for candidate in model_candidates:
            print(f"\nTraining candidate model: {candidate}")
            result_payload, model = train_single_model(
                x_train,
                y_train,
                x_test,
                y_test,
                candidate,
                random_state,
            )
            candidate_results.append((result_payload, model))

        candidate_results.sort(
            key=lambda item: (
                item[0]["test_accuracy"],
                item[0]["test_f1_weighted"],
            ),
            reverse=True,
        )
        best_result, best_model = candidate_results[0]
        print("\nModel comparison summary:")
        for result_payload, _ in candidate_results:
            print(
                f"- {result_payload['model_family']}: "
                f"accuracy={result_payload['test_accuracy']:.4f}, "
                f"f1={result_payload['test_f1_weighted']:.4f}, "
                f"cv={result_payload['cv_best_score']:.4f}"
            )
    else:
        best_result, best_model = train_single_model(
            x_train,
            y_train,
            x_test,
            y_test,
            model_name,
            random_state,
        )

    artifacts_dir = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best_model.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    payload = {
        "dataset": dataset or "sklearn.breast_cancer",
        "model_family": best_result["model_family"],
        "best_params": best_result["best_params"],
        "cv_best_score": best_result["cv_best_score"],
        "test_accuracy": best_result["test_accuracy"],
        "test_f1_weighted": best_result["test_f1_weighted"],
        "classification_report": best_result["classification_report"],
        "feature_count": int(x.shape[1]),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "xgboost_available": XGBOOST_AVAILABLE,
    }

    if label_encoder is not None:
        classes_path = artifacts_dir / "label_classes.json"
        classes_path.write_text(json.dumps(label_encoder.classes_.tolist(), indent=2), encoding="utf-8")
        payload["label_classes"] = label_encoder.classes_.tolist()

    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Selected model: {payload['model_family']}")
    print(f"Best CV accuracy: {payload['cv_best_score']:.4f}")
    print(f"Test accuracy: {payload['test_accuracy']:.4f}")
    print(f"Test weighted F1: {payload['test_f1_weighted']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature map for prediction")
    top_k: int = Field(default=3, ge=1, le=10)


class PredictResponse(BaseModel):
    predicted_label: str
    probabilities: list[dict[str, Any]]


MODEL_CACHE: dict[str, Any] = {
    "model": None,
    "metrics": {},
    "feature_names": [],
    "label_names": [],
}


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    return pd.read_json(metrics_path, typ="series").to_dict()


def _resolve_label_names(metrics: dict[str, Any], model: Any) -> list[str]:
    labels = metrics.get("label_classes")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]

    model_step = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    classes = getattr(model_step, "classes_", None)
    if classes is not None:
        return [str(label) for label in classes.tolist()]
    return []


def load_inference_assets(model_path: Path = DEFAULT_MODEL_PATH, metrics_path: Path = DEFAULT_METRICS_PATH) -> None:
    if MODEL_CACHE["model"] is not None:
        return

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found at ml/artifacts/best_model.joblib. "
            "Run training first before starting API/Streamlit."
        )

    model = joblib.load(model_path)
    metrics = _load_metrics(metrics_path)
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
    feature_names = list(getattr(preprocessor, "feature_names_in_", []))
    label_names = _resolve_label_names(metrics, model)

    MODEL_CACHE["model"] = model
    MODEL_CACHE["metrics"] = metrics
    MODEL_CACHE["feature_names"] = feature_names
    MODEL_CACHE["label_names"] = label_names


def create_api_app() -> FastAPI:
    api = FastAPI(title="MDroid ML Inference API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def health_payload() -> dict[str, Any]:
        load_inference_assets()
        return {
            "status": "ok",
            "model_path": str(DEFAULT_MODEL_PATH),
            "feature_count": len(MODEL_CACHE["feature_names"]),
            "labels": MODEL_CACHE["label_names"],
        }

    def predict_payload(request: PredictRequest) -> PredictResponse:
        load_inference_assets()

        feature_names = MODEL_CACHE["feature_names"]
        model = MODEL_CACHE["model"]
        label_names = MODEL_CACHE["label_names"]

        if not feature_names:
            raise HTTPException(status_code=500, detail="Model feature names are unavailable")

        missing = [name for name in feature_names if name not in request.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features: {', '.join(missing)}")

        row = pd.DataFrame(
            [[float(request.features[name]) for name in feature_names]],
            columns=feature_names,
        )

        prediction_raw = model.predict(row)[0]
        if label_names and str(prediction_raw).isdigit():
            pred_idx = int(prediction_raw)
            predicted_label = label_names[pred_idx] if 0 <= pred_idx < len(label_names) else str(prediction_raw)
        else:
            predicted_label = str(prediction_raw)

        if not hasattr(model, "predict_proba"):
            return PredictResponse(predicted_label=predicted_label, probabilities=[])

        probs = model.predict_proba(row)[0]
        class_names = label_names if label_names and len(label_names) == len(probs) else [f"Class {i}" for i in range(len(probs))]

        pairs = [{"label": class_names[idx], "probability": float(prob)} for idx, prob in enumerate(probs)]
        pairs.sort(key=lambda item: item["probability"], reverse=True)

        return PredictResponse(predicted_label=predicted_label, probabilities=pairs[: request.top_k])

    @api.get("/health")
    def health() -> dict[str, Any]:
        return health_payload()

    @api.get("/api/health")
    def health_api() -> dict[str, Any]:
        return health_payload()

    @api.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        return predict_payload(request)

    @api.post("/api/predict", response_model=PredictResponse)
    def predict_api(request: PredictRequest) -> PredictResponse:
        return predict_payload(request)

    if DEFAULT_FRONTEND_DIST.exists():
        index_file = DEFAULT_FRONTEND_DIST / "index.html"

        @api.get("/", include_in_schema=False)
        def serve_frontend_index() -> FileResponse:
            return FileResponse(index_file)

        @api.get("/{full_path:path}", include_in_schema=False)
        def serve_frontend_files(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")

            candidate = DEFAULT_FRONTEND_DIST / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)

            return FileResponse(index_file)

    return api


app = create_api_app()
# Keep compatibility for loaders expecting main:api
api = app


@st.cache_resource
def streamlit_load_model(model_path: Path):
    return joblib.load(model_path)


def is_binary_feature(name: str) -> bool:
    binary_keywords = {
        "fever",
        "cough",
        "headache",
        "nausea",
        "vomiting",
        "fatigue",
        "sore_throat",
        "chills",
        "body_pain",
        "loss_of_appetite",
        "abdominal_pain",
        "diarrhea",
        "sweating",
        "rapid_breathing",
        "dizziness",
    }
    return name in binary_keywords


def run_streamlit_ui() -> None:
    # Streamlit rejects invalid page_icon values; use a valid emoji and fallback safely.
    try:
        st.set_page_config(page_title="MDroid Symptom Predictor", page_icon="🩺", layout="wide")
    except Exception:
        st.set_page_config(page_title="MDroid Symptom Predictor", layout="wide")
    st.title("MDroid Symptom Predictor")
    st.caption("Unified app: training, API, and Streamlit inference from main.py")

    with st.sidebar:
        st.header("Model Setup")
        model_input = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
        metrics_input = st.text_input("Metrics path", str(DEFAULT_METRICS_PATH))

    model_path = Path(model_input)
    metrics_path = Path(metrics_input)

    if not model_path.exists():
        st.error(
            "Model file not found. Train first with:\n"
            "python main.py train --dataset data/symptom-based-disease-prediction-dataset/disease_prediction.csv --target label --model compare"
        )
        st.stop()

    try:
        model = streamlit_load_model(model_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    metrics = _load_metrics(metrics_path)
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
    feature_names = [str(x) for x in getattr(preprocessor, "feature_names_in_", [])]
    label_names = _resolve_label_names(metrics, model)

    if not feature_names:
        st.error("Could not infer feature names from model. Retrain and try again.")
        st.stop()

    with st.sidebar:
        st.header("Model Info")
        st.write(f"Feature count: {len(feature_names)}")
        if "model_family" in metrics:
            st.write(f"Model: {metrics['model_family']}")
        if "test_accuracy" in metrics:
            st.write(f"Test accuracy: {float(metrics['test_accuracy']):.4f}")

    st.subheader("Enter Symptoms")
    st.write("Set each symptom as present (1) or absent (0), then click Predict.")

    cols = st.columns(3)
    input_values: dict[str, float] = {}
    for idx, feature in enumerate(feature_names):
        with cols[idx % 3]:
            label = feature.replace("_", " ")
            if is_binary_feature(feature):
                input_values[feature] = 1.0 if st.checkbox(label, value=False) else 0.0
            else:
                input_values[feature] = float(st.number_input(label, value=0.0))

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([input_values], columns=feature_names)
        pred_idx = model.predict(row)[0]
        if label_names and isinstance(pred_idx, (int, float)):
            pred_int = int(pred_idx)
            prediction = label_names[pred_int] if 0 <= pred_int < len(label_names) else str(pred_idx)
        else:
            prediction = str(pred_idx)

        st.success(f"Predicted disease: {prediction}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row)[0]
            class_names = label_names if label_names and len(label_names) == len(probs) else [f"Class {i}" for i in range(len(probs))]
            prob_df = pd.DataFrame({"label": class_names, "probability": probs}).sort_values("probability", ascending=False)
            st.subheader("Prediction Confidence")
            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.set_index("label"))


def running_in_streamlit() -> bool:
    # In some hosted runs, script context may be unavailable briefly.
    # Environment variables are a reliable fallback for Streamlit runtime detection.
    if os.getenv("STREAMLIT_SERVER_PORT") or os.getenv("STREAMLIT_RUNTIME"):
        return True

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_streamlit_subprocess() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
    subprocess.run(cmd, check=True)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ML entrypoint for training, API, and Streamlit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", type=str, default=None)
    train_parser.add_argument("--target", type=str, default="target")
    train_parser.add_argument(
        "--model",
        choices=["random_forest", "log_reg", "xgboost", "compare"],
        default="random_forest",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--output-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))

    api_parser = subparsers.add_parser("api", help="Run inference API")
    api_parser.add_argument("--host", type=str, default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000)

    subparsers.add_parser("streamlit", help="Launch Streamlit UI")
    return parser.parse_args()


def run_cli() -> None:
    args = parse_cli_args()

    if args.command == "train":
        run_training(
            dataset=args.dataset,
            target=args.target,
            model_name=args.model,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
        )
        return

    if args.command == "api":
        uvicorn.run(app, host=args.host, port=args.port, reload=False)
        return

    if args.command == "streamlit":
        launch_streamlit_subprocess()
        return


if __name__ == "__main__":
    if running_in_streamlit():
        run_streamlit_ui()
    else:
        run_cli()
