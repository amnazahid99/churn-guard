from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

CLASS_LABELS = {
    0: "No Risk",
    1: "Low Risk",
    2: "Medium Risk",
    3: "High Risk / Churn",
}

YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

app = FastAPI(
    title="ChurnGuard AI API",
    description="Multi-class churn severity inference API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.joblib"
EXPLAINER_PATH = MODEL_DIR / "shap_explainer.joblib"

model = None
scaler = None
feature_names: list[str] = []
explainer = None
last_batch_predictions: list[dict[str, Any]] = []


def canonical_column(name: str) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def load_artifacts() -> None:
    global model, scaler, feature_names, explainer

    if not all(path.exists() for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, EXPLAINER_PATH]):
        model = None
        scaler = None
        feature_names = []
        explainer = None
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    explainer = joblib.load(EXPLAINER_PATH)


def ensure_model_loaded() -> None:
    if model is None or scaler is None or not feature_names or explainer is None:
        load_artifacts()

    if model is None or scaler is None or not feature_names or explainer is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded. Run train.py first.")


def revenue_at_risk(probabilities: np.ndarray, monthly_charges: float) -> float:
    p_medium = float(probabilities[2])
    p_high = float(probabilities[3])
    return round(float(monthly_charges) * 12 * (p_medium + p_high), 2)


def encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()
    encoded.columns = [column.strip() for column in encoded.columns]

    if "TotalCharges" in encoded.columns:
        encoded["TotalCharges"] = pd.to_numeric(encoded["TotalCharges"], errors="coerce")

    drop_columns = [column for column in ["customerID", "Churn", "risk_label"] if column in encoded.columns]
    if drop_columns:
        encoded = encoded.drop(columns=drop_columns)

    for column in YES_NO_COLS:
        if column in encoded.columns:
            if encoded[column].dtype == object:
                encoded[column] = encoded[column].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0)
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(0).astype(int)

    if "gender" in encoded.columns:
        if encoded["gender"].dtype == object:
            encoded["gender"] = (encoded["gender"].astype(str).str.strip().str.lower() == "male").astype(int)
        else:
            encoded["gender"] = pd.to_numeric(encoded["gender"], errors="coerce").fillna(0).astype(int)

    object_columns = encoded.select_dtypes(include="object").columns.tolist()
    if object_columns:
        encoded = pd.get_dummies(encoded, columns=object_columns, drop_first=True)

    return encoded


def align_columns(frame: pd.DataFrame) -> pd.DataFrame:
    canonical_to_actual = {canonical_column(column): column for column in frame.columns}
    aligned = pd.DataFrame(index=frame.index)

    for target in feature_names:
        match = canonical_to_actual.get(canonical_column(target))
        if match is None:
            aligned[target] = 0.0
        else:
            aligned[target] = pd.to_numeric(frame[match], errors="coerce").fillna(0.0)

    return aligned


def prepare_input(payload: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(payload, dict):
        raw = pd.DataFrame([payload])
    else:
        raw = pd.DataFrame(payload)

    if raw.empty:
        raise HTTPException(status_code=400, detail="Input data is empty.")

    encoded = encode_features(raw)
    aligned = align_columns(encoded)
    return raw, aligned


def run_prediction(aligned: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = scaler.transform(aligned)
    probabilities = model.predict_proba(scaled)
    predicted = model.predict(scaled)
    return scaled, probabilities, predicted


def build_prediction_response(raw_row: pd.Series, aligned_row: pd.Series, proba_row: np.ndarray, predicted_class: int) -> dict[str, Any]:
    churn_probability = float(proba_row[3])
    confidence = float(np.max(proba_row))
    monthly_charges = float(pd.to_numeric(raw_row.get("MonthlyCharges", 0.0), errors="coerce") or 0.0)

    return {
        "churn_probability": round(churn_probability, 4),
        "risk_class": int(predicted_class),
        "risk_level": CLASS_LABELS[int(predicted_class)],
        "revenue_at_risk": revenue_at_risk(proba_row, monthly_charges),
        "confidence": round(confidence, 4),
        "probabilities": {
            "no_risk": round(float(proba_row[0]), 4),
            "low_risk": round(float(proba_row[1]), 4),
            "medium_risk": round(float(proba_row[2]), 4),
            "high_risk": round(float(proba_row[3]), 4),
        },
        "monthly_charges": round(monthly_charges, 2),
        "contract_type": str(raw_row.get("Contract", "Unknown")),
        "customer_features": {
            feature: float(aligned_row[feature]) for feature in feature_names if feature in aligned_row.index
        },
    }


def shap_waterfall_for_row(scaled_row: np.ndarray, aligned_row: pd.Series, predicted_class: int) -> list[dict[str, Any]]:
    shap_values = explainer.shap_values(scaled_row)

    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_shap = shap_values[0, :, predicted_class]
    else:
        class_shap = shap_values[0]

    top_indices = np.argsort(np.abs(class_shap))[::-1][:8]
    waterfall = []

    for index in top_indices:
        feature_name = feature_names[index]
        waterfall.append(
            {
                "feature": feature_name,
                "value": round(float(class_shap[index]), 6),
                "customer_value": round(float(aligned_row.iloc[index]), 6),
            }
        )

    return waterfall


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/health")
def health() -> dict[str, Any]:
    load_artifacts()
    loaded = model is not None and scaler is not None and bool(feature_names) and explainer is not None
    return {
        "status": "ok" if loaded else "not_ready",
        "model_loaded": loaded,
        "feature_count": len(feature_names),
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    ensure_model_loaded()
    return {
        "model_type": type(model).__name__,
        "num_classes": 4,
        "class_labels": CLASS_LABELS,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }


@app.get("/model/feature-importance")
def model_feature_importance() -> dict[str, Any]:
    ensure_model_loaded()

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise HTTPException(status_code=500, detail="Feature importance is unavailable for this model.")

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    return {
        "top_features": [
            {"feature": str(row.feature), "importance": round(float(row.importance), 6)}
            for row in ranking.itertuples(index=False)
        ]
    }


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_model_loaded()
    raw, aligned = prepare_input(payload)
    _, probabilities, predicted = run_prediction(aligned)
    return build_prediction_response(raw.iloc[0], aligned.iloc[0], probabilities[0], int(predicted[0]))


@app.post("/predict/explain")
def predict_explain(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_model_loaded()
    raw, aligned = prepare_input(payload)
    scaled, probabilities, predicted = run_prediction(aligned)

    prediction = build_prediction_response(raw.iloc[0], aligned.iloc[0], probabilities[0], int(predicted[0]))
    prediction["shap_waterfall"] = shap_waterfall_for_row(scaled[:1], aligned.iloc[0], int(predicted[0]))
    return prediction


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    global last_batch_predictions
    ensure_model_loaded()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    raw = pd.read_csv(BytesIO(content))
    raw_rows, aligned = prepare_input(raw)
    _, probabilities, predicted = run_prediction(aligned)

    predictions = []
    for index in range(len(raw_rows)):
        row_prediction = build_prediction_response(raw_rows.iloc[index], aligned.iloc[index], probabilities[index], int(predicted[index]))
        row_prediction["row"] = index
        predictions.append(row_prediction)

    predictions.sort(key=lambda item: item["churn_probability"], reverse=True)
    last_batch_predictions = predictions
    return {
        "filename": file.filename,
        "total": len(predictions),
        "predictions": predictions,
    }


@app.get("/cohort-analysis")
def cohort_analysis() -> dict[str, Any]:
    ensure_model_loaded()

    if not last_batch_predictions:
        return {"cohorts": []}

    cohort_frame = pd.DataFrame(last_batch_predictions)
    if "contract_type" not in cohort_frame.columns:
        return {"cohorts": []}

    grouped = (
        cohort_frame.groupby("contract_type", dropna=False)
        .agg(
            avg_churn_probability=("churn_probability", "mean"),
            customer_count=("churn_probability", "size"),
            total_revenue_at_risk=("revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values("avg_churn_probability", ascending=False)
    )

    return {
        "cohorts": [
            {
                "contract_type": str(row.contract_type),
                "avg_churn_probability": round(float(row.avg_churn_probability), 4),
                "customer_count": int(row.customer_count),
                "total_revenue_at_risk": round(float(row.total_revenue_at_risk), 2),
            }
            for row in grouped.itertuples(index=False)
        ]
    }
