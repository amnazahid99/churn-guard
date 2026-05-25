from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("data/telco_churn.csv")
MODEL_DIR = Path("model")
RANDOM_STATE = 42


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)

    score += (df["tenure"] <= 3).astype(float) * 2.5
    score += (df["tenure"] <= 12).astype(float) * 1.0
    score += (df["Contract"] == "Month-to-month").astype(float) * 1.5
    score += (df["PaymentMethod"] == "Electronic check").astype(float) * 0.8
    score += (df["InternetService"] == "Fiber optic").astype(float) * 0.6
    score += (df["OnlineSecurity"] == "No").astype(float) * 0.5
    score += (df["TechSupport"] == "No").astype(float) * 0.4
    score += (df["MonthlyCharges"] > 70).astype(float) * 0.5

    score -= (df["Contract"] == "Two year").astype(float) * 1.5
    score -= (df["tenure"] >= 36).astype(float) * 1.0
    score -= (df["PaymentMethod"] == "Bank transfer (automatic)").astype(float) * 0.5

    return score


def score_to_class(score: float) -> int:
    if score < 1.5:
        return 0
    if score < 3.0:
        return 1
    if score < 5.0:
        return 2
    return 3


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/telco_churn.csv. Download WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle, rename it to telco_churn.csv, and place it under data/."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)

    df["risk_label"] = compute_risk_score(df).apply(score_to_class)
    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)

    yes_no_cols = [
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
    for col in yes_no_cols:
        df[col] = (df[col] == "Yes").astype(int)

    df["gender"] = (df["gender"] == "Male").astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=["risk_label"])
    y = df["risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_val_score(xgb, X_train_sc, y_train_res, cv=5, scoring="f1_macro")
    print(f"CV F1-Macro: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")

    xgb.fit(X_train_sc, y_train_res)

    y_pred = xgb.predict(X_test_sc)
    y_proba = xgb.predict_proba(X_test_sc)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Risk", "Low Risk", "Medium Risk", "High Risk"],
        )
    )
    print(f"AUC-ROC (macro): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.4f}")

    explainer = shap.TreeExplainer(xgb)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb, MODEL_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_names.joblib")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.joblib")
    print("All artifacts saved to model/")


if __name__ == "__main__":
    main()
