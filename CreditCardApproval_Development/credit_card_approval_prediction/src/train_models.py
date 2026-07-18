"""
train_models.py
-----------------
Trains four classification algorithms on the cleaned dataset:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

Evaluates each with accuracy, a confusion matrix, and a classification report,
then saves the best-performing model (plus the feature list and scaler) to
the models/ folder for use by the Flask app.
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_preprocessing import build_clean_dataset

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
TARGET_COL = "TARGET_APPROVED"
DROP_COLS = ["TARGET_APPROVED", "PAYMENT_STATUS", "IS_HIGH_RISK", "DAYS_BIRTH", "DAYS_EMPLOYED"]


def get_feature_target(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, feature_cols


def evaluate_model(name, model, X_test, y_test, out_dir: str = OUTPUTS_DIR) -> dict:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix \u2014 {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=120)
    plt.close()

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    return {"name": name, "accuracy": acc, "report": report}


def train_all_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = build_clean_dataset()
    X, y, feature_cols = get_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    trained_models = {}

    # Logistic Regression (benefits from scaled features)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_scaled, y_test))
    trained_models["Logistic Regression"] = (lr, True)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    results.append(evaluate_model("Decision Tree", dt, X_test, y_test))
    trained_models["Decision Tree"] = (dt, False)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))
    trained_models["Random Forest"] = (rf, False)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost", xgb, X_test, y_test))
    trained_models["XGBoost"] = (xgb, False)

    # ---- Pick the best model by accuracy ----
    best = max(results, key=lambda r: r["accuracy"])
    best_model, needs_scaling = trained_models[best["name"]]
    print(f"\nBest performing model: {best['name']} (accuracy = {best['accuracy']:.4f})")

    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(f"{MODELS_DIR}/model_meta.json", "w") as f:
        json.dump(
            {
                "best_model_name": best["name"],
                "needs_scaling": needs_scaling,
                "accuracy": best["accuracy"],
                "all_results": [{"name": r["name"], "accuracy": r["accuracy"]} for r in results],
            },
            f,
            indent=2,
        )

    # ---- Model comparison bar chart ----
    plt.figure(figsize=(6, 4))
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    sns.barplot(x=names, y=accs, hue=names, palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=15)
    for i, a in enumerate(accs):
        plt.text(i, a + 0.02, f"{a*100:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/model_accuracy_comparison.png", dpi=120)
    plt.close()

    print(f"\nModel + scaler + metadata saved to {MODELS_DIR}/")
    return results


if __name__ == "__main__":
    train_all_models()
