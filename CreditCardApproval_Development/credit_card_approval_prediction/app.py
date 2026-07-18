"""
app.py
------
Flask web application for the Credit Card Approval Prediction system.

Routes:
  GET  /            Home page introduction
  GET  /predict      Prediction input form
  POST /predict      Runs the trained model on submitted applicant details
                      and renders the result page.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, render_template, request

MODELS_DIR = "models"

app = Flask(__name__)

# ---- Load trained model artifacts at startup ----
model = None
scaler = None
feature_columns = []
model_meta = {}

CATEGORY_MAPS = {
    "CODE_GENDER": {"Male": "M", "Female": "F"},
    "FLAG_OWN_CAR": {"Yes": "Y", "No": "N"},
    "FLAG_OWN_REALTY": {"Yes": "Y", "No": "N"},
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    "NAME_HOUSING_TYPE": [
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ],
}


def load_artifacts():
    global model, scaler, feature_columns, model_meta
    model_path = f"{MODELS_DIR}/best_model.pkl"
    scaler_path = f"{MODELS_DIR}/scaler.pkl"
    features_path = f"{MODELS_DIR}/feature_columns.json"
    meta_path = f"{MODELS_DIR}/model_meta.json"

    if not os.path.exists(model_path):
        print("WARNING: no trained model found. Run `python src/train_models.py` first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_columns = json.load(f)
    with open(meta_path) as f:
        model_meta = json.load(f)


def encode_label(column: str, value: str) -> int:
    """Re-creates the same ordinal encoding used by LabelEncoder in preprocessing
    (alphabetical order of the categories seen during training)."""
    options = CATEGORY_MAPS[column]
    if isinstance(options, dict):
        value = options.get(value, value)
        classes = sorted(set(options.values()))
    else:
        classes = sorted(options)
    return classes.index(value) if value in classes else 0


@app.route("/")
def home():
    return render_template("index.html", model_meta=model_meta)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", income_types=CATEGORY_MAPS["NAME_INCOME_TYPE"],
                                education_types=CATEGORY_MAPS["NAME_EDUCATION_TYPE"],
                                family_statuses=CATEGORY_MAPS["NAME_FAMILY_STATUS"],
                                housing_types=CATEGORY_MAPS["NAME_HOUSING_TYPE"])

    if model is None:
        return render_template("result.html", error="No trained model found. Please run training first.")

    form = request.form
    try:
        gender = encode_label("CODE_GENDER", form["gender"])
        own_car = encode_label("FLAG_OWN_CAR", form["own_car"])
        own_realty = encode_label("FLAG_OWN_REALTY", form["own_realty"])
        cnt_children = int(form["cnt_children"])
        annual_income = float(form["annual_income"])
        income_type = encode_label("NAME_INCOME_TYPE", form["income_type"])
        education_type = encode_label("NAME_EDUCATION_TYPE", form["education_type"])
        family_status = encode_label("NAME_FAMILY_STATUS", form["family_status"])
        housing_type = encode_label("NAME_HOUSING_TYPE", form["housing_type"])
        age_years = int(form["age_years"])
        employment_years = float(form["employment_years"])
        cnt_fam_members = int(form["cnt_fam_members"])
        existing_loan_balance = float(form["existing_loan_balance"])
        credit_inquiries_6m = int(form["credit_inquiries_6m"])

        debt_to_income_ratio = existing_loan_balance / annual_income if annual_income else 0

        row = {
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty,
            "CNT_CHILDREN": cnt_children,
            "AMT_INCOME_TOTAL": annual_income,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education_type,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": housing_type,
            "CNT_FAM_MEMBERS": cnt_fam_members,
            "EXISTING_LOAN_BALANCE": existing_loan_balance,
            "CREDIT_INQUIRIES_6M": credit_inquiries_6m,
            "AGE_YEARS": age_years,
            "EMPLOYMENT_YEARS": employment_years,
            "DEBT_TO_INCOME_RATIO": debt_to_income_ratio,
        }

        # Ensure column order matches training (use a DataFrame so sklearn doesn't warn
        # about missing feature names)
        import pandas as pd

        input_vector = pd.DataFrame([[row.get(col, 0) for col in feature_columns]], columns=feature_columns)

        if model_meta.get("needs_scaling"):
            input_vector = scaler.transform(input_vector)

        prediction = model.predict(input_vector)[0]
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_vector)[0][1])

        result_text = "Approved" if prediction == 1 else "Rejected"
        return render_template(
            "result.html",
            result=result_text,
            probability=probability,
            model_name=model_meta.get("best_model_name", "Model"),
        )
    except Exception as exc:  # keep it simple and user-friendly for a student project
        return render_template("result.html", error=f"Could not process your input: {exc}")


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
else:
    load_artifacts()
