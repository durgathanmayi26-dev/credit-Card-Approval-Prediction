"""
generate_synthetic_data.py
---------------------------
Generates a synthetic credit-card-approval dataset that mirrors the schema of
public "Credit Card Approval Prediction" datasets (e.g. the Kaggle dataset by
rikdifos), so the rest of the pipeline (EDA, preprocessing, training, Flask
app) can run end-to-end without needing an internet download.

If you have the real dataset (application_record.csv + credit_record.csv),
just drop it into the data/ folder as `credit_card_data.csv` with the same
column names used below and skip running this script.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 5000


def generate(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["M", "F"], size=n_rows, p=[0.45, 0.55])
    own_car = rng.choice(["Y", "N"], size=n_rows, p=[0.4, 0.6])
    own_realty = rng.choice(["Y", "N"], size=n_rows, p=[0.55, 0.45])
    cnt_children = rng.poisson(0.6, size=n_rows).clip(0, 5)

    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n_rows,
        p=[0.55, 0.20, 0.15, 0.08, 0.02],
    )
    education_type = rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
        size=n_rows,
        p=[0.45, 0.35, 0.10, 0.07, 0.03],
    )
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n_rows,
        p=[0.55, 0.20, 0.10, 0.10, 0.05],
    )
    housing_type = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n_rows,
        p=[0.70, 0.12, 0.08, 0.06, 0.02, 0.02],
    )

    annual_income = rng.lognormal(mean=11.2, sigma=0.5, size=n_rows).round(2)  # roughly 40k - 300k
    age_years = rng.integers(21, 70, size=n_rows)
    days_birth = -(age_years * 365 + rng.integers(0, 365, size=n_rows))

    employed_mask = income_type != "Pensioner"
    employment_years = np.where(
        employed_mask,
        rng.gamma(shape=2.0, scale=3.0, size=n_rows).clip(0, 40),
        0,
    )
    days_employed = np.where(employed_mask, -(employment_years * 365).astype(int), 365243)  # 365243 = "not employed" sentinel used in the real dataset

    cnt_fam_members = (cnt_children + rng.integers(1, 3, size=n_rows)).clip(1, 8)

    # Multi-class payment status (mirrors the STATUS column in the real credit_record.csv)
    # 0 = 1-29 days past due ... 5 = written off, C = paid off that month, X = no loan this month
    payment_status = rng.choice(
        ["C", "X", "0", "1", "2", "3", "4", "5"],
        size=n_rows,
        p=[0.35, 0.30, 0.20, 0.07, 0.03, 0.02, 0.02, 0.01],
    )

    existing_loan_balance = rng.gamma(shape=2.0, scale=4000, size=n_rows).round(2)
    credit_inquiries_last_6m = rng.poisson(1.2, size=n_rows).clip(0, 12)

    df = pd.DataFrame(
        {
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty,
            "CNT_CHILDREN": cnt_children,
            "AMT_INCOME_TOTAL": annual_income,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education_type,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": housing_type,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "CNT_FAM_MEMBERS": cnt_fam_members,
            "EXISTING_LOAN_BALANCE": existing_loan_balance,
            "CREDIT_INQUIRIES_6M": credit_inquiries_last_6m,
            "PAYMENT_STATUS": payment_status,
        }
    )

    # ---- Introduce a small amount of realistic messiness ----
    # missing values
    for col in ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "EXISTING_LOAN_BALANCE"]:
        missing_idx = rng.choice(n_rows, size=int(n_rows * 0.02), replace=False)
        df.loc[missing_idx, col] = np.nan

    # duplicate rows
    dup_idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # ---- Derive the binary target: is_high_risk (mirrors converting PAYMENT_STATUS to binary) ----
    risk_from_status = df["PAYMENT_STATUS"].isin(["2", "3", "4", "5"]).astype(int)
    debt_to_income = df["EXISTING_LOAN_BALANCE"].fillna(0) / df["AMT_INCOME_TOTAL"].fillna(df["AMT_INCOME_TOTAL"].median())
    risk_score = (
        0.5 * risk_from_status
        + 0.3 * (debt_to_income > debt_to_income.quantile(0.60)).astype(int)
        + 0.25 * (df["CREDIT_INQUIRIES_6M"] > 2).astype(int)
        + 0.15 * (df["EXISTING_LOAN_BALANCE"] > df["EXISTING_LOAN_BALANCE"].quantile(0.70)).astype(int)
    )
    noise = rng.normal(0, 0.2, size=len(df))
    # threshold tuned so roughly 55-65% of applicants are approved (realistic, and
    # balanced enough for the models to learn a genuine signal rather than just
    # predicting the majority class)
    df["TARGET_APPROVED"] = np.where((risk_score + noise) < 0.3, 1, 0)

    return df


if __name__ == "__main__":
    dataset = generate()
    out_path = "data/credit_card_data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to {out_path}  ({len(dataset)} rows, {dataset.shape[1]} columns)")
    print(dataset["TARGET_APPROVED"].value_counts(normalize=True).rename("share"))
