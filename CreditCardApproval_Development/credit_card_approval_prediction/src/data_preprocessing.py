"""
data_preprocessing.py
----------------------
Cleans the raw dataset and engineers features for model training:
 - handles missing values
 - removes duplicate records
 - encodes categorical variables into numerical format
 - converts the multi-class PAYMENT_STATUS into a binary risk flag
 - derives a few useful features (age, employment length, debt-to-income ratio)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/credit_card_data.csv"
CLEAN_PATH = "data/credit_card_data_clean.csv"

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] // 365).astype(int)

    # 365243 is the sentinel used for "not currently employed" (e.g. pensioners)
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] // 365).clip(lower=0)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "EMPLOYMENT_YEARS"] = 0

    df["DEBT_TO_INCOME_RATIO"] = (
        df["EXISTING_LOAN_BALANCE"] / df["AMT_INCOME_TOTAL"].replace(0, df["AMT_INCOME_TOTAL"].median())
    ).round(4)

    # Convert the multi-class payment status into a binary high-risk label:
    # C (paid off) and X (no loan) and 0 (<30 days past due) => not high risk
    # 1-5 (30+ days past due / written off) => high risk
    high_risk_codes = {"1", "2", "3", "4", "5"}
    df["IS_HIGH_RISK"] = df["PAYMENT_STATUS"].astype(str).isin(high_risk_codes).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame, columns=None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    columns = columns or CATEGORICAL_COLUMNS
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def build_clean_dataset(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = engineer_features(df)
    df, _ = encode_categoricals(df)
    return df


if __name__ == "__main__":
    clean_df = build_clean_dataset()
    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"Clean dataset written to {CLEAN_PATH}  ({clean_df.shape[0]} rows, {clean_df.shape[1]} columns)")
    print(clean_df.head())
