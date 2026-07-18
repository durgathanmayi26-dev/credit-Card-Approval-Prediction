"""
eda.py
------
Generates exploratory data analysis plots (count plots & distribution plots)
and saves them to the outputs/ folder.
"""

import os

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW_PATH = "data/credit_card_data.csv"
OUTPUT_DIR = "outputs"


def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    plt.figure(figsize=(5, 4))
    sns.countplot(x="TARGET_APPROVED", hue="TARGET_APPROVED", data=df, palette="Set2", legend=False)
    plt.title("Approval Outcome Distribution")
    plt.xlabel("0 = Rejected, 1 = Approved")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/target_distribution.png", dpi=120)
    plt.close()


def plot_categorical_counts(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    categorical_cols = ["CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"]
    for col in categorical_cols:
        plt.figure(figsize=(7, 4))
        order = df[col].value_counts().index
        sns.countplot(y=col, hue=col, data=df, order=order, palette="viridis", legend=False)
        plt.title(f"Applicant Count by {col}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/countplot_{col.lower()}.png", dpi=120)
        plt.close()


def plot_numeric_distributions(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    numeric_cols = ["AMT_INCOME_TOTAL", "EXISTING_LOAN_BALANCE", "CREDIT_INQUIRIES_6M"]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/distplot_{col.lower()}.png", dpi=120)
        plt.close()


def run_eda(raw_path: str = RAW_PATH, out_dir: str = OUTPUT_DIR) -> None:
    ensure_output_dir(out_dir)
    df = pd.read_csv(raw_path)
    plot_target_distribution(df, out_dir)
    plot_categorical_counts(df, out_dir)
    plot_numeric_distributions(df, out_dir)
    print(f"EDA plots saved to {out_dir}/")


if __name__ == "__main__":
    run_eda()
