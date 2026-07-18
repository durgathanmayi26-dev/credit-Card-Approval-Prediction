# Credit Card Approval Prediction

A hands-on machine learning project that predicts whether a credit card
application will be approved or rejected, using Python, Flask, and
classic ML classification algorithms, with an optional IBM Watson
Machine Learning cloud deployment path.

## Project Structure

```
credit_card_approval_prediction/
├── app.py                     # Flask web application
├── requirements.txt
├── data/
│   ├── generate_synthetic_data.py   # creates a synthetic dataset (no internet needed)
│   ├── credit_card_data.csv         # raw dataset (generated)
│   └── credit_card_data_clean.csv   # cleaned dataset (generated)
├── src/
│   ├── data_preprocessing.py  # cleaning, feature engineering, encoding
│   ├── eda.py                 # exploratory data analysis / plots
│   └── train_models.py        # trains & compares LR, Decision Tree, RF, XGBoost
├── models/                    # saved best_model.pkl, scaler.pkl, metadata (generated)
├── outputs/                   # EDA charts, confusion matrices, accuracy comparison (generated)
├── templates/                 # Flask HTML templates (home, predict form, result)
└── static/style.css           # Flask app styling
```

## 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.8+.

## 2. Get a Dataset

**Option A — quick start (synthetic data):**
```bash
cd credit_card_approval_prediction
python data/generate_synthetic_data.py
```
This creates `data/credit_card_data.csv` with realistic applicant fields
(income, employment, credit history, payment status, etc.) so the rest of
the pipeline runs end-to-end without needing an external download.

**Option B — real dataset:**
Download a public credit card approval dataset (e.g. the Kaggle
"Credit Card Approval Prediction" dataset) and save it as
`data/credit_card_data.csv`, matching the column names used in
`src/data_preprocessing.py`. Adjust column names there if they differ.

## 3. Exploratory Data Analysis

```bash
python src/eda.py
```
Saves count plots and distribution plots to `outputs/`.

## 4. Preprocessing + Model Training

```bash
python src/train_models.py
```
This will:
- clean the data (missing values, duplicates, encoding, binary risk label)
- train Logistic Regression, Decision Tree, Random Forest, and XGBoost
- print accuracy / confusion matrix / classification report for each
- save the best-performing model to `models/best_model.pkl`
- save a model accuracy comparison chart to `outputs/model_accuracy_comparison.png`

## 5. Run the Web Application

```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser:
- **Home** — project introduction
- **Check Eligibility** — enter applicant details and get an instant
  approval / rejection prediction with model confidence

## 6. (Optional) Cloud Deployment via IBM Watson Machine Learning

Once you're happy with the trained model, you can register and deploy it
using the `ibm-watson-machine-learning` Python SDK so the same prediction
logic can be served from the cloud instead of (or alongside) the local
Flask app. This step requires an IBM Cloud account and a Watson Machine
Learning service instance.

## Notes

- The synthetic dataset is generated with a deliberately simple rule
  (debt-to-income ratio + credit inquiries + payment history) driving the
  approval label, so the models have a genuine signal to learn from. Swap
  in a real dataset for production-quality results.
- `PAYMENT_STATUS` mirrors the multi-class `STATUS` column found in public
  credit bureau datasets, and is converted into a binary `IS_HIGH_RISK`
  flag during preprocessing, matching the project's feature engineering
  requirement.
