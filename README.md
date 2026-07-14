# Loan Default Prediction System

Major Project — Data Science Internship
Submitted by: K Jasnavi Reddy

## 1. Project Overview

An end-to-end machine learning pipeline that predicts loan approval likelihood (used as a proxy for credit risk / default) from applicant demographic, financial, and credit history data. The project covers data cleaning, feature engineering, class-imbalance handling, model training and comparison, evaluation, explainability (SHAP), SQL analysis, and an interactive Streamlit dashboard.

**Final model:** Random Forest (200 trees) — Accuracy 0.85, F1 0.89, ROC-AUC 0.85 on held-out test data.

## 2. Dataset

- **Source:** [Loan Prediction Dataset — Analytics Vidhya / Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Size:** 614 rows, 13 columns
- **Target:** `Loan_Status` (Y = Approved, N = Not Approved), used as a proxy for creditworthiness/default risk.

## 3. Folder Structure

```
Loan_Default_Prediction/
├── dashboard/
│   └── app.py                            # Streamlit prediction dashboard
├── data/
│   └── raw/
│       ├── train_u6lujuX_CVtuZ9i.csv     # training data (Kaggle download)
│       └── test_Y3wMUE5_7gLdaTN.csv      # test data (Kaggle download)
├── models/
│   ├── rf_final.pkl                       # final trained model (Random Forest)
│   ├── xgb_v1.pkl                         # XGBoost model (comparison / SHAP illustration)
│   └── scaler_v1.pkl                      # StandardScaler used for Logistic Regression
├── notebooks/
│   └── 01_eda.ipynb                       # cleaning, feature engineering, EDA, modelling, evaluation, SHAP
├── outputs/
│   ├── eda_overview.png
│   ├── roc_pr_curves.png
│   └── shap_summary.png
├── Report/
│   └── Loan_Default_Prediction_Report.pdf
├── sql/
│   └── .sql                                # SQL queries used for aggregation
├── venv/                                    # local virtual environment (not submitted in ZIP)
└── README.md
```


## 4. How to Run

### 4.1 Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap streamlit jupyter ipykernel joblib
```

### 4.2 Run the notebook
Open `notebooks/01_eda.ipynb` in VS Code / Jupyter, select the `venv` kernel, and run all cells top to bottom. This reproduces cleaning, feature engineering, model training, evaluation, and saves the model artifacts to `models/`.

### 4.3 Run the dashboard
```bash
cd dashboard
streamlit run app.py
```
Then open the local URL shown in the terminal (usually `http://localhost:8501`). Enter applicant details to get a live approval probability and risk classification.

## 5. Methodology Summary

1. **Cleaning:** missing values imputed (mode for categoricals, median for `LoanAmount`, mode for `Loan_Amount_Term` and `Credit_History`).
2. **Feature engineering:** `TotalIncome`, `LoanAmount_to_Income`, `EMI`, `BalanceIncome`, log-transforms of skewed income/loan columns.
3. **Encoding:** binary label encoding for Yes/No fields; one-hot encoding for `Property_Area`.
4. **SQL:** cleaned data loaded into SQLite; aggregation queries by `Property_Area` (avg loan amount, application count, approval count).
5. **Imbalance handling:** 80/20 stratified train-test split; SMOTE applied to the training set only.
6. **Modelling:** Logistic Regression, Random Forest, and XGBoost trained and compared.
7. **Evaluation:** precision, recall, F1, ROC-AUC, confusion matrix, ROC and Precision-Recall curves — accuracy alone was not used as the deciding metric due to class imbalance.
8. **Explainability:** SHAP summary plot to identify and rank the features driving predictions.
9. **Deployment:** Streamlit dashboard for real-time, interactive predictions.

## 6. Results

| Model | Accuracy | F1 (class 1) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.80 | 0.86 | 0.83 |
| **Random Forest (final)** | **0.85** | **0.89** | **0.85** |
| XGBoost | 0.81 | 0.86 | 0.81 |

Top predictive features (via SHAP): `Credit_History` (by far the strongest), `ApplicantIncome`, `LoanAmount`, `LoanAmount_to_Income`.

## 7. Future Enhancements

- Swap in a true default-labelled dataset (Lending Club / Home Credit Default Risk) using the same pipeline.
- Hyperparameter-tune XGBoost to test whether it can surpass the current Random Forest baseline.
- Deploy via FastAPI + Docker; have the Streamlit app call the API rather than load the model directly.
- Add model-monitoring for data/prediction drift.
- Try deep learning approaches for tabular data (e.g. TabNet) as a stretch comparison.

## 8. Author

K Jasnavi Reddy — Data Science Internship