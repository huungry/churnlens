# ChurnLens

A machine learning app that predicts whether a telecom customer will churn. It trains and compares three classifiers, keeps the best one (XGBoost), and serves predictions through a Streamlit web app for both single customers and CSV batches.

The model is trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Features

- **Single-customer prediction** — fill in a customer's attributes and get a churn label, probability, and risk level.
- **Batch prediction** — upload a CSV of customers and download predictions for all of them.
- **Risk levels** — probabilities are bucketed into Low, Moderate, Elevated, and High Risk.
- **Model selection** — training compares Logistic Regression, Random Forest, and XGBoost with randomized hyperparameter search (5-fold CV, scored on F1) and saves the best estimator.

## Project structure

```
.
├── app.py                    # Streamlit web app (single + batch prediction)
├── app_predict_process.py    # Encoding, prediction, and risk-label helpers for the app
├── app_utils.py              # UI helpers: risk display and uploaded-CSV validation
├── src/
│   ├── preprocess.py         # Data cleaning, encoding, train/test split
│   ├── train.py              # Train and compare models, save the best one
│   ├── confusion_matrix.py   # Plot a confusion matrix from saved predictions
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis with data visualization
├── data/
│   └── raw/                  # Telco Customer Churn CSV
└── model/                    # Saved model, transformer, and cached test data (.joblib)
```

## Requirements

- Python 3.15

## Setup

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the web app

```bash
streamlit run app.py
```

This loads the saved model and transformer from `model/` and opens the app in your browser. Use the **Single Customer Prediction** form for one customer, or upload a CSV under **Batch Predictions**. The app shows a sample CSV so you can match the expected columns and value formats.

CSV upload rules (also shown in the app):

1. Use the same wording, spelling, and formats as the single-prediction form for all feature values.
2. Include every required feature column (use the sample CSV for column names).
3. When `InternetService` is `No`, all internet-related features must be `No internet service`.
4. When `PhoneService` is `No`, `MultipleLines` must be `No phone service`.
5. `SeniorCitizen` may be `Yes`/`No` or `1`/`0`.
6. Numeric fields must contain plain numbers, with no blanks or currency symbols.

### Retrain the model

Training reads `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, writes the model and preprocessing artifacts to `model/`, and prints a CV F1 score, test ROC-AUC, best parameters, and a classification report for each model.

```bash
cd src
python train.py
```

### Plot the confusion matrix

After training (so the `model/*.joblib` prediction files exist):

```bash
python src/confusion_matrix.py
```

## How it works

1. **Preprocessing** (`src/preprocess.py`) — trims whitespace, coerces numeric columns, fills `TotalCharges` for zero-tenure customers, one-hot encodes categorical features, scales numeric features, and splits into train/test sets. The fitted transformer and cached test data are saved to `model/`.
2. **Training** (`src/train.py`) — runs randomized hyperparameter search over Logistic Regression, Random Forest, and XGBoost, picks the model with the best cross-validated F1 score, and saves it as `model/model.joblib`.
3. **Prediction** (`app_predict_process.py`) — reloads the saved transformer and model, encodes incoming data the same way, and returns the churn probability and a 50% cutoff prediction. Probabilities map to risk labels: Low (≤25%), Moderate (>25%), Elevated (>50%), High (>80%).

## Notes

- The classifier uses a 50% cutoff: scores of 50% or higher are labeled Churn, below 50% as No Churn.
- Predictions reflect patterns in the training data and are not guaranteed accurate.
