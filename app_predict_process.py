import pandas as pd
import joblib
from src.preprocess import fix_data

def get_churn_risk_label(pred_prob):
    if pred_prob > 0.8:
        return 'High Risk'
    elif pred_prob > 0.50:
        return 'Elevated Risk'
    elif pred_prob > 0.25:
        return 'Moderate Risk'
    else:
        return 'Low Risk'

def fix(data):
    data = fix_data(data)
    return data

def encode(data):
    data['SeniorCitizen'] = data['SeniorCitizen'].replace({'Yes': 1, 'No': 0})
    process_data = joblib.load('model/transformer.joblib')
    encoded_data = process_data.transform(data)
    return encoded_data

def default_mode_predict(data):
    model = joblib.load('model/model.joblib')
    pred_prob = (model.predict_proba(data)[:, 1])
    pred = model.predict(data)
    return pred_prob, pred
