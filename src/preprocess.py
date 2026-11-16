import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model"

def fix_data(df):
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()
    numeric_cols = ['TotalCharges', 'tenure', 'MonthlyCharges', 'SeniorCitizen']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    total_na = (df["tenure"] == 0) & df["TotalCharges"].isna()
    if total_na.any():
        df.loc[total_na, 'TotalCharges'] = 0
    return df

def encode_numerical_columns(df):
    num_encoder = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = numerical_columns.drop('tenure')
    return num_encoder, numerical_columns

def encode_categorical_columns(df):
    cols = df.select_dtypes(include=['object', 'str']).columns
    cols = [col for col in cols if col not in ['customerID', 'Churn']]
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    return cat_encoder, cols

def encode_churn_column(df):
    churn_encoder = LabelEncoder()
    return churn_encoder

def encode_cat_num_data(df): 
    num_encoder, num_cols = encode_numerical_columns(df)
    cat_encoder, cat_cols = encode_categorical_columns(df)

    #One hot encode categorical values & scale numerical values
    transformer = ColumnTransformer(transformers=[('cols_encoder', cat_encoder, cat_cols), ('numerical_cols', num_encoder, num_cols)], verbose_feature_names_out=False)
    return transformer

def process_data(df):
    df = fix_data(df)
    transformer = encode_cat_num_data(df)
    churn_encoder = encode_churn_column(df)
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    #Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Save unencoded data for analyzing risk level
    joblib.dump(X_test, MODEL_PATH/'X_test_preencode.joblib')
    joblib.dump(y_test, MODEL_PATH/'y_test_preencode.joblib')

    #Fit and transform data
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    y_train = churn_encoder.fit_transform(y_train)
    y_test = churn_encoder.transform(y_test)

    #Save encoded data for analyzing risk level
    joblib.dump(X_test, MODEL_PATH/'X_test.joblib')
    joblib.dump(y_test, MODEL_PATH/'y_test.joblib')

    joblib.dump(transformer, '../model/transformer.joblib')

    return X_train, X_test, y_train, y_test
