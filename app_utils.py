import streamlit as st

def risk(label):
    if label == 'High Risk':
        st.markdown(f"##### Risk level : 🔴 :red[{label}]")
    elif label == 'Elevated Risk':
        st.markdown(f"##### Risk level : 🟠 :orange[{label}]")
    elif label == 'Moderate Risk':
        st.markdown(f"##### Risk level : 🟡 :yellow[{label}]")
    else:
        st.markdown(f"##### Risk level : 🟢 :green[{label}]")

def checkup(df):
    cols = ['customerID', 'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    if df.empty:
        st.error('No data to display')
        st.stop()

    missingcol = [col for col in cols if col not in df.columns]
    if missingcol:
        st.error(f'Missing columns: {missingcol}')
        st.stop()

    missingvals = [col for col in cols if df[col].isna().any()]
    if missingvals:
        st.error(f'Columns with missing values: {missingvals}')
        st.stop()
