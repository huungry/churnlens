import streamlit as st
import pandas as pd
import joblib
import app_utils
from app_predict_process import fix_data, encode, default_mode_predict, get_churn_risk_label

st.set_page_config(page_title='ChurnLens', layout='wide', page_icon='🔎')
st.markdown(
    "<h1 style='text-align: center;'>ChurnLens</h1>",
    unsafe_allow_html=True
)

st.divider(width='stretch')

with st.container(border=True):
    st.subheader('ⓘ About the model')
    st.write('A machine learning model engined by XGBoost that predicts whether a customer will churn or not, based on various features like tenure, monthly charges, total charges, etc.')
    st.markdown("The model is trained on Telco Customer Churn dataset. Check out this [dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) for more details.")
    st.caption('Note: This model is not 100% accurate. Its predictions are based on patterns learned from the training data.')

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('🔍 Single Customer Prediction')
st.write('Try single prediction by adjusting features below:')


with st.container(border=True, width='stretch'):
    st.markdown("<h5 style='text-align: center;'>Customer Information</h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Female', 'Male'])
        senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        phone_ser = st.selectbox('Phone Service', ['No', 'Yes'])
        phone_options = (["No phone service"] if phone_ser == "No" else ["No", "Yes"])
        multi_lines = st.selectbox('Multiple Lines', phone_options)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    with col2:
        internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        internet_options = (["No internet service"] if internet == "No" else ["No", "Yes"])
        dev_protec = st.selectbox('Device Protection', internet_options)
        tech_sup = st.selectbox('Tech Support', internet_options)
        onl_back = st.selectbox('Online Backup', internet_options)
        stream_mov = st.selectbox('Streaming Movies', internet_options)
        online_sec = st.selectbox('Online Security', internet_options)
        stream_tv = st.selectbox('Streaming TV', internet_options)
    with col3:
        paper_bill = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        tenure = st.number_input('Tenure', min_value=0, value=0, step=1, help='How many full months has the customer been with the company?')
        monthly_charge = st.number_input('Monthly Charge', min_value=0.0, value=0.0, step=10.0, help='The amount charged to the customer monthly.')
        total_charge = st.number_input('Total Charge', min_value=0.0, value=0.0, step=10.0, help='The total amount charged to the customer.')
        st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.button('Predict')
    st.caption('Note: The prediction will appear below.')

st.markdown("<br>", unsafe_allow_html=True)
if submitted:
    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_ser,
        'MultipleLines': multi_lines,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': onl_back,
        'DeviceProtection': dev_protec,
        'TechSupport': tech_sup,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_mov,
        'Contract': contract,
        'PaperlessBilling': paper_bill,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charge,
        'TotalCharges': total_charge
    }
    data = pd.DataFrame(data, index=[0])
    encode_data = encode(data)
    pred_prob, pred = default_mode_predict(encode_data)
    pred_txt = 'No Churn' if pred == 0 else 'Churn'
    label = get_churn_risk_label(pred_prob)

    with st.container(border=True, width='content'):
        st.markdown(f"#### Model Prediction : {pred_txt}")
        app_utils.risk(label)
        st.metric('Estimated Churning Probability: ', f'{pred_prob[0]:.2%}')
        st.write('The model uses a 50% cutoff: scores of 50% or higher are classified as Churn, and scores below 50% as No Churn.')


st.divider(width='stretch')
st.subheader('🔍 Batch Predictions')
st.write('Try uploading a CSV to get churn predictions for multiple customers at once.')
st.warning(title='Please follow these instructions before uploading:', body="""
            1. Must use the same wording, spelling, and formats shown in Single Customer Prediction for all feature values.
            2. Must include all required feature columns. ***Use the sample CSV as a guide for column names***.
            3. When the Internet Service is No, then all related features must be No internet service.
            4. When Phone Service is No, then Multiple Lines must be No phone service.
            5. SeniorCitizen can be Yes/No or Yes as 1 and No as 0.
            6. Numeric fields must contain numbers, with no blanks or currency symbols.
            """)
st.write('Sample CSV:')
st.dataframe(joblib.load('model/sample.joblib'))
st.markdown("<br>", unsafe_allow_html=True)
user_uploaded_file = st.file_uploader('Upload:', type='csv', width=400)

if user_uploaded_file is not None:
    df = pd.read_csv(user_uploaded_file)
    app_utils.checkup(df)
    df = fix_data(df)
    encode_df = encode(df)
    pred_prob_batch, pred_batch = default_mode_predict(encode_df)
    risk = [get_churn_risk_label(float(i)) for i in pred_prob_batch]
    pred_batch = ['No' if i == 0 else 'Yes' for i in pred_batch]
    pred_prob_batch = [f'{i:.2%}' for i in pred_prob_batch]
    risk = pd.DataFrame(risk, columns=['Risk Level'])
    pred_batch = pd.DataFrame(pred_batch, columns=['Churn'])
    pred_prob_batch = pd.DataFrame(pred_prob_batch, columns=['Churn Probability'])
    result = pd.concat([df, pred_batch, pred_prob_batch, risk], axis=1)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Predictions Preview</h5>", unsafe_allow_html=True) 
    st.dataframe(result.head(10), width='stretch')
    st.download_button('Download full predictions', data=result.to_csv(), file_name='churn_predictions.csv', mime='text/csv')

