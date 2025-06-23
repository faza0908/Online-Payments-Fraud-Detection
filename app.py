import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the trained XGBoost model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the features used for training
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest', 'type', 'isFlaggedFraud']

# Create the Streamlit application interface
st.title('Online Payment Fraud Detection')

st.write("""
Enter the transaction details below to predict if it is a fraudulent transaction.
""")

# Create input fields for each feature
input_data = {}
input_data['step'] = st.number_input('Step (Hour)', min_value=0)
input_data['amount'] = st.number_input('Amount', min_value=0.0)
input_data['oldbalanceOrg'] = st.number_input('Old Balance (Origin)', min_value=0.0)
input_data['newbalanceOrig'] = st.number_input('New Balance (Origin)', min_value=0.0)
input_data['oldbalanceDest'] = st.number_input('Old Balance (Destination)', min_value=0.0)
input_data['newbalanceDest'] = st.number_input('New Balance (Destination)', min_value=0.0)
input_data['balanceDiffOrig'] = st.number_input('Balance Difference (Origin)')
input_data['balanceDiffDest'] = st.number_input('Balance Difference (Destination)')
input_data['type'] = st.selectbox('Transaction Type', [0, 1, 2, 3, 4], format_func=lambda x: ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'][x])
input_data['isFlaggedFraud'] = st.selectbox('Is Flagged Fraud', [0, 1])


# Create a button to trigger the prediction
if st.button('Predict Fraud'):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_data], columns=features)

    # Make a prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    # Display the prediction result
    if prediction[0] == 1:
        st.error(f'Prediction: Fraudulent Transaction (Probability: {prediction_proba[0]:.4f})')
    else:
        st.success(f'Prediction: Not a Fraudulent Transaction (Probability: {prediction_proba[0]:.4f})')
