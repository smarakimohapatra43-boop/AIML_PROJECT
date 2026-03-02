import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config for aesthetics
st.set_page_config(page_title="Marketing Campaign Prediction", page_icon="🏦", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .prediction-yes {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .prediction-no {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Load the model, features, and original data for dropdown options
@st.cache_resource
def load_assets():
    model = joblib.load('c:/Sakshi/random_forest_model.pkl')
    model_features = joblib.load('c:/Sakshi/model_features.pkl')
    # Load original dataset simply to get the unique categories for the UI
    df_raw = pd.read_csv('c:/Sakshi/bank.csv')
    return model, model_features, df_raw

try:
    model, model_features, df_raw = load_assets()
except Exception as e:
    st.error(f"Error loading model or data. Please ensure 'train_model.py' has been run. Details: {e}")
    st.stop()

# Header
st.title("🏦 Marketing Campaign Prediction")
st.markdown("Enter customer details below to predict whether they will subscribe to a bank term deposit.")

st.markdown("---")

# Layout: Split into columns
col1, col2, col3 = st.columns(3)

# Define inputs
with col1:
    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    job = st.selectbox("Job", options=df_raw['job'].unique())
    marital = st.selectbox("Marital Status", options=df_raw['marital'].unique())
    education = st.selectbox("Education", options=df_raw['education'].unique())

with col2:
    st.subheader("Financial Status")
    balance = st.number_input("Yearly Balance (€)", value=1000)
    default = st.selectbox("Has Credit in Default?", options=df_raw['default'].unique())
    housing = st.selectbox("Has Housing Loan?", options=df_raw['housing'].unique())
    loan = st.selectbox("Has Personal Loan?", options=df_raw['loan'].unique())

with col3:
    st.subheader("Campaign Interaction")
    contact = st.selectbox("Contact Communication Type", options=df_raw['contact'].unique())
    month = st.selectbox("Last Contact Month", options=df_raw['month'].unique())
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=250)
    campaign = st.number_input("Number of Contacts during this Campaign", min_value=1, value=1)
    pdays = st.number_input("Passed Days after previous campaign (-1 if not previously contacted)", value=-1)
    previous = st.number_input("Number of Contacts before this Campaign", min_value=0, value=0)
    poutcome = st.selectbox("Outcome of Previous Campaign", options=df_raw['poutcome'].unique())

st.markdown("---")

# Prediction logic
if st.button("Predict Term Deposit Subscription"):
    
    # We create a dictionary containing exactly the features the model expects, initialized to 0
    input_dict = {feature: 0 for feature in model_features}
    
    # Set the numerical variables correctly
    input_dict['age'] = age
    input_dict['balance'] = balance
    input_dict['day'] = day
    input_dict['duration'] = duration
    input_dict['campaign'] = campaign
    input_dict['pdays'] = pdays
    input_dict['previous'] = previous
    
    # For categorical variables, set the appropriate one-hot encoded columns to 1 (True)
    # The get_dummies format is "VariableName_Value" (e.g. "job_admin.")
    categorical_vars = {
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'poutcome': poutcome
    }
    
    for var, value in categorical_vars.items():
        # Recreate the column name that get_dummies would have generated
        col_name = f"{var}_{value}"
        # If this category was present during training and wasn't dropped as first, set it to 1
        if col_name in input_dict:
            input_dict[col_name] = 1
            
    # Convert exactly to correct DataFrame layout
    final_input = pd.DataFrame([input_dict])
    
    # Ensure correct column order
    final_input = final_input[model_features]
    
    # Predict
    prediction = model.predict(final_input)[0]
    probabilities = model.predict_proba(final_input)[0]
    
    st.markdown("### Prediction Result")
    
    if prediction == 1:
        st.markdown(
            f"""<div class='prediction-box prediction-yes'>
            <h2>🟢 YES!</h2>
            <p>The model predicts that this customer <b>WILL</b> subscribe to a term deposit.</p>
            <p>Confidence: <b>{probabilities[1]*100:.1f}%</b></p>
            </div>""", 
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            f"""<div class='prediction-box prediction-no'>
            <h2>🔴 NO</h2>
            <p>The model predicts that this customer will <b>NOT</b> subscribe to a term deposit.</p>
             <p>Confidence: <b>{probabilities[0]*100:.1f}%</b></p>
            </div>""", 
            unsafe_allow_html=True
        )
