import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Visit with Us - Wellness Predictor", layout="centered")

st.title("Tourism Package Purchase Predictor")
st.markdown("Enter customer details below to predict the likelihood of a Wellness Package purchase.")

# Dummy version string to force content change
APP_VERSION = "2026-03-29 11:25:14"

# Downloading the latest version of my model from the Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="pvinayv/tourism-package-predictor", filename="model.joblib")
    return joblib.load(model_path)

model = load_model()

# Creating the Input Form based on our dataset features
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        income = st.number_input("Monthly Income", min_value=0, value=25000)
        passport = st.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])
        pitch_duration = st.number_input("Duration of Pitch (min)", min_value=0, value=15)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    submit = st.form_submit_button("Predict Conversion")

if submit:
    # Constructing a DataFrame for the model, ensuring dtypes match the training phase
    input_data = pd.DataFrame([{
        'Age': float(age),
        'TypeofContact': contact,
        'CityTier': city_tier,
        'Occupation': occupation,
        'Gender': gender,
        'MonthlyIncome': float(income),
        'Passport': passport,
        'DurationOfPitch': float(pitch_duration),
        'Designation': designation,
        'MaritalStatus': marital,
        # Defaulting other features to dataset means to keep the UI simple for the demo
        'NumberOfPersonVisiting': 3,
        'NumberOfFollowups': 4,
        'ProductPitched': 'Basic',
        'PreferredPropertyStar': 3,
        'NumberOfTrips': 3,
        'PitchSatisfactionScore': 3,
        'OwnCar': 1,
        'NumberOfChildrenVisiting': 1
    }])

    # Ensuring categorical types for XGBoost compatibility
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = input_data[col].astype('category')

    # Reorder columns to match the model's expected feature order
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"High Conversion Potential! (Probability: {prob:.2f})")
    else:
        st.warning(f"Low Conversion Potential. (Probability: {prob:.2f})")
# Forced update at 2026-03-29 11:25:14