import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Visit with Us - Wellness Predictor", layout="centered")

st.title("Tourism Package Purchase Predictor.")
st.markdown("Enter customer details below to predict the likelihood of a Wellness Package purchase.")

# Dummy version string to force content change
APP_VERSION = "2026-03-29 18:13:57"

# Downloading the latest version of my model from the Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="pvinayv/tourism-package-predictor", filename="model.joblib")
    return joblib.load(model_path)

model = load_model()

# Creating the Input Form based on our dataset features
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Unmarried", "Divorced", "Married"])

    with col2:
        income = st.number_input("Monthly Income", min_value=0, value=25000)
        passport = st.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])
        pitch_duration = st.number_input("Duration of Pitch (min)", min_value=0, value=15)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        number_of_trips = st.number_input("Number of Trips Annually", min_value=0, value=3)

    with col3:
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=3)
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0, value=4)
        product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'Luxury', 'Premium'])
        pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
        own_car = st.selectbox("Owns a Car? (0=No, 1=Yes)", [0, 1])
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=1)

    submit = st.form_submit_button("Predict Conversion")

if submit:
    # Constructing a DataFrame for the model, ensuring dtypes match the training phase
    input_data = pd.DataFrame([{
        'Age': float(age),
        'TypeofContact': contact,
        'CityTier': city_tier,
        'DurationOfPitch': float(pitch_duration),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': int(number_of_person_visiting),
        'NumberOfFollowups': float(number_of_followups),
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': float(preferred_property_star),
        'MaritalStatus': marital,
        'NumberOfTrips': float(number_of_trips),
        'Passport': passport,
        'PitchSatisfactionScore': int(pitch_satisfaction_score),
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': float(number_of_children_visiting),
        'Designation': designation,
        'MonthlyIncome': float(income)
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
# Forced update at 2026-03-29 18:13:57