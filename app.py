import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Health Stratify",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("stacking_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_model_and_scaler()

except Exception as e:
    st.error("Could not load the trained model or scaler.")
    st.exception(e)
    st.stop()


# =========================================================
# GET EXACT FEATURES USED DURING TRAINING
# =========================================================

if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    st.error(
        "The saved scaler does not contain the original "
        "feature names used during training."
    )
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🩺 Health Stratify")

st.markdown(
    """
    ### AI-Powered Health Risk Classification

    Enter the patient's health and lifestyle information.
    The trained machine learning model will generate a
    classification prediction.

    **Please enter values within the ranges shown below.**
    """
)

st.info(
    "⚠️ This application is for educational and predictive purposes. "
    "It is not a medical diagnosis."
)

st.divider()


# =========================================================
# STORE USER INPUT
# =========================================================

input_values = {}


# =========================================================
# BASIC HEALTH INFORMATION
# =========================================================

st.header("👤 Basic Health Information")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# AGE
# ---------------------------------------------------------

if "Age" in feature_names:

    with col1:

        input_values["Age"] = st.number_input(
            "🎂 Age (years)",
            min_value=18,
            max_value=100,
            value=25,
            step=1,
            help="Enter age between 18 and 100 years."
        )


# ---------------------------------------------------------
# BMI
# ---------------------------------------------------------

if "BMI" in feature_names:

    with col2:

        input_values["BMI"] = st.number_input(
            "⚖️ BMI (kg/m²)",
            min_value=15.0,
            max_value=45.0,
            value=22.5,
            step=0.1,
            help="Enter BMI between 15 and 45 kg/m²."
        )


# ---------------------------------------------------------
# BLOOD PRESSURE
# ---------------------------------------------------------

blood_pressure_feature = None

if "Blood_Pressure" in feature_names:
    blood_pressure_feature = "Blood_Pressure"

elif "Blood Pressure" in feature_names:
    blood_pressure_feature = "Blood Pressure"


if blood_pressure_feature:

    with col1:

        input_values[blood_pressure_feature] = st.number_input(
            "🩸 Blood Pressure (mmHg)",
            min_value=80.0,
            max_value=200.0,
            value=120.0,
            step=1.0,
            help="Enter systolic blood pressure between 80 and 200 mmHg."
        )


# ---------------------------------------------------------
# HEART RATE
# ---------------------------------------------------------

heart_rate_feature = None

if "Heart_Rate" in feature_names:
    heart_rate_feature = "Heart_Rate"

elif "Heart Rate" in feature_names:
    heart_rate_feature = "Heart Rate"


if heart_rate_feature:

    with col2:

        input_values[heart_rate_feature] = st.number_input(
            "❤️ Resting Heart Rate (BPM)",
            min_value=40.0,
            max_value=150.0,
            value=72.0,
            step=1.0,
            help="Enter resting heart rate between 40 and 150 BPM."
        )


# ---------------------------------------------------------
# GLUCOSE
# ---------------------------------------------------------

glucose_feature = None

if "Glucose_Level" in feature_names:
    glucose_feature = "Glucose_Level"

elif "Glucose Level" in feature_names:
    glucose_feature = "Glucose Level"


if glucose_feature:

    with col1:

        input_values[glucose_feature] = st.number_input(
            "🧪 Glucose Level (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=95.0,
            step=1.0,
            help="Enter blood glucose level between 50 and 300 mg/dL."
        )


# ---------------------------------------------------------
# CHOLESTEROL
# ---------------------------------------------------------

if "Cholesterol" in feature_names:

    with col2:

        input_values["Cholesterol"] = st.number_input(
            "🧪 Cholesterol (mg/dL)",
            min_value=100.0,
            max_value=350.0,
            value=180.0,
            step=1.0,
            help="Enter cholesterol level between 100 and 350 mg/dL."
        )


# =========================================================
# DAILY LIFESTYLE
# =========================================================

st.divider()

st.header("🌱 Daily Lifestyle")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# SLEEP
# ---------------------------------------------------------

sleep_feature = None

if "Sleep_Hours" in feature_names:
    sleep_feature = "Sleep_Hours"

elif "Sleep Hours" in feature_names:
    sleep_feature = "Sleep Hours"


if sleep_feature:

    with col1:

        input_values[sleep_feature] = st.number_input(
            "😴 Sleep per Day (hours)",
            min_value=3.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            help="Enter average sleep between 3 and 12 hours per day."
        )


# ---------------------------------------------------------
# EXERCISE
# ---------------------------------------------------------

exercise_feature = None

if "Exercise_Hours" in feature_names:
    exercise_feature = "Exercise_Hours"

elif "Exercise Hours" in feature_names:
    exercise_feature = "Exercise Hours"


if exercise_feature:

    with col2:

        input_values[exercise_feature] = st.number_input(
            "🏃 Exercise per Day (hours)",
            min_value=0.0,
            max_value=8.0,
            value=1.0,
            step=0.5,
            help="Enter average exercise between 0 and 8 hours per day."
        )


# ---------------------------------------------------------
# WATER
# ---------------------------------------------------------

water_feature = None

if "Water_Intake" in feature_names:
    water_feature = "Water_Intake"

elif "Water Intake" in feature_names:
    water_feature = "Water Intake"


if water_feature:

    with col1:

        input_values[water_feature] = st.number_input(
            "💧 Water Intake (litres/day)",
            min_value=0.5,
            max_value=6.0,
            value=2.5,
            step=0.1,
            help="Enter average water intake between 0.5 and 6 litres per day."
        )


# ---------------------------------------------------------
# STRESS
# ---------------------------------------------------------

stress_feature = None

if "Stress_Level" in feature_names:
    stress_feature = "Stress_Level"

elif "Stress Level" in feature_names:
    stress_feature = "Stress Level"


if stress_feature:

    with col2:

        input_values[stress_feature] = st.slider(
            "🧠 Stress Level",
            min_value=1,
            max_value=10,
            value=4,
            help="1 = Very Low Stress, 10 = Very High Stress."
        )


# =========================================================
# HABITS
# =========================================================

st.divider()

st.header("🚭 Habits")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# SMOKING
# ---------------------------------------------------------

if "Smoking" in feature_names:

    with col1:

        smoking = st.radio(
            "🚭 Do you smoke?",
            ["No", "Yes"],
            horizontal=True
        )

        input_values["Smoking"] = (
            1 if smoking == "Yes" else 0
        )


# ---------------------------------------------------------
# ALCOHOL
# ---------------------------------------------------------

if "Alcohol" in feature_names:

    with col2:

        alcohol = st.radio(
            "🍷 Do you consume alcohol?",
            ["No", "Yes"],
            horizontal=True
        )

        input_values["Alcohol"] = (
            1 if alcohol == "Yes" else 0
        )


# =========================================================
# PERSONAL & MEDICAL INFORMATION
# =========================================================

st.divider()

st.header("🏥 Personal & Medical Information")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# PHYSICAL ACTIVITY
# ---------------------------------------------------------

physical_feature = None

if "Physicalactivity" in feature_names:
    physical_feature = "Physicalactivity"

elif "Physical_Activity" in feature_names:
    physical_feature = "Physical_Activity"


if physical_feature:

    with col1:

        physical_activity = st.selectbox(
            "🏃 Physical Activity Level",
            [
                "Low",
                "Moderate",
                "High"
            ],
            help="Choose the patient's usual physical activity level."
        )

        # IMPORTANT:
        # This mapping must match the encoding used during training.
        activity_mapping = {
            "Low": 0,
            "Moderate": 1,
            "High": 2
        }

        input_values[physical_feature] = (
            activity_mapping[physical_activity]
        )


# ---------------------------------------------------------
# MENTAL HEALTH
# ---------------------------------------------------------

mental_feature = None

if "Mentalhealth" in feature_names:
    mental_feature = "Mentalhealth"

elif "Mental_Health" in feature_names:
    mental_feature = "Mental_Health"


if mental_feature:

    with col2:

        mental_health = st.selectbox(
            "🧠 Mental Health",
            [
                "Poor",
                "Average",
                "Good"
            ],
            help="Choose the option that best describes the patient's mental health."
        )

        # IMPORTANT:
        # This mapping must match the encoding used during training.
        mental_mapping = {
            "Poor": 0,
            "Average": 1,
            "Good": 2
        }

        input_values[mental_feature] = (
            mental_mapping[mental_health]
        )


# ---------------------------------------------------------
# MEDICAL HISTORY
# ---------------------------------------------------------

medical_feature = None

if "Medicalhistory" in feature_names:
    medical_feature = "Medicalhistory"

elif "Medical_History" in feature_names:
    medical_feature = "Medical_History"


if medical_feature:

    with col1:

        medical_history = st.radio(
            "🏥 Previous Medical Conditions?",
            ["No", "Yes"],
            horizontal=True,
            help="Select Yes if the patient has a significant previous medical condition."
        )

        input_values[medical_feature] = (
            1 if medical_history == "Yes" else 0
        )


# ---------------------------------------------------------
# ALLERGIES
# ---------------------------------------------------------

if "Allergies" in feature_names:

    with col2:

        allergies = st.radio(
            "🌿 Known Allergies?",
            ["No", "Yes"],
            horizontal=True,
            help="Select Yes if the patient has known allergies."
        )

        input_values["Allergies"] = (
            1 if allergies == "Yes" else 0
        )


# =========================================================
# BLOOD GROUP
# =========================================================

blood_group_features = [
    feature
    for feature in feature_names
    if feature.startswith("Blood_Group_")
]


if blood_group_features:

    st.divider()

    st.header("🩸 Blood Group")

    blood_groups = []

    for feature in blood_group_features:

        group = feature.replace(
            "Blood_Group_",
            ""
        )

        blood_groups.append(group)


    blood_group = st.selectbox(
        "Select Blood Group",
        blood_groups,
        help="Select the patient's blood group."
    )


    for feature in blood_group_features:

        group = feature.replace(
            "Blood_Group_",
            ""
        )

        input_values[feature] = (
            1 if group == blood_group else 0
        )


# =========================================================
# OTHER ONE-HOT FEATURES
# =========================================================

for feature in feature_names:

    if feature not in input_values:

        # Initialize one-hot encoded columns
        # that are not directly selected by the user.

        if "_" in feature:

            input_values[feature] = 0


# =========================================================
# FINAL PREDICTION SECTION
# =========================================================

st.divider()

st.header("🔍 Get Health Risk Prediction")

st.write(
    "Review your information above and click the button "
    "to generate a prediction."
)


predict_button = st.button(
    "🔍 Predict Health Risk",
    use_container_width=True,
    type="primary"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [input_values]
        )


        # -------------------------------------------------
        # EXACT SAME FEATURE ORDER AS TRAINING
        # -------------------------------------------------

        input_data = input_data.reindex(
            columns=feature_names,
            fill_value=0
        )


        # -------------------------------------------------
        # CONVERT TO NUMERIC
        # -------------------------------------------------

        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )


        input_data = input_data.fillna(0)


        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        scaled_input = scaler.transform(
            input_data
        )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        prediction = model.predict(
            scaled_input
        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")


        if prediction[0] == 1:

            st.error(
                "⚠️ Predicted Classification: Class 1"
            )

        else:

            st.success(
                "✅ Predicted Classification: Class 0"
            )


        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_input
            )[0]

            confidence = (
                max(probabilities) * 100
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


        # -------------------------------------------------
        # SHOW MODEL INPUT
        # -------------------------------------------------

        with st.expander(
            "🔎 View data sent to the model"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction could not be generated."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Health Stratify • Machine Learning Classification Project"
)

st.caption(
    "⚠️ This prediction is for educational purposes and "
    "should not be used as a medical diagnosis."
)