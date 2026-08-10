import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Health Stratify",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_model_and_scaler():

    model = joblib.load("stacking_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


try:

    model, scaler = load_model_and_scaler()

except Exception as e:

    st.error("Unable to load the trained model or scaler.")
    st.exception(e)
    st.stop()


# =========================================================
# GET EXACT TRAINING FEATURES
# =========================================================

if hasattr(scaler, "feature_names_in_"):

    feature_names = list(
        scaler.feature_names_in_
    )

else:

    st.error(
        "The saved scaler does not contain the feature names "
        "used during training."
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🩺 Health Stratify")

st.markdown(
    """
    ### Understand your health profile with Machine Learning

    Enter the information below and Health Stratify will analyze
    the provided health and lifestyle factors using the trained
    machine learning model.
    """
)

st.info(
    "⚠️ This is a machine-learning project for educational and "
    "predictive purposes. It is not a medical diagnosis."
)


# =========================================================
# INPUT STORAGE
# =========================================================

input_values = {}


# =========================================================
# BASIC INFORMATION
# =========================================================

st.header("👤 Basic Information")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# AGE
# ---------------------------------------------------------

if "Age" in feature_names:

    with col1:

        input_values["Age"] = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=100,
            value=25,
            step=1
        )

        st.caption("Enter your age in years.")


# ---------------------------------------------------------
# BMI
# ---------------------------------------------------------

if "BMI" in feature_names:

    with col2:

        input_values["BMI"] = st.number_input(
            "⚖️ BMI",
            min_value=15.0,
            max_value=45.0,
            value=22.5,
            step=0.1
        )

        st.caption("Enter your BMI value.")


# =========================================================
# BODY / VITAL INFORMATION
# =========================================================

st.header("❤️ Body & Vital Information")

col1, col2 = st.columns(2)


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
            "🩸 Blood Pressure",
            min_value=80.0,
            max_value=200.0,
            value=120.0,
            step=1.0
        )

        st.caption("Enter systolic blood pressure.")


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
            "❤️ Resting Heart Rate",
            min_value=40.0,
            max_value=150.0,
            value=72.0,
            step=1.0
        )

        st.caption("Enter resting heart rate in BPM.")


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
            "🧪 Glucose Level",
            min_value=50.0,
            max_value=300.0,
            value=95.0,
            step=1.0
        )

        st.caption("Enter blood glucose level.")


# ---------------------------------------------------------
# CHOLESTEROL
# ---------------------------------------------------------

if "Cholesterol" in feature_names:

    with col2:

        input_values["Cholesterol"] = st.number_input(
            "🧪 Cholesterol",
            min_value=100.0,
            max_value=350.0,
            value=180.0,
            step=1.0
        )

        st.caption("Enter cholesterol level.")


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
            "😴 Sleep",
            min_value=3.0,
            max_value=12.0,
            value=7.0,
            step=0.5
        )

        st.caption("Average hours of sleep per day.")


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
            "🏃 Exercise",
            min_value=0.0,
            max_value=8.0,
            value=1.0,
            step=0.5
        )

        st.caption("Average hours of exercise per day.")


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
            "💧 Water Intake",
            min_value=0.5,
            max_value=6.0,
            value=2.5,
            step=0.1
        )

        st.caption("Average water intake per day.")


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
            value=4
        )

        st.caption(
            "1 = very low stress  •  10 = very high stress"
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
# PERSONAL / MEDICAL INFORMATION
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
            "🏃 Physical Activity",
            [
                "Low",
                "Moderate",
                "High"
            ]
        )

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
            ]
        )

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
            horizontal=True
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
            horizontal=True
        )

        input_values["Allergies"] = (
            1 if allergies == "Yes" else 0
        )


# =========================================================
# BLOOD GROUP
# =========================================================

# =========================================================
# BLOOD GROUP
# =========================================================

blood_group_features = [
    f for f in feature_names
    if f.startswith("Blood_Group_")
]

if blood_group_features:

    st.divider()

    st.header("🩸 Blood Group")

    # Include common blood groups
    available_groups = ["A", "B", "AB", "O"]

    # Keep only groups actually represented in the model
    model_groups = [
        f.replace("Blood_Group_", "")
        for f in blood_group_features
    ]

    # Combine model groups with common groups
    blood_groups = [
        group for group in available_groups
        if group in model_groups
    ]

    blood_group = st.selectbox(
        "Select Blood Group",
        blood_groups,
        help="Select the patient's blood group."
    )

    # Set one-hot encoded blood-group columns
    for feature in blood_group_features:

        group = feature.replace(
            "Blood_Group_",
            ""
        )

        input_values[feature] = (
            1 if group == blood_group else 0
        )

# =========================================================
# INITIALIZE ANY REMAINING ONE-HOT FEATURES
# =========================================================

for feature in feature_names:

    if feature not in input_values:

        if "_" in feature:

            input_values[feature] = 0


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

st.header("🔍 Check Your Result")

st.write(
    "Once all information is entered, click the button "
    "below to analyze the profile."
)

predict_button = st.button(
    "🔍 Analyze My Health Profile",
    use_container_width=True,
    type="primary"
)


# =========================================================
# HUMAN-FRIENDLY ANALYSIS
# =========================================================

def generate_feedback(values):

    feedback = []

    positives = []

    attention = []


    # -----------------------------------------------------
    # SLEEP
    # -----------------------------------------------------

    if sleep_feature in values:

        sleep = values[sleep_feature]

        if 7 <= sleep <= 9:

            positives.append(
                f"Your reported sleep is {sleep:g} hours per day."
            )

        elif sleep < 7:

            attention.append(
                f"You reported {sleep:g} hours of sleep per day. "
                "Consider maintaining a consistent sleep routine "
                "and giving adequate time for rest."
            )

        else:

            attention.append(
                f"You reported {sleep:g} hours of sleep per day. "
                "Try to keep your sleep schedule consistent."
            )


    # -----------------------------------------------------
    # EXERCISE
    # -----------------------------------------------------

    if exercise_feature in values:

        exercise = values[exercise_feature]

        if exercise >= 0.5:

            positives.append(
                f"You reported about {exercise:g} hours of exercise per day."
            )

        else:

            attention.append(
                "Your reported exercise level is quite low. "
                "Adding regular movement to your routine may be useful."
            )


    # -----------------------------------------------------
    # WATER
    # -----------------------------------------------------

    if water_feature in values:

        water = values[water_feature]

        if water >= 2:

            positives.append(
                f"You reported around {water:g} litres of water per day."
            )

        else:

            attention.append(
                f"You reported around {water:g} litres of water per day. "
                "Try to maintain regular hydration throughout the day."
            )


    # -----------------------------------------------------
    # STRESS
    # -----------------------------------------------------

    if stress_feature in values:

        stress = values[stress_feature]

        if stress <= 4:

            positives.append(
                f"Your reported stress level is {stress}/10."
            )

        elif stress <= 7:

            attention.append(
                f"Your reported stress level is {stress}/10. "
                "You may benefit from regular breaks, physical activity, "
                "relaxation, or time away from screens."
            )

        else:

            attention.append(
                f"Your reported stress level is {stress}/10, "
                "which is high on the scale you provided. "
                "Consider giving more attention to stress management."
            )


    # -----------------------------------------------------
    # BMI
    # -----------------------------------------------------

    if "BMI" in values:

        bmi = values["BMI"]

        feedback.append(
            f"Your entered BMI is {bmi:.1f}. "
            "BMI is one factor among many and should be interpreted "
            "in context rather than by itself."
        )


    # -----------------------------------------------------
    # SMOKING
    # -----------------------------------------------------

    if "Smoking" in values:

        if values["Smoking"] == 0:

            positives.append(
                "You reported that you do not smoke."
            )

        else:

            attention.append(
                "You reported smoking. Reducing or avoiding tobacco "
                "use can be an important lifestyle consideration."
            )


    # -----------------------------------------------------
    # ALCOHOL
    # -----------------------------------------------------

    if "Alcohol" in values:

        if values["Alcohol"] == 0:

            positives.append(
                "You reported no alcohol consumption."
            )

        else:

            attention.append(
                "You reported alcohol consumption. "
                "Keeping consumption within your personal health goals "
                "is worth considering."
            )


    return positives, attention


# =========================================================
# RUN MODEL
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [input_values]
        )


        # -------------------------------------------------
        # EXACT TRAINING COLUMN ORDER
        # -------------------------------------------------

        input_data = input_data.reindex(
            columns=feature_names,
            fill_value=0
        )


        # -------------------------------------------------
        # NUMERIC CONVERSION
        # -------------------------------------------------

        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )


        input_data = input_data.fillna(0)


        # -------------------------------------------------
        # SCALE
        # -------------------------------------------------

        scaled_input = scaler.transform(
            input_data
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            scaled_input
        )


        predicted_class = prediction[0]


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.header("📊 Your Result")


        # -------------------------------------------------
        # CLASS RESULT
        # -------------------------------------------------

        if predicted_class == 1:

            st.warning(
                "⚠️ The model classified your profile as **Class 1**."
            )

        else:

            st.success(
                "✅ The model classified your profile as **Class 0**."
            )


        st.caption(
            "This classification is the output of the trained "
            "machine learning model. It is not a medical diagnosis."
        )


        # =================================================
        # CONFIDENCE
        # =================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_input
            )[0]

            confidence = (
                float(max(probabilities)) * 100
            )

            st.metric(
                "🤖 Model Confidence",
                f"{confidence:.2f}%"
            )


        # =================================================
        # SIMPLE EXPLANATION
        # =================================================

        st.subheader("💬 What your information tells us")

        st.write(
            "Here are some simple observations based on the "
            "information you entered. These observations are "
            "separate from the model's classification."
        )


        positives, attention = generate_feedback(
            input_values
        )


        # -------------------------------------------------
        # POSITIVE OBSERVATIONS
        # -------------------------------------------------

        if positives:

            st.markdown("### ✅ Positive points")

            for item in positives:

                st.write(
                    f"• {item}"
                )


        # -------------------------------------------------
        # AREAS TO PAY ATTENTION TO
        # -------------------------------------------------

        if attention:

            st.markdown("### 💡 Things you may want to pay attention to")

            for item in attention:

                st.write(
                    f"• {item}"
                )

        else:

            st.success(
                "Your entered lifestyle information does not "
                "highlight any of the basic areas checked by this app."
            )


        # =================================================
        # PERSONALIZED SUMMARY
        # =================================================

        st.subheader("📝 Simple Summary")

        if predicted_class == 1:

            st.write(
                "The machine learning model placed this profile "
                "in Class 1. This result should be viewed together "
                "with the individual factors above rather than "
                "as a diagnosis."
            )

        else:

            st.write(
                "The machine learning model placed this profile "
                "in Class 0. This is the model's classification "
                "based on the information provided."
            )


        # =================================================
        # GENERAL NEXT STEPS
        # =================================================

        st.subheader("🌱 General Next Steps")

        st.write(
            """
            • Keep track of your health information regularly.

            • Maintain a consistent sleep and activity routine.

            • Pay attention to stress and lifestyle habits.

            • If any health measurement is concerning or you have
              symptoms, discuss it with a qualified healthcare
              professional.
            """
        )


        # =================================================
        # TECHNICAL DETAILS
        # =================================================

        with st.expander(
            "🔎 View technical model details"
        ):

            st.write(
                "The following processed values were sent "
                "to the trained machine learning model:"
            )

            st.dataframe(
                input_data,
                use_container_width=True
            )

            st.write(
                f"Predicted class: {predicted_class}"
            )


    except Exception as e:

        st.error(
            "❌ The prediction could not be generated."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Health Stratify • End-to-End Machine Learning Project"
)

st.caption(
    "⚠️ For educational and predictive purposes only. "
    "Not a substitute for professional medical advice."
)
