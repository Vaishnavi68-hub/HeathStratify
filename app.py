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

    st.error("Unable to load the trained model or scaler.")
    st.exception(e)
    st.stop()


# =========================================================
# GET EXACT FEATURES USED DURING TRAINING
# =========================================================

if hasattr(scaler, "feature_names_in_"):

    feature_names = list(scaler.feature_names_in_)

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

    Enter your health, lifestyle, and personal information below.
    Health Stratify will analyze the information using a trained
    machine learning model and provide an easy-to-understand result.
    """
)

st.info(
    "⚠️ This application is designed for educational and "
    "predictive purposes. It is not a medical diagnosis."
)

st.divider()


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
            step=1,
            help="Enter age between 18 and 100 years."
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
            step=0.1,
            help="Enter BMI between 15 and 45."
        )

        st.caption("Body Mass Index (BMI).")


# =========================================================
# BODY AND VITAL INFORMATION
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
            step=1.0,
            help="Enter systolic blood pressure between 80 and 200 mmHg."
        )

        st.caption("Systolic blood pressure in mmHg.")


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
            step=1.0,
            help="Enter resting heart rate between 40 and 150 BPM."
        )

        st.caption("Resting heart rate in BPM.")


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
            step=1.0,
            help="Enter glucose level between 50 and 300 mg/dL."
        )

        st.caption("Blood glucose level in mg/dL.")


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
            step=1.0,
            help="Enter cholesterol between 100 and 350 mg/dL."
        )

        st.caption("Cholesterol level in mg/dL.")


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
            step=0.5,
            help="Enter average sleep between 3 and 12 hours per day."
        )

        st.caption("Average sleep per day in hours.")


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
            step=0.5,
            help="Enter average exercise between 0 and 8 hours per day."
        )

        st.caption("Average exercise per day in hours.")


# ---------------------------------------------------------
# WATER INTAKE
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
            step=0.1,
            help="Enter average water intake between 0.5 and 6 litres per day."
        )

        st.caption("Average water intake per day in litres.")


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

        st.caption(
            "Choose a value from 1 (very low) to 10 (very high)."
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
# PERSONAL AND MEDICAL INFORMATION
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
            ],
            help="Choose the level that best describes your usual activity."
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
            ],
            help="Choose the option that best describes your current mental health."
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
# DIET
# =========================================================

diet_feature = None

if "Diet" in feature_names:

    diet_feature = "Diet"

elif "diet" in feature_names:

    diet_feature = "diet"


if diet_feature:

    st.divider()

    st.header("🥗 Diet")

    diet = st.selectbox(
        "How would you describe your usual diet?",
        [
            "Vegetarian",
            "Non-Vegetarian",
            "Mixed",
            "Other"
        ]
    )

    # IMPORTANT:
    # This assumes the training encoding follows this mapping.
    # Change this mapping if your training preprocessing used
    # different numerical values.

    diet_mapping = {
        "Vegetarian": 0,
        "Non-Vegetarian": 1,
        "Mixed": 2,
        "Other": 3
    }

    input_values[diet_feature] = diet_mapping[diet]


# =========================================================
# BLOOD GROUP
# =========================================================

st.divider()

st.header("🩸 Blood Group")

blood_group = st.selectbox(
    "Select your blood group",
    [
        "A",
        "B",
        "AB",
        "O"
    ],
    help="Select your blood group."
)

# Find blood-group columns actually used during training
blood_group_features = [
    f for f in feature_names
    if f.startswith("Blood_Group_")
]

# Set all known blood-group model features to 0
for feature in blood_group_features:

    input_values[feature] = 0


# Set the selected blood group to 1
selected_feature = f"Blood_Group_{blood_group}"

if selected_feature in feature_names:

    input_values[selected_feature] = 1

else:

    # The UI allows A/B/AB/O, but the trained model
    # does not contain this particular blood-group feature.

    st.warning(
        f"Blood group {blood_group} is available for selection, "
        "but this blood group was not present as a feature in "
        "the trained model. It will therefore not affect this prediction."
    )


# =========================================================
# INITIALIZE REMAINING FEATURES
# =========================================================

for feature in feature_names:

    if feature not in input_values:

        input_values[feature] = 0


# =========================================================
# PREDICTION SECTION
# =========================================================

st.divider()

st.header("🔍 Check Your Result")

st.write(
    "Review your information and click the button below "
    "to analyze your profile."
)


predict_button = st.button(
    "🔍 Analyze My Health Profile",
    use_container_width=True,
    type="primary"
)


# =========================================================
# HUMAN-FRIENDLY FEEDBACK FUNCTION
# =========================================================

def generate_feedback(values):

    positives = []
    attention = []


    # -----------------------------------------------------
    # SLEEP
    # -----------------------------------------------------

    if sleep_feature in values:

        sleep = values[sleep_feature]

        if 7 <= sleep <= 9:

            positives.append(
                f"You reported {sleep:g} hours of sleep per day, "
                "which is within the commonly recommended range for adults."
            )

        elif sleep < 7:

            attention.append(
                f"You reported only {sleep:g} hours of sleep per day. "
                "Consider maintaining a consistent sleep routine "
                "and allowing enough time for rest."
            )

        else:

            attention.append(
                f"You reported {sleep:g} hours of sleep per day. "
                "Try to maintain a consistent sleep schedule."
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
                "Consider adding more regular movement to your routine."
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
                "Consider maintaining regular hydration throughout the day."
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
                "Regular breaks, physical activity, relaxation, "
                "or time away from screens may help with stress management."
            )

        else:

            attention.append(
                f"Your reported stress level is {stress}/10. "
                "This is high on the scale used in this application. "
                "Consider paying more attention to stress management."
            )


    # -----------------------------------------------------
    # BMI
    # -----------------------------------------------------

    if "BMI" in values:

        bmi = values["BMI"]

        if 18.5 <= bmi < 25:

            positives.append(
                f"Your entered BMI is {bmi:.1f}."
            )

        else:

            attention.append(
                f"Your entered BMI is {bmi:.1f}. "
                "BMI is only one health indicator and should be "
                "interpreted in the context of other factors."
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
                "Consider keeping consumption within your personal "
                "health goals."
            )


    return positives, attention


# =========================================================
# RUN PREDICTION
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
        # CONVERT EVERYTHING TO NUMERIC
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
        # PREDICT
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


        if predicted_class == 1:

            st.warning(
                "⚠️ The model classified your profile as **Class 1**."
            )

        else:

            st.success(
                "✅ The model classified your profile as **Class 0**."
            )


        st.caption(
            "Class 0 and Class 1 represent the target classes "
            "learned from the training dataset. They should not "
            "be interpreted as a medical diagnosis."
        )


        # =================================================
        # MODEL CONFIDENCE
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
        # HUMAN-FRIENDLY OBSERVATIONS
        # =================================================

        st.subheader(
            "💬 What your information tells us"
        )

        st.write(
            "These observations are based on the information "
            "you entered. They are separate from the model's "
            "classification."
        )


        positives, attention = generate_feedback(
            input_values
        )


        # -------------------------------------------------
        # POSITIVE POINTS
        # -------------------------------------------------

        if positives:

            st.markdown(
                "### ✅ Positive points"
            )

            for item in positives:

                st.write(
                    f"• {item}"
                )


        # -------------------------------------------------
        # ATTENTION
        # -------------------------------------------------

        if attention:

            st.markdown(
                "### 💡 Things you may want to pay attention to"
            )

            for item in attention:

                st.write(
                    f"• {item}"
                )

        else:

            st.success(
                "No major lifestyle areas were highlighted "
                "by the basic checks in this application."
            )


        # =================================================
        # SIMPLE SUMMARY
        # =================================================

        st.subheader(
            "📝 Simple Summary"
        )


        if predicted_class == 1:

            st.write(
                "Based on the information provided, the machine "
                "learning model placed this profile in Class 1. "
                "This is a model prediction and should be considered "
                "alongside the individual factors shown above."
            )

        else:

            st.write(
                "Based on the information provided, the machine "
                "learning model placed this profile in Class 0. "
                "This is the classification produced by the trained model."
            )


        # =================================================
        # GENERAL SUGGESTIONS
        # =================================================

        st.subheader(
            "🌱 General Suggestions"
        )

        st.markdown(
            """
            - Maintain a consistent sleep routine.
            - Include regular physical activity in your routine.
            - Stay adequately hydrated.
            - Pay attention to stress and mental well-being.
            - Avoid or reduce smoking and excessive alcohol consumption.
            - Keep track of important health measurements.
            - If you have symptoms or concerns about your health,
              consult a qualified healthcare professional.
            """
        )


        # =================================================
        # TECHNICAL DETAILS
        # =================================================

        with st.expander(
            "🔎 View technical model details"
        ):

            st.write(
                "Processed data sent to the machine learning model:"
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
