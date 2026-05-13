import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load saved files safely
# -------------------------------

try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction App")

st.write("Enter Customer Information")

# -------------------------------
# Input fields
# -------------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education",
    [
        "Bachelor",
        "College",
        "Doctor",
        "High School",
        "Master"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

loyalty_card = st.selectbox(
    "Loyalty Card",
    [
        "Aurora",
        "Nova",
        "Star"
    ]
)

enrollment_type = st.selectbox(
    "Enrollment Type",
    [
        "Standard",
        "2018 Promotion"
    ]
)

total_flights = st.number_input(
    "Total Flights",
    min_value=0,
    value=0
)

distance = st.number_input(
    "Distance",
    min_value=0.0,
    value=0.0
)

points_accumulated = st.number_input(
    "Points Accumulated",
    min_value=0.0,
    value=0.0
)

points_redeemed = st.number_input(
    "Points Redeemed",
    min_value=0.0,
    value=0.0
)

salary = st.number_input(
    "Salary",
    min_value=0.0,
    value=0.0
)

clv = st.number_input(
    "CLV",
    min_value=0.0,
    value=0.0
)

# -------------------------------
# Predict Button
# -------------------------------

if st.button("Predict Churn"):

    try:

        # -------------------------------
        # Feature Engineering
        # -------------------------------

        redemption_rate = (
            points_redeemed /
            (points_accumulated + 1)
        )

        avg_distance_per_flight = (
            distance /
            (total_flights + 1)
        )

        points_per_flight = (
            points_accumulated /
            (total_flights + 1)
        )

        clv_per_flight = (
            clv /
            (total_flights + 1)
        )

        # -------------------------------
        # Create dataframe
        # -------------------------------

        input_df = pd.DataFrame({

            'Gender': [gender],
            'Education': [education],
            'Marital Status': [marital_status],
            'Loyalty Card': [loyalty_card],
            'Enrollment Type': [enrollment_type],

            'Total Flights': [total_flights],
            'Distance': [distance],
            'Points Accumulated': [points_accumulated],
            'Points Redeemed': [points_redeemed],
            'Salary': [salary],
            'CLV': [clv],

            'Redemption Rate': [redemption_rate],
            'Avg Distance Per Flight': [avg_distance_per_flight],
            'Points Per Flight': [points_per_flight],
            'CLV Per Flight': [clv_per_flight]

        })

        # -------------------------------
        # Transform data
        # -------------------------------

        input_encoded = encoder.transform(input_df)

        input_scaled = scaler.transform(input_encoded)

        # -------------------------------
        # Prediction
        # -------------------------------

        prediction = model.predict(input_scaled)

        # -------------------------------
        # Result
        # -------------------------------

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error(
                "Customer is likely to churn"
            )

        else:

            st.success(
                "Customer is likely to stay"
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")