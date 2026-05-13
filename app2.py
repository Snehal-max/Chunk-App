import streamlit as st
import pandas as pd
import pickle

# Load model files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Load datasets
loyalty = pd.read_csv("Customer Loyalty History.csv")
flight = pd.read_csv("Customer Flight Activity.csv")

# Merge datasets
df = pd.merge(
    flight,
    loyalty,
    on="Loyalty Number",
    how="left"
)

# Fill missing salary
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

st.title("Customer Churn Prediction")

st.write("Enter Loyalty Number")

# Loyalty number input
loyalty_number = st.number_input(
    "Loyalty Number",
    min_value=0,
    step=1
)

# Predict button
if st.button("Predict Churn"):

    # Filter customer
    customer = df[df['Loyalty Number'] == loyalty_number]

    if customer.empty:
        st.error("Loyalty Number not found")
    
    else:

        # Aggregate customer data
        customer_df = customer.groupby('Loyalty Number').agg({
            'Total Flights':'sum',
            'Distance':'sum',
            'Points Accumulated':'sum',
            'Points Redeemed':'sum',
            'Salary':'first',
            'CLV':'first',
            'Gender':'first',
            'Education':'first',
            'Marital Status':'first',
            'Loyalty Card':'first',
            'Enrollment Type':'first'
        }).reset_index()

        # Feature engineering
        customer_df['Redemption Rate'] = (
            customer_df['Points Redeemed'] /
            (customer_df['Points Accumulated'] + 1)
        )

        customer_df['Avg Distance Per Flight'] = (
            customer_df['Distance'] /
            (customer_df['Total Flights'] + 1)
        )

        customer_df['Points Per Flight'] = (
            customer_df['Points Accumulated'] /
            (customer_df['Total Flights'] + 1)
        )

        customer_df['CLV Per Flight'] = (
            customer_df['CLV'] /
            (customer_df['Total Flights'] + 1)
        )

        # Keep only model features
        X = customer_df.drop(
            ['Loyalty Number'],
            axis=1
        )

        # Encode
        X_encoded = encoder.transform(X)

        # Scale
        X_scaled = scaler.transform(X_encoded)

        # Predict
        prediction = model.predict(X_scaled)

        # Show customer details
        st.subheader("Customer Information")

        st.write(customer_df)

        # Prediction result
        if prediction[0] == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is likely to stay")