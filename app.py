import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -------------------------------
# Load datasets
# -------------------------------

loyalty = pd.read_csv("Customer Loyalty History.csv")
flight = pd.read_csv("Customer Flight Activity.csv")

# -------------------------------
# Merge datasets
# -------------------------------

df = pd.merge(
    flight,
    loyalty,
    on="Loyalty Number",
    how="left"
)

# -------------------------------
# Create target column
# -------------------------------

df['Churn'] = (
    df['Cancellation Year']
    .notnull()
    .astype(int)
)

# -------------------------------
# Handle missing values
# -------------------------------

df['Salary'] = (
    df['Salary']
    .fillna(df['Salary'].median())
)

# -------------------------------
# Aggregate customer data
# -------------------------------

customer_df = df.groupby(
    'Loyalty Number'
).agg({

    'Total Flights': 'sum',
    'Distance': 'sum',
    'Points Accumulated': 'sum',
    'Points Redeemed': 'sum',

    'Salary': 'first',
    'CLV': 'first',

    'Gender': 'first',
    'Education': 'first',
    'Marital Status': 'first',
    'Loyalty Card': 'first',
    'Enrollment Type': 'first',

    'Churn': 'max'

}).reset_index()

# -------------------------------
# Feature Engineering
# -------------------------------

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

# -------------------------------
# Features and Target
# -------------------------------

X = customer_df.drop(
    ['Loyalty Number', 'Churn'],
    axis=1
)

y = customer_df['Churn']

# -------------------------------
# Encoding
# -------------------------------

ct = ColumnTransformer(
    transformers=[
        (
            'encoder',
            OneHotEncoder(drop='first'),
            [
                'Gender',
                'Education',
                'Marital Status',
                'Loyalty Card',
                'Enrollment Type'
            ]
        )
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

# -------------------------------
# Scaling
# -------------------------------

sc = StandardScaler(with_mean=False)

X = sc.fit_transform(X)

# -------------------------------
# Train Model
# -------------------------------

model = SVC(
    kernel='rbf',
    random_state=0
)

model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Customer Churn Prediction App")

st.write("Enter Customer Information")

# Inputs

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
    min_value=0
)

distance = st.number_input(
    "Distance",
    min_value=0.0
)

points_accumulated = st.number_input(
    "Points Accumulated",
    min_value=0.0
)

points_redeemed = st.number_input(
    "Points Redeemed",
    min_value=0.0
)

salary = st.number_input(
    "Salary",
    min_value=0.0
)

clv = st.number_input(
    "CLV",
    min_value=0.0
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

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

    # Encode
    input_encoded = ct.transform(input_df)

    # Scale
    input_scaled = sc.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)

    # Output
    if prediction[0] == 1:

        st.error(
            "Customer is likely to churn"
        )

    else:

        st.success(
            "Customer is likely to stay"
        )