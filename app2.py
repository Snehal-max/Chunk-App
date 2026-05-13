import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Create churn column
df['Churn'] = df['Cancellation Year'].notnull().astype(int)

# Fill missing salary
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# Aggregate customer data
customer_df = df.groupby('Loyalty Number').agg({
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
    'Enrollment Type':'first',
    'Churn':'max'
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

# Features and target
X = customer_df.drop(
    ['Loyalty Number', 'Churn'],
    axis=1
)

y = customer_df['Churn']

# Encoding
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

# Scaling
sc = StandardScaler(with_mean=False)

X = sc.fit_transform(X)

# Train model
classifier = SVC(
    kernel='rbf',
    random_state=0
)

classifier.fit(X, y)

# Streamlit UI
st.title("Customer Churn Prediction")

loyalty_number = st.number_input(
    "Enter Loyalty Number",
    min_value=0,
    step=1
)

if st.button("Predict"):

    customer = customer_df[
        customer_df['Loyalty Number'] == loyalty_number
    ]

    if customer.empty:

        st.error("Loyalty Number not found")

    else:

        customer_input = customer.drop(
            ['Loyalty Number', 'Churn'],
            axis=1
        )

        customer_encoded = ct.transform(customer_input)

        customer_scaled = sc.transform(customer_encoded)

        prediction = classifier.predict(customer_scaled)

        st.subheader("Customer Details")
        st.write(customer)

        if prediction[0] == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is likely to stay")