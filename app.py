
import streamlit as st
import pandas as pd
import joblib

# Load our trained Linear Regression model
model = joblib.load("linear_regression_model.pkl")

# Configure the webpage
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Frontend heading
st.title("🏠 House Price Prediction")
st.write(
    "Predict the median house value for a California district "
    "using our trained Linear Regression model."
)

st.divider()

# Collect input features
st.subheader("Enter Housing Details")

med_inc = st.number_input(
    "Median Income (dataset units)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

house_age = st.number_input(
    "Median House Age (years)",
    min_value=1.0,
    max_value=52.0,
    value=25.0,
    step=1.0
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.1,
    value=5.0,
    step=0.1
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.1,
    value=1.0,
    step=0.1
)

population = st.number_input(
    "Population",
    min_value=1.0,
    value=1000.0,
    step=100.0
)

ave_occup = st.number_input(
    "Average Occupancy",
    min_value=0.1,
    value=3.0,
    step=0.1
)

latitude = st.number_input(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=34.05,
    step=0.01,
    format="%.4f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-118.25,
    step=0.01,
    format="%.4f"
)

# Run prediction when the button is clicked
if st.button("Predict House Value", type="primary"):

    # Keep the feature names and order identical to training
    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    # Ask our trained model to make a prediction
    prediction = model.predict(input_data)[0]

    # The target is measured in units of $100,000
    predicted_dollars = prediction * 100000

    st.success("Prediction completed!")
    st.metric(
        label="Estimated Median House Value",
        value=f"${predicted_dollars:,.0f}"
    )

    st.caption(
        "This is a model estimate based on district-level California "
        "housing data, not a guaranteed price for an individual house."
    )
