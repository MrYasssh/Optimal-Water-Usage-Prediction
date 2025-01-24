import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_filename = "rf_regressor_model_joblib.pkl"

try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Model file not found. Ensure the '.pkl' file is in the same directory as this app.")
    st.stop()

# Title and description
st.title("💧 Optimal Water Usage Prediction 🌱")
st.markdown("""
Welcome to the **Optimal Water Usage Prediction App**! 🌾

This tool helps you predict the water requirement for crops based on soil type, crop type, and environmental conditions. 
Please provide the necessary details below to get started.
""")

st.sidebar.header("Input Parameters")

# Input fields in the sidebar
soil_type = st.sidebar.selectbox("Select Soil Type", ["Sandy", "Clay", "Loamy", "Silty", "Peaty"], index=0)
crop_type = st.sidebar.selectbox("Select Crop Type", ["Rice", "Wheat", "Maize", "Soybean", "Cotton"], index=0)
growth_stage = st.sidebar.selectbox("Select Growth Stage", ["Seedling", "Vegetative", "Flowering", "Fruiting"], index=0)
temperature = st.sidebar.slider("Temperature (°C)", min_value=15.0, max_value=40.0, value=25.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=50.0, step=0.1)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0.5, max_value=15.0, value=5.0, step=0.1)
evapotranspiration = st.sidebar.slider("Evapotranspiration (mm/day)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
rainfall = st.sidebar.slider("Rainfall (mm)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
soil_moisture = st.sidebar.slider("Soil Moisture Levels (%)", min_value=5.0, max_value=40.0, value=20.0, step=0.1)
water_retention = st.sidebar.slider("Water Retention Capacity (%)", min_value=10.0, max_value=50.0, value=30.0, step=0.1)
drainage_properties = st.sidebar.slider("Drainage Properties (1=Poor, 5=Excellent)", min_value=1, max_value=5, value=3, step=1)
irrigation_method = st.sidebar.selectbox("Select Irrigation Method", ["Drip", "Sprinkler", "Flood", "Manual"], index=0)

# Map categorical inputs to numerical values
soil_type_mapping = {"Sandy": 0, "Clay": 1, "Loamy": 2, "Silty": 3, "Peaty": 4}
crop_type_mapping = {"Rice": 0, "Wheat": 1, "Maize": 2, "Soybean": 3, "Cotton": 4}
growth_stage_mapping = {"Seedling": 0, "Vegetative": 1, "Flowering": 2, "Fruiting": 3}
irrigation_method_mapping = {"Drip": 0, "Sprinkler": 1, "Flood": 2, "Manual": 3}

# Create feature array
features = np.array([
    soil_type_mapping[soil_type],
    crop_type_mapping[crop_type],
    growth_stage_mapping[growth_stage],
    temperature,
    humidity,
    wind_speed,
    evapotranspiration,
    rainfall,
    soil_moisture,
    water_retention,
    drainage_properties,
    irrigation_method_mapping[irrigation_method]
]).reshape(1, -1)

# Predict and display results
st.header("Prediction Results")
if st.button("💧 Predict Water Requirement"):
    prediction = model.predict(features)
    st.success(f"🌟 Predicted Optimal Water Requirement: **{prediction[0]:.2f} mm/day**")

    # Provide suggestions for optimizing water usage
    st.header("Optimization Suggestions")

    suggestions = []

    # Check soil type
    if soil_type == "Sandy":
        suggestions.append("Consider improving soil structure by adding organic matter or compost to enhance water retention.")

    # Check evapotranspiration
    if evapotranspiration > 8.0:
        suggestions.append("High evapotranspiration detected. Use mulching to reduce evaporation and conserve soil moisture.")

    # Check soil moisture
    if soil_moisture < 15.0:
        suggestions.append("Low soil moisture detected. Ensure proper irrigation scheduling to maintain adequate levels.")

    # Check water retention
    if water_retention < 20.0:
        suggestions.append("Low water retention capacity. Incorporate soil amendments to improve retention.")

    # Check drainage
    if drainage_properties < 3:
        suggestions.append("Poor drainage detected. Improve soil aeration or use soil conditioners to balance drainage.")

    # Display suggestions
    if suggestions:
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write("✅ All parameters are well-balanced for optimal water usage!")
