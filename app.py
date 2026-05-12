import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📘"
)

st.title("Student Score Predictor")

st.write("Predict student exam score")

# Load model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("model.pkl file not found!")
    st.stop()

# Inputs
study_hours = st.slider(
    "Study Hours",
    1, 12, 5
)

sleep_hours = st.slider(
    "Sleep Hours",
    1, 12, 7
)

attendance = st.slider(
    "Attendance %",
    50, 100, 75
)

# Prediction
if st.button("Predict"):

    features = pd.DataFrame({
        'study_hours': [study_hours],
        'sleep_hours': [sleep_hours],
        'attendance': [attendance]
    })

    prediction = model.predict(features)

    st.success(
        f"Predicted Score: {prediction[0]:.2f}"
    )