import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading trained model
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("linear_regression_model.pkl")  # Ensure this file exists
scaler = joblib.load("scaler.pkl")  # Ensure this file exists

# Streamlit UI
st.title("📊 Predict CGPA using Multiple Regression")
st.write("Enter the details below to predict your CGPA")

# User Inputs (Relevant features from df_encoded)
study_hours = st.number_input("📚 Study Hours per day", min_value=0.0, max_value=24.0, step=0.1)
before_study = st.number_input("📆 Study Hours Before Exam", min_value=0.0, max_value=24.0, step=0.1)
during_study = st.number_input("📆 Study Hours During Exam", min_value=0.0, max_value=24.0, step=0.1)

# Categorical Inputs (One-hot encoded in df_encoded)
study_methods = [
    "Class lectures and notes only",
    "Solo(Online via video lectures)",
    "I prefer both class notes, solo study with online lectures when needed",
    "Solo(Offline via books and lecture notes)",
    "Solo(Offline via books and lecture notes), Class lectures and notes only",
    "Solo(Offline via books and lecture notes), Class lectures and notes only, Solo(Online via video lectures)",
    "Solo(Offline via books and lecture notes), Solo(Online via video lectures)",
    "Solo(Online via video lectures)",
    "Solo(Online via video lectures), Friend teaches me"
]
selected_study_method = st.selectbox("📖 Study Method", study_methods)

environments = ["Library", "Room(Group Study)", "Room(Solo)"]
selected_environment = st.selectbox("🏠 Study Environment", environments)

# Convert categorical input to one-hot encoding
study_method_features = [1 if method == selected_study_method else 0 for method in study_methods]
environment_features = [1 if env == selected_environment else 0 for env in environments]

# Prepare input data (Ensure order matches df_encoded)
input_data = [before_study, during_study] + study_method_features + environment_features
input_data = np.array([input_data])  # Convert to NumPy array

# Scale input data (same transformation as during training)
input_data_scaled = scaler.transform(input_data)  # Ensure same number of features

# Predict CGPA
if st.button("Predict CGPA"):
    predicted_cgpa = model.predict(input_data_scaled)[0]
    st.success(f"🎓 Predicted CGPA: {predicted_cgpa:.2f}")
