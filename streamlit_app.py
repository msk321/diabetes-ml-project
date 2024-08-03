import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Title
st.title("Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=150)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=85)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=30)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.5)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input("Age", min_value=0, max_value=120, value=45)

# Prediction
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict_proba(features)
    output = prediction[0][1] * 100  # Convert to percentage

    if output > 25:
        precaution = "Your risk is high. Please consider consulting a doctor and follow these precautionary measures: ..."
    else:
        precaution = "Your risk is low. Maintain a healthy lifestyle to keep it that way."

    st.write(f"Your diabetes risk is {output:.2f}%. {precaution}")

