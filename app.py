import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💓 Heart Disease Prediction App")

st.markdown("Enter the patient's medical details:")

# User input
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0 = Normal, 1, 2)", [0, 1, 2])
thalachh = st.number_input("Maximum Heart Rate", min_value=60, max_value=220)
exng = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak", step=0.1)
slp = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
caa = st.selectbox("Number of Major Vessels Colored (0–4)", [0, 1, 2, 3, 4])
thall = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)", [0, 1, 2])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                            thalachh, exng, oldpeak, slp, caa, thall]])

    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    if prediction == 0:
        st.success(f"🟢 No Heart Disease Detected.\nProbability: {probabilities[0]:.2f}")
    else:
        st.error(f"🔴 Heart Disease Likely Detected!\nProbability: {probabilities[1]:.2f}")
