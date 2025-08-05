import streamlit as st
import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Fake Job Detector", layout="centered")

st.title("🕵️‍♂️ Fake Job Posting Detection")

st.markdown("""
Enter job details below. The system will analyze the content and predict whether the job posting is **real or fake**.
""")

# Input fields for important textual features
title = st.text_input("Job Title", "")
location = st.text_input("Location", "")
department = st.text_input("Department", "")
company_profile = st.text_area("Company Profile", "")
description = st.text_area("Job Description", "")
requirements = st.text_area("Requirements", "")
benefits = st.text_area("Benefits", "")

# Button to predict
if st.button("Predict"):
    # Combine all text fields into one input for vectorizer
    input_text = ' '.join([title, location, department, company_profile, description, requirements, benefits])

    # Vectorize input
    input_vector = vectorizer.transform([input_text])

    # Predict
    prediction = model.predict(input_vector)[0]
    confidence = model.predict_proba(input_vector)[0][prediction]

    # Output result
    label = "🚨 Fake Job Posting" if prediction == 1 else "✅ Real Job Posting"
    color = "red" if prediction == 1 else "green"

    st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)
    st.write(f"**Confidence:** {confidence:.2f}")
