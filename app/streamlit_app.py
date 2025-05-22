# app/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from scripts.predict import predict_audio

st.title("👶 Baby Cry Reason Predictor")

uploaded_file = st.file_uploader("Upload baby cry audio (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    reason = predict_audio("temp.wav")
    st.success(f"Predicted Reason: **{reason}**")
