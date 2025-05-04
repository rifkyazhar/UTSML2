import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load model
model = load_model("real.h5") # Pastikan file berada di folder yang sama

# Title
st.title("Prediksi Harga Real Estate")

# Input form
house_age = st.number_input("Usia Rumah (tahun)", min_value=0.0, max_value=100.0)
mrt_distance = st.number_input("Jarak ke MRT Terdekat (meter)", min_value=0.0)
stores = st.number_input("Jumlah Toko Serbaguna", min_value=0)
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

# Buat array input
input_data = np.array([[house_age, mrt_distance, stores, latitude, longitude]])

# Normalisasi jika diperlukan
# scaler = joblib.load("scaler.sav")
# input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi Harga"):
    prediction = model.predict(input_data)  # gunakan input_scaled jika pakai scaler
    st.success(f"Perkiraan Harga: {prediction[0][0]:,.2f} juta")

