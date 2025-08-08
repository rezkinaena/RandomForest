import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ========================
# 1. Load Model & Dataset
# ========================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("onlinefoods.csv")

model = load_model()
dataset = load_data()

# ========================
# 2. Preprocessing Helper
# ========================
def preprocess_input(input_data):
    # Copy dataset untuk ambil encoder dari data asli
    df = dataset.copy()

    # Pastikan urutan kolom sesuai saat training
    required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation',
                        'Monthly Income', 'Educational Qualifications',
                        'Family size']

    # Gabungkan input user ke dataset
    df = pd.concat([df[required_columns], pd.DataFrame([input_data])], ignore_index=True)

    # Encode data kategorik
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    # Ambil baris terakhir (input user)
    processed = df.iloc[-1].values.reshape(1, -1)
    return processed

# ========================
# 3. Streamlit UI
# ========================
st.title("Prediksi Pelanggan Online Food Service 🍔📦")

st.write("Isi data di bawah ini untuk memprediksi apakah pelanggan akan memesan makanan online.")

age = st.number_input("Umur", min_value=10, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", dataset["Gender"].unique())
marital_status = st.selectbox("Status Pernikahan", dataset["Marital Status"].unique())
occupation = st.selectbox("Pekerjaan", dataset["Occupation"].unique())
income = st.selectbox("Pendapatan Bulanan", dataset["Monthly Income"].unique())
education = st.selectbox("Kualifikasi Pendidikan", dataset["Educational Qualifications"].unique())
family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=20, value=3)

if st.button("Prediksi"):
    input_data = {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Monthly Income': income,
        'Educational Qualifications': education,
        'Family size': family_size
    }

    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)[0]

    if prediction == 1:
        st.success("✅ Pelanggan kemungkinan akan memesan makanan online.")
    else:
        st.warning("❌ Pelanggan kemungkinan tidak akan memesan makanan online.")
