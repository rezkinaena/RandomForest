import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === KONFIGURASI ===
# Ganti link di bawah dengan link RAW CSV di GitHub kamu
CSV_URL = "https://raw.githubusercontent.com/username/repo/main/onlinefoods.csv"

# Nama file model
MODEL_FILE = "best_model.pkl"

# Kolom yang digunakan saat training
FITUR_WAJIB = [
    'Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
    'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code'
]

FITUR_NUMERIK = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']

# === LOAD MODEL ===
model = joblib.load(MODEL_FILE)

# === LOAD DATASET UNTUK ENCODER & SCALER ===
dataset = pd.read_csv(CSV_URL)
dataset = dataset[FITUR_WAJIB].copy()

# Buat label encoders
label_encoders = {}
for kolom in dataset.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataset[kolom] = dataset[kolom].astype(str)
    dataset[kolom] = le.fit_transform(dataset[kolom])
    label_encoders[kolom] = le

# Buat scaler
scaler = StandardScaler()
dataset[FITUR_NUMERIK] = scaler.fit_transform(dataset[FITUR_NUMERIK])

# === FUNGSI PROSES INPUT ===
def proses_input(user_data):
    df = pd.DataFrame([user_data])

    # Transform label encoding
    for kolom in label_encoders:
        if kolom in df.columns:
            df[kolom] = label_encoders[kolom].transform(df[kolom])

    # Transform scaling numerik
    df[FITUR_NUMERIK] = scaler.transform(df[FITUR_NUMERIK])

    # Pastikan urutan kolom sama seperti training
    df = df[FITUR_WAJIB]

    return df

# === STREAMLIT UI ===
st.title("Prediksi Keberadaan Pelanggan")

# Input user
user_input = {
    'Age': st.number_input('Usia', 18, 100),
    'Gender': st.selectbox('Jenis Kelamin', label_encoders['Gender'].classes_),
    'Marital Status': st.selectbox('Status Pernikahan', label_encoders['Marital Status'].classes_),
    'Occupation': st.selectbox('Pekerjaan', label_encoders['Occupation'].classes_),
    'Monthly Income': st.selectbox('Pendapatan Bulanan', label_encoders['Monthly Income'].classes_),
    'Educational Qualifications': st.selectbox('Pendidikan', label_encoders['Educational Qualifications'].classes_),
    'Family size': st.number_input('Jumlah Keluarga', 1, 20),
    'latitude': st.number_input('Latitude', format="%.6f"),
    'longitude': st.number_input('Longitude', format="%.6f"),
    'Pin code': st.number_input('Kode Pos', 1, 999999)
}

# Tombol prediksi
if st.button("Prediksi"):
    try:
        data_terproses = proses_input(user_input)
        pred = model.predict(data_terproses)[0]
        st.success("Ditemukan ✅" if pred == 1 else "Tidak Ditemukan ❌")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")
