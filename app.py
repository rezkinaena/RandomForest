import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === Load model ===
model = joblib.load('best_model.pkl')

# === Load dataset untuk encoder & scaler ===
dataset = pd.read_csv('onlinefoods.csv')

fitur_wajib = [
    'Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
    'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code'
]
dataset = dataset[fitur_wajib]

# Label encoding
label_encoders = {}
for kolom in dataset.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataset[kolom] = le.fit_transform(dataset[kolom].astype(str))
    label_encoders[kolom] = le

# Scaling numerik
scaler = StandardScaler()
fitur_numerik = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
dataset[fitur_numerik] = scaler.fit_transform(dataset[fitur_numerik])

# === Fungsi proses input ===
def proses_input(user_data):
    df = pd.DataFrame([user_data])
    for kolom in label_encoders:
        df[kolom] = label_encoders[kolom].transform(df[kolom])
    df[fitur_numerik] = scaler.transform(df[fitur_numerik])
    return df

# === Streamlit App ===
st.title("Prediksi Keberadaan Pelanggan")

# Input singkat
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
    data_terproses = proses_input(user_input)
    pred = model.predict(data_terproses)[0]
    st.success("Ditemukan" if pred == 1 else "Tidak Ditemukan")
