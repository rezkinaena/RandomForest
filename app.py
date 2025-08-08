import streamlit as st
import pandas as pd
import joblib

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
    df = dataset.copy()
    required_columns = [
    'Age', 'Gender', 'Marital Status', 'Occupation',
    'Monthly Income', 'Educational Qualifications',
    'Family size', 'Pin code', 'Feedback', 'Output'
]

    # Gabungkan input user ke dataset asli untuk encoding konsisten
    df = pd.concat([df[required_columns], pd.DataFrame([input_data])], ignore_index=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    return df.iloc[-1].values.reshape(1, -1)

# ========================
# 3. CSS Soft Pastel Theme
# ========================
st.markdown("""
    <style>
    body {
        background-color: #fdf6f0;
        color: #333333;
    }
    .stApp {
        background-color: #fefcfb;
    }
    h1 {
        color: #6d6875;
        text-align: center;
    }
    .stButton button {
        background-color: #a8dadc;
        color: #1d3557;
        border-radius: 10px;
        padding: 0.6em 1em;
        font-weight: bold;
        border: none;
    }
    .stButton button:hover {
        background-color: #457b9d;
        color: white;
    }
    .stSelectbox label, .stNumberInput label {
        color: #6d6875;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# 4. Streamlit UI
# ========================
st.title("🍔 Prediksi Pelanggan Online Food Service 📦")
st.write("Isi data di bawah ini untuk memprediksi apakah pelanggan akan memesan makanan online.")

# Input Form
age = st.number_input("Umur", min_value=10, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", dataset["Gender"].unique())
marital_status = st.selectbox("Status Pernikahan", dataset["Marital Status"].unique())
occupation = st.selectbox("Pekerjaan", dataset["Occupation"].unique())
income = st.selectbox("Pendapatan Bulanan", dataset["Monthly Income"].unique())
education = st.selectbox("Kualifikasi Pendidikan", dataset["Educational Qualifications"].unique())
family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=20, value=3)

# Prediksi
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

