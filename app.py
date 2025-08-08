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

    # Encoding kategori ke angka
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    return df.iloc[-1].values.reshape(1, -1)

# ========================
# 3. CSS Soft Pastel Theme
# ========================
st.markdown("""
    <style>
    body {
        background-color: #fff8f0;
        color: #333333;
    }
    .stApp {
        background-color: #fffaf5;
    }
    h1 {
        color: #ff8fab;
        text-align: center;
    }
    .stButton button {
        background-color: #a8e6cf;
        color: #1d3557;
        border-radius: 10px;
        padding: 0.6em 1em;
        font-weight: bold;
        border: none;
    }
    .stButton button:hover {
        background-color: #ffb5a7;
        color: white;
    }
    .stSelectbox label, .stNumberInput label {
        color: #6d6875;
        font-weight: bold;
    }
    .stInfo, .stSuccess, .stWarning {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# 4. Streamlit UI
# ========================
st.title("🍔 Prediksi Pelanggan Online Food Service 📦")
st.write("Isi data di bawah ini untuk memprediksi apakah pelanggan akan memesan makanan online.")
st.caption("📌 Keterangan Label: **0 = Tidak Memesan**, **1 = Memesan**")

# Form Input
age = st.number_input("Umur", min_value=10, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", dataset["Gender"].unique())
marital_status = st.selectbox("Status Pernikahan", dataset["Marital Status"].unique())
occupation = st.selectbox("Pekerjaan", dataset["Occupation"].unique())
income = st.selectbox("Pendapatan Bulanan", dataset["Monthly Income"].unique())
education = st.selectbox("Kualifikasi Pendidikan", dataset["Educational Qualifications"].unique())
family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=20, value=3)
pin_code = st.number_input("Kode Pos", min_value=10000, max_value=999999, value=123456)
feedback = st.selectbox("Feedback", dataset["Feedback"].unique())

# ========================
# 5. Prediction
# ========================
if st.button("🔮 Prediksi"):
    input_data = {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Monthly Income': income,
        'Educational Qualifications': education,
        'Family size': family_size,
        'Pin code': pin_code,
        'Feedback': feedback,
        'Output': 0  # placeholder
    }

    processed_input = preprocess_input(input_data)
    
    # Prediksi kelas
    prediction = model.predict(processed_input)[0]
    
    # Prediksi probabilitas
    prediction_proba = model.predict_proba(processed_input)[0]

    st.info(f"🔢 **Hasil Prediksi Model**: {prediction}  \n📌 (0 = Tidak Memesan, 1 = Memesan)")
    st.write(f"🍃 Probabilitas **Tidak Memesan (0)**: **{prediction_proba[0]*100:.2f}%**")
    st.write(f"🌸 Probabilitas **Memesan (1)**: **{prediction_proba[1]*100:.2f}%**")

    if prediction == 1:
        st.success("✅ Interpretasi: Pelanggan kemungkinan **akan** memesan makanan online.")
    else:
        st.warning("❌ Interpretasi: Pelanggan kemungkinan **tidak akan** memesan makanan online.")
