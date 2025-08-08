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
# 3. CSS Pastel Peach-Pink-Mint 🌸🍑🍃
# ========================
st.markdown("""
    <style>
    body {
        background-color: #fffaf6;
        color: #444444;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #fffaf6;
    }
    h1 {
        color: #ff8fa3;
        text-align: center;
        font-weight: 800;
    }
    /* Tombol */
    .stButton button {
        background: linear-gradient(135deg, #ffd6a5, #ffcad4, #caffbf);
        color: #444;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        transition: 0.3s ease-in-out;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #ffb5a7, #ff99ac, #a3f7bf);
        color: white;
        transform: scale(1.05);
    }
    /* Label */
    .stSelectbox label, .stNumberInput label {
        color: #ff99ac;
        font-weight: bold;
    }
    /* Info Box */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# 4. Streamlit UI
# ========================
st.title("🌸🍑🍃 Prediksi Pelanggan Online Food Service 🍃🍑🌸")
st.write("Isi data di bawah ini untuk memprediksi apakah pelanggan akan memesan makanan online.")

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
if st.button("💡 Prediksi Sekarang!"):
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
    prediction_proba = model.predict_proba(processed_input)[0]  # [prob_class0, prob_class1]

    # Keterangan label
    label_keterangan = {
        0: "Tidak Memesan Makanan Online",
        1: "Akan Memesan Makanan Online"
    }

    st.info(f"🔢 Hasil Prediksi Model: **{prediction}** ({label_keterangan[prediction]})")
    st.write(f"📊 Probabilitas Tidak Memesan (0): **{prediction_proba[0]*100:.2f}%**")
    st.write(f"📊 Probabilitas Memesan (1): **{prediction_proba[1]*100:.2f}%**")

    if prediction == 1:
        st.success("🌸 Pelanggan kemungkinan akan memesan makanan online.")
    else:
        st.warning("🍑 Pelanggan kemungkinan tidak akan memesan makanan online.")
