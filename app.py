import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Muat model terbaik
model = joblib.load('best_model.pkl')

# Muat dataset untuk kebutuhan encoding dan normalisasi
dataset = pd.read_csv('onlinefoods.csv')

# Kolom yang digunakan saat pelatihan
fitur_wajib = [
    'Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
    'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code'
]

dataset = dataset[fitur_wajib]

# Proses encoding label
label_encoders = {}
for kolom in dataset.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataset[kolom] = dataset[kolom].astype(str)
    le.fit(dataset[kolom])
    dataset[kolom] = le.transform(dataset[kolom])
    label_encoders[kolom] = le

# Normalisasi fitur numerik
scaler = StandardScaler()
fitur_numerik = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
dataset[fitur_numerik] = scaler.fit_transform(dataset[fitur_numerik])

# Fungsi untuk memproses input user
def proses_input(user_data):
    data_terproses = {col: [user_data.get(col, 'Unknown')] for col in fitur_wajib}
    for kolom in label_encoders:
        if kolom in data_terproses:
            nilai_input = data_terproses[kolom][0]
            if nilai_input in label_encoders[kolom].classes_:
                data_terproses[kolom] = label_encoders[kolom].transform([nilai_input])
            else:
                data_terproses[kolom] = [-1]  # nilai default jika tidak dikenal
    data_terproses = pd.DataFrame(data_terproses)
    data_terproses[fitur_numerik] = scaler.transform(data_terproses[fitur_numerik])
    return data_terproses

# CSS Kustom - tema biru pastel + hijau dengan tombol orange
st.markdown("""
    <style>
    .main {
        background-color: #e0f7fa;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    h1 {
        color: #00695c;
        text-align: center;
        margin-bottom: 25px;
    }
    h3 {
        color: #004d40;
    }
    .stButton>button {
        background-color: #ff7043;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #f4511e;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("Prediksi Keberadaan Pelanggan")

st.markdown("""
    <h3>Isi Data Pelanggan yang Ingin Diperiksa</h3>
""", unsafe_allow_html=True)

# Input pengguna
age = st.number_input('Usia', min_value=18, max_value=100)
gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
marital_status = st.selectbox('Status Pernikahan', ['Belum Menikah', 'Sudah Menikah'])
occupation = st.selectbox('Pekerjaan', ['Pelajar', 'Karyawan', 'Wira Swasta'])
monthly_income = st.selectbox('Pendapatan Bulanan', ['Tidak Ada', 'Dibawah Rs.10000', '10001 hingga 25000', '25001 hingga 50000', 'Lebih dari 50000'])
educational_qualifications = st.selectbox('Pendidikan Terakhir', ['Sarjana Muda', 'Lulusan/Sarjana', 'Pasca Sarjana'])
family_size = st.number_input('Jumlah Anggota Keluarga', min_value=1, max_value=20)
latitude = st.number_input('Latitude', format="%f")
longitude = st.number_input('Longitude', format="%f")
pin_code = st.number_input('Kode Pos', min_value=100000, max_value=999999)

# Data user
user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
}

# Tombol prediksi
if st.button('Prediksi'):
    user_processed = proses_input(user_input)
    try:
        hasil = model.predict(user_processed)
        st.success(f"Prediksi: {'Ditemukan' if hasil[0] == 1 else 'Tidak Ditemukan'}")
    except ValueError as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

# Keterangan
st.markdown("""
    <div style='color:#004d40; font-weight:bold; margin-top:20px;'>
    <p>Keterangan:</p>
    <ul>
        <li>0 : Tidak ada pelanggan dengan kriteria tersebut dalam data</li>
        <li>1 : Ada pelanggan dengan kriteria tersebut dalam data</li>
    </ul>
    </div>
""", unsafe_allow_html=True)
