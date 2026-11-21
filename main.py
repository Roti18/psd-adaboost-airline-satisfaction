"""
Aplikasi Web Streamlit: Klasifikasi Kepuasan Penumpang Maskapai (AdaBoost).
File utama untuk routing navigasi sidebar dan pemanggilan modul halaman.
"""

import streamlit as st

from src.halaman import beranda, eda, model_evaluasi, prediksi

# Konfigurasi dasar halaman antarmuka Streamlit
st.set_page_config(
    page_title="Prediksi Kepuasan Penumpang",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigasi sidebar dengan Google Material Icons bawaan Streamlit
st.sidebar.title(":material/flight: Menu Utama")
pilihan_menu = st.sidebar.radio(
    label="Pilih Halaman",
    options=[
        ":material/home: Beranda",
        ":material/bar_chart: EDA",
        ":material/model_training: Model dan Evaluasi",
        ":material/online_prediction: Prediksi",
    ],
)


# Pemanggilan fungsi tampil() halaman terpilih
if "Beranda" in pilihan_menu:
    beranda.tampil()
elif "EDA" in pilihan_menu:
    eda.tampil()
elif "Model dan Evaluasi" in pilihan_menu:
    model_evaluasi.tampil()
elif "Prediksi" in pilihan_menu:
    prediksi.tampil()
