"""
Modul pemuatan file CSV dengan caching Streamlit dan penanganan berkas hilang.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from src.config import FOLDER_DATA


def _baca_file(nama_file: str, index_col=None) -> pd.DataFrame | None:
    """Fungsi pembantu membaca file CSV dan menampilkan error jika berkas hilang."""
    path_file = FOLDER_DATA / nama_file
    if not path_file.exists():
        st.error(
            f"File '{nama_file}' tidak ditemukan di lokasi: {path_file.resolve()}",
            icon=":material/cancel:"
        )
        return None
    try:
        return pd.read_csv(path_file, index_col=index_col)
    except Exception as e:
        st.error(
            f"Gagal membaca file '{nama_file}' di lokasi: {path_file.resolve()} (Pesan: {e})",
            icon=":material/cancel:"
        )
        return None


@st.cache_data
def muat_train() -> pd.DataFrame | None:
    """Memuat data latih train.csv sebanyak 83.123 baris."""
    return _baca_file("train.csv")


@st.cache_data
def muat_test() -> pd.DataFrame | None:
    """Memuat data uji test.csv sebanyak 20.781 baris."""
    return _baca_file("test.csv")


@st.cache_data
def muat_model() -> pd.DataFrame | None:
    """Memuat parameter 67 decision stump AdaBoost dari model_adaboost.csv."""
    return _baca_file("model_adaboost.csv")


@st.cache_data
def muat_korelasi() -> pd.DataFrame | None:
    """Memuat korelasi Pearson fitur terhadap target dari korelasi.csv."""
    return _baca_file("korelasi.csv", index_col=0)


@st.cache_data
def muat_hasil_cv() -> pd.DataFrame | None:
    """Memuat hasil validasi silang 4-fold dari hasil_cv.csv."""
    return _baca_file("hasil_cv.csv", index_col=0)


@st.cache_data
def muat_ringkasan_cv() -> pd.DataFrame | None:
    """Memuat ringkasan rata-rata dan deviasi standar CV dari ringkasan_cv.csv."""
    return _baca_file("ringkasan_cv.csv", index_col=0)


@st.cache_data
def muat_perbandingan_konfig() -> pd.DataFrame | None:
    """Memuat perbandingan berbagai konfigurasi fitur dan T dari perbandingan_konfig.csv."""
    return _baca_file("perbandingan_konfig.csv")


@st.cache_data
def muat_kurva_T() -> pd.DataFrame | None:
    """Memuat data kurva akurasi terhadap jumlah stump T dari kurva_T.csv."""
    return _baca_file("kurva_T.csv")
