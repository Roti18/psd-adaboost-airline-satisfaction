"""
Halaman 2: Analisis Eksplorasi Data (EDA) - Visualisasi data latih, korelasi fitur, dan heatmap.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import DAFTAR_FITUR
from src.data_loader import muat_train, muat_test, muat_korelasi
from src.charts import (
    chart_distribusi_target,
    chart_korelasi_fitur,
    chart_persen_puas,
    chart_heatmap_korelasi,
)


def tampil():
    """Menampilkan konten lengkap halaman EDA."""
    st.title(":material/bar_chart: Analisis Eksplorasi Data (EDA)")
    st.markdown("Eksplorasi karakteristik data latih (`train.csv`) dan relevansi fitur terhadap kepuasan penumpang.")
    st.markdown("---")

    df_train = muat_train()
    df_test = muat_test()
    df_korelasi = muat_korelasi()

    if df_train is None:
        return

    # Pilihan menu tab analisis dengan Material Icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/pie_chart: Distribusi Target",
        ":material/bar_chart: Korelasi Fitur vs Target",
        ":material/trending_up: Persen Puas per Fitur",
        ":material/table_chart: Statistik Agregat Fitur",
        ":material/grid_on: Heatmap Korelasi Antar Fitur",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Distribusi Kelas Target Train dan Test
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader(":material/pie_chart: Distribusi Kelas Target: Data Latih vs Data Uji")
        st.write("Perbandingan proporsi penumpang yang Puas (+1) dan Netral/Tidak Puas (-1).")

        train_counts = df_train["y"].value_counts()
        test_counts = df_test["y"].value_counts() if df_test is not None else pd.Series({1: 0, -1: 0})

        n_train_puas = int(train_counts.get(1, 0))
        n_train_tidak = int(train_counts.get(-1, 0))
        n_test_puas = int(test_counts.get(1, 0))
        n_test_tidak = int(test_counts.get(-1, 0))

        fig1 = chart_distribusi_target(n_train_puas, n_train_tidak, n_test_puas, n_test_tidak)
        st.pyplot(fig1)
        plt.close(fig1)

    # -------------------------------------------------------------------------
    # TAB 2: Korelasi Tiap Fitur dengan Target
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader(":material/bar_chart: Korelasi Pearson Fitur terhadap Target")
        st.write(
            "Visualisasi nilai mutlak korelasi Pearson ($|r|$) dari seluruh 22 fitur asli. "
            "17 fitur teratas (berwarna biru) dipilih untuk pemodelan, sedangkan 5 fitur terbawah (abu-abu) dieliminasi."
        )

        if df_korelasi is not None:
            fig2 = chart_korelasi_fitur(df_korelasi)
            st.pyplot(fig2)
            plt.close(fig2)

            st.write("**:material/table_chart: Tabel Lengkap Korelasi Fitur:**")
            tabel_tampil = df_korelasi.sort_values(by="peringkat").copy()
            tabel_tampil["Status"] = np.where(tabel_tampil["peringkat"] <= 17, "Terpilih (Top 17)", "Dieliminasi")
            tabel_tampil.columns = ["Korelasi (r)", "Nilai Mutlak (|r|)", "Dipakai", "Peringkat", "Status"]
            st.dataframe(
                tabel_tampil.style.format({
                    "Korelasi (r)": "{:.6f}",
                    "Nilai Mutlak (|r|)": "{:.6f}",
                    "Peringkat": "{:.0f}"
                }),
                use_container_width=True
            )

    # -------------------------------------------------------------------------
    # TAB 3: Persen Puas per Nilai Fitur
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader(":material/trending_up: Persentase Penumpang Puas Berdasarkan Nilai Fitur")
        fitur_pilihan = st.selectbox("Pilih salah satu dari 17 fitur terpilih:", DAFTAR_FITUR, index=0)

        rata_puas_global = (df_train["y"] == 1).mean() * 100

        # Jika fitur memiliki <= 10 nilai unik, gunakan nilai diskrit aslinya
        if df_train[fitur_pilihan].nunique() <= 10:
            df_grup = df_train.groupby(fitur_pilihan, observed=False)["y"].agg(
                total="count",
                puas=lambda s: (s == 1).sum(),
                persen=lambda s: (s == 1).mean() * 100
            ).reset_index()

            if fitur_pilihan == "Class":
                label_peta = {1: "1 (Eco)", 2: "2 (Eco Plus)", 3: "3 (Business)"}
                df_grup["label_x"] = df_grup[fitur_pilihan].map(label_peta)
            elif fitur_pilihan == "Type_of_Travel":
                label_peta = {0: "0 (Personal Travel)", 1: "1 (Business travel)"}
                df_grup["label_x"] = df_grup[fitur_pilihan].map(label_peta)
            elif fitur_pilihan == "Customer_Type":
                label_peta = {0: "0 (Disloyal)", 1: "1 (Loyal)"}
                df_grup["label_x"] = df_grup[fitur_pilihan].map(label_peta)
            else:
                df_grup["label_x"] = df_grup[fitur_pilihan].astype(str)
        else:
            # Jika fitur kontinu (> 10 nilai unik, Flight_Distance & Age), bagi 5 kuantil
            bins = pd.qcut(df_train[fitur_pilihan], q=5, duplicates="drop")
            df_grup = df_train.groupby(bins, observed=False)["y"].agg(
                total="count",
                puas=lambda s: (s == 1).sum(),
                persen=lambda s: (s == 1).mean() * 100
            ).reset_index()
            df_grup["label_x"] = df_grup[fitur_pilihan].apply(lambda interval: f"{interval.left:.0f} - {interval.right:.0f}")

        fig3 = chart_persen_puas(df_grup, rata_puas_global, fitur_pilihan)
        st.pyplot(fig3)
        plt.close(fig3)

    # -------------------------------------------------------------------------
    # TAB 4: Tabel Statistik Agregat Fitur
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader(":material/table_chart: Tabel Statistik Agregat 17 Fitur Terpilih (Data Latih)")
        st.write("Ringkasan statistik deskriptif untuk setiap fitur numerik pada 83.123 baris data latih:")

        df_stats = df_train[DAFTAR_FITUR].describe().T[["count", "min", "max", "mean", "50%", "std"]].copy()
        df_stats.columns = ["Jumlah (Count)", "Minimum", "Maksimum", "Rata-rata (Mean)", "Median", "Standar Deviasi (Std)"]
        st.dataframe(
            df_stats.style.format({
                "Jumlah (Count)": "{:,.0f}",
                "Minimum": "{:.2f}",
                "Maksimum": "{:.2f}",
                "Rata-rata (Mean)": "{:.2f}",
                "Median": "{:.2f}",
                "Standar Deviasi (Std)": "{:.2f}"
            }),
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # TAB 5: Heatmap Korelasi Antar 17 Fitur
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader(":material/grid_on: Matriks Korelasi Pearson Antar 17 Fitur Terpilih")
        st.write("Melihat hubungan linier dan potensi multikolinearitas antar fitur pada data latih:")

        corr_matrix = df_train[DAFTAR_FITUR].corr()
        fig5 = chart_heatmap_korelasi(corr_matrix, DAFTAR_FITUR)
        st.pyplot(fig5)
        plt.close(fig5)
