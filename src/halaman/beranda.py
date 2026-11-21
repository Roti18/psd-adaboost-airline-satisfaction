"""
Halaman 1: Beranda / Tentang - Menampilkan ringkasan dataset, alur preprocessing, dan teori model.
"""

import streamlit as st
from src.data_loader import muat_perbandingan_konfig


def tampil():
    """Menampilkan konten lengkap halaman Beranda."""
    st.title(":material/flight: Prediksi Kepuasan Penumpang Maskapai Penerbangan")
    st.markdown(
        "**Tugas Kuliah Penambangan Sains Data:** Klasifikasi kepuasan penumpang maskapai "
        "menggunakan algoritma ensemble **AdaBoost** berbasis **Decision Stump**."
    )
    st.markdown("---")

    # Bagian Tujuan Proyek
    st.subheader(":material/track_changes: Tujuan Proyek")
    st.write(
        "Proyek ini bertujuan mengklasifikasikan tingkat kepuasan penumpang maskapai penerbangan ke dalam dua kategori: "
        "**Puas (*Satisfied* / +1)** atau **Netral / Tidak Puas (*Neutral or Dissatisfied* / -1)**. "
        "Dengan mengetahui faktor penentu kepuasan penumpang, maskapai dapat memprioritaskan perbaikan "
        "pada layanan kabin, layanan digital, dan operasional penerbangan secara terarah."
    )

    # Ringkasan Dataset dalam st.metric
    st.subheader(":material/dataset: Ringkasan Dataset")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Data Baris", value="103.904")
    with col2:
        st.metric(label="Fitur Asli", value="22 Kolom")
    with col3:
        st.metric(label="Fitur Terpilih", value="17 Fitur")
    with col4:
        st.metric(label="Split Train / Test", value="83.123 / 20.781", help="Rasio 80:20 Stratified Split")
    with col5:
        st.metric(label="Missing Value", value="310 Nilai", help="Diimputasi median train pada Arrival_Delay_in_Minutes")

    st.markdown("---")

    # Alur Preprocessing Singkat
    st.subheader(":material/sync_alt: Alur Preprocessing Data")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(
            "**1. Encoding Kategorikal & Split**\n"
            "- `Class`: Eco = 1, Eco Plus = 2, Business = 3\n"
            "- `Type_of_Travel`: Personal = 0, Business = 1\n"
            "- `Customer_Type`: Disloyal = 0, Loyal = 1\n"
            "- Pembagian data 80:20 secara stratified."
        )
    with col_p2:
        st.markdown(
            "**2. Imputasi & Seleksi Fitur**\n"
            "- Imputasi 310 nilai kosong di `Arrival_Delay_in_Minutes` dengan median data latih.\n"
            "- Seleksi 17 fitur dengan korelasi Pearson ($|r|$) tertinggi terhadap target $y$ pada data latih.\n"
            "- 5 fitur dengan korelasi terendah dieliminasi."
        )
    with col_p3:
        st.markdown(
            "**3. Validasi Silang (Cross-Validation)**\n"
            "- Pembuatan 4 fold stratified pada 83.123 data latih.\n"
            "- Setiap fold berisi ~20.781 data validasi.\n"
            "- Evaluasi kestabilan metrik akurasi, presisi, recall, dan F1."
        )

    st.markdown("---")

    # Penjelasan Konsep Model: Decision Stump dan AdaBoost
    st.subheader(":material/psychology: Dasar Teori: Decision Stump & AdaBoost")
    st.markdown(
        "Model yang digunakan adalah algoritma ensemble buatan tangan (manual dengan NumPy murni tanpa sklearn) "
        "terdiri dari kumpulan **Decision Stump**:"
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            "#### Apa itu Decision Stump?\n"
            "Decision stump adalah pohon keputusan (*decision tree*) sederhana dengan kedalaman tepat 1 tingkat "
            "(memiliki 1 akar dan 2 daun). Stump hanya menguji **satu fitur** terhadap suatu nilai ambang batas (*threshold* $\\theta$):\n"
        )
        st.latex(r"h(x) = \text{pol} \times \begin{cases} +1, & \text{jika } x[\text{fitur}] > \text{thr} \\ -1, & \text{jika } x[\text{fitur}] \le \text{thr} \end{cases}")
        st.markdown("di mana $\\text{pol} \\in \\{+1, -1\\}$ adalah arah polaritas keputusan.")

    with col_t2:
        st.markdown(
            "#### Apa itu AdaBoost (Adaptive Boosting)?\n"
            "AdaBoost melatih rangkaian $T$ decision stump secara sekuensial:\n"
            "1. **Mencari Stump Terbaik:** Pada tiap iterasi $t$, cari stump $h_t$ yang meminimalkan galat terbobot $\\epsilon_t$.\n"
            "2. **Menghitung Bobot Suara:** Hitung kekuatan voting $\\alpha_t$:\n"
        )
        st.latex(r"\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)")
        st.markdown(
            "3. **Memperbarui Bobot Sampel:** Bobot sampel yang salah diprediksi dinaikkan, lalu dinormalisasi.\n"
            "4. **Prediksi Akhir Ensemble:** Voting terbobot dari seluruh stump:"
        )
        st.latex(r"H(x) = \sum_{t=1}^T \alpha_t h_t(x), \quad \hat{y} = \text{sign}(H(x)) = \begin{cases} +1 \text{ (Puas)}, & H(x) > 0 \\ -1 \text{ (Tidak Puas)}, & H(x) \le 0 \end{cases}")

    st.markdown("---")

    # Ringkasan Hasil Akhir Model Terpilih (T=67, 17 Fitur)
    st.subheader(":material/emoji_events: Ringkasan Hasil Evaluasi Model Akhir (T = 67, 17 Fitur)")
    df_konfig = muat_perbandingan_konfig()
    if df_konfig is not None:
        baris_terbaik = df_konfig[(df_konfig["Fitur"] == 17) & (df_konfig["T"] == 67)]
        if not baris_terbaik.empty:
            acc_test = baris_terbaik["Akurasi Test"].values[0]
            f1_test = baris_terbaik["F1 Test"].values[0]
            auc_test = baris_terbaik["AUC Test"].values[0]

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Akurasi Data Uji (Test Accuracy)", f"{acc_test * 100:.2f}%", help="Persentase tebakan benar pada 20.781 data uji")
            with col_r2:
                st.metric("F1-Score Data Uji", f"{f1_test:.4f}", help="Harmonic mean dari presisi dan recall kelas Puas")
            with col_r3:
                st.metric("ROC-AUC Data Uji", f"{auc_test:.4f}", help="Kemampuan diskriminasi model membedakan penumpang puas vs tidak puas")
        else:
            st.info("Baris konfigurasi Fitur=17, T=67 tidak ditemukan di perbandingan_konfig.csv.", icon=":material/info:")
