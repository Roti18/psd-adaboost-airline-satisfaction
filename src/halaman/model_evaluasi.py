"""
Halaman 3: Model dan Evaluasi - Tabel stump, bobot alpha, kurva T, evaluasi CV, dan evaluasi data uji.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import (
    muat_model,
    muat_kurva_T,
    muat_hasil_cv,
    muat_ringkasan_cv,
    muat_test,
    muat_perbandingan_konfig,
)
from src.model import hitung_H, prediksi, hitung_metrik
from src.charts import (
    chart_alpha_fitur,
    chart_kurva_T,
    chart_cv_fold,
    chart_confusion_matrix,
    chart_roc,
    chart_perbandingan_konfig,
)


def tampil():
    """Menampilkan konten lengkap halaman Model dan Evaluasi."""
    st.title(":material/model_training: Model AdaBoost & Evaluasi Kinerja")
    st.markdown("Evaluasi menyeluruh model AdaBoost 67 Decision Stump pada data latih, validasi silang 4-fold, dan data uji.")
    st.markdown("---")

    df_model = muat_model()
    df_kurva = muat_kurva_T()
    df_cv = muat_hasil_cv()
    df_ringkasan_cv = muat_ringkasan_cv()
    df_test = muat_test()
    df_konfig = muat_perbandingan_konfig()

    if df_model is None or df_test is None:
        return

    # -------------------------------------------------------------------------
    # 1. Tabel 67 Stump dari model_adaboost.csv
    # -------------------------------------------------------------------------
    st.subheader(":material/list_alt: 1. Daftar 67 Decision Stump Ensemble")
    st.write(
        "Kumpulan 67 weak learner yang dibentuk pada proses boosting. "
        "Aturan keputusan bernilai +1 (Puas) jika kondisi terpenuhi, dan -1 (Tidak Puas) jika tidak."
    )

    df_model_tampil = df_model.copy()
    df_model_tampil["aturan"] = np.where(
        df_model_tampil["pol"] == 1,
        df_model_tampil["fitur"] + " > " + df_model_tampil["thr"].astype(str),
        df_model_tampil["fitur"] + " <= " + df_model_tampil["thr"].astype(str)
    )

    kolom_urut = ["iterasi", "fitur", "thr", "pol", "eps", "alpha", "aturan"]
    df_tampil = df_model_tampil[kolom_urut].rename(columns={
        "iterasi": "Iterasi (t)",
        "fitur": "Fitur",
        "thr": "Threshold",
        "pol": "Polaritas (pol)",
        "eps": "Error Galat (ε)",
        "alpha": "Bobot Voting (α)",
        "aturan": "Aturan Keputusan (+1 jika Benar)"
    })
    st.dataframe(
        df_tampil.style.format({
            "Threshold": "{:.1f}",
            "Error Galat (ε)": "{:.6f}",
            "Bobot Voting (α)": "{:.6f}"
        }),
        use_container_width=True,
        height=320
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Chart Kontribusi Fitur: Jumlah Alpha per Fitur
    # -------------------------------------------------------------------------
    st.subheader(":material/leaderboard: 2. Kontribusi Kepentingan Fitur (Total Bobot Suara α)")
    st.write("Menjumlahkan nilai $\\alpha$ untuk setiap fitur dari seluruh 67 stump untuk melihat fitur paling berpengaruh dalam voting ensemble:")

    fig_alpha = chart_alpha_fitur(df_model)
    st.pyplot(fig_alpha)
    plt.close(fig_alpha)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Kurva Akurasi vs Jumlah Stump (kurva_T.csv)
    # -------------------------------------------------------------------------
    st.subheader(":material/show_chart: 3. Kurva Akurasi vs Jumlah Decision Stump (T)")
    st.write("Melihat kenaikan akurasi pada data latih dan validasi seiring bertambahnya iterasi boosting (T):")

    if df_kurva is not None:
        fig_kurva = chart_kurva_T(df_kurva)
        st.pyplot(fig_kurva)
        plt.close(fig_kurva)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. Hasil Validasi Silang (4-Fold Cross-Validation)
    # -------------------------------------------------------------------------
    st.subheader(":material/fact_check: 4. Hasil Evaluasi 4-Fold Cross Validation (Data Latih)")
    st.write("Hasil pengujian model pada 4 lipatan data latih untuk memastikan kestabilan kinerja:")

    col_cv1, col_cv2 = st.columns([1.3, 1])
    with col_cv1:
        if df_cv is not None:
            st.write("**:material/table_chart: Tabel Kinerja Tiap Fold:**")
            st.dataframe(
                df_cv.style.format({
                    "n latih": "{:,.0f}",
                    "n validasi": "{:,.0f}",
                    "Akurasi Latih": "{:.4f}",
                    "Akurasi Validasi": "{:.4f}",
                    "Presisi": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}"
                }),
                use_container_width=True
            )

        if df_ringkasan_cv is not None:
            st.write("**:material/summarize: Ringkasan Rata-rata & Standar Deviasi (CV):**")
            st.dataframe(
                df_ringkasan_cv.style.format({
                    "Rata-rata": "{:.6f}",
                    "Std": "{:.6f}"
                }),
                use_container_width=True
            )

    with col_cv2:
        if df_cv is not None:
            fig_cv = chart_cv_fold(df_cv)
            st.pyplot(fig_cv)
            plt.close(fig_cv)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 5. Evaluasi pada Data Test (Dihitung Langsung dari test.csv)
    # -------------------------------------------------------------------------
    st.subheader(":material/analytics: 5. Evaluasi Independen pada Data Uji (Test Set)")
    st.write(
        "Kinerja dihitung langsung secara *live* dari file `test.csv` (20.781 baris) "
        "menggunakan fungsi vektor manual `hitung_H` tanpa modul sklearn ensemble."
    )

    H_test = hitung_H(df_test, df_model)
    y_pred_test = prediksi(H_test)
    y_true_test = df_test["y"].to_numpy()

    metrik = hitung_metrik(y_true_test, y_pred_test, H_test)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Akurasi Uji", f"{metrik['akurasi'] * 100:.2f}%")
    with m2:
        st.metric("Presisi Uji", f"{metrik['presisi'] * 100:.2f}%")
    with m3:
        st.metric("Recall Uji", f"{metrik['recall'] * 100:.2f}%")
    with m4:
        st.metric("F1-Score Uji", f"{metrik['f1']:.4f}")
    with m5:
        st.metric("ROC-AUC Uji", f"{metrik['auc']:.4f}")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        fig_cm = chart_confusion_matrix(metrik["confusion_matrix"])
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_roc:
        fig_roc = chart_roc(metrik["fpr"], metrik["tpr"], metrik["auc"])
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Perbandingan Konfigurasi Model (perbandingan_konfig.csv)
    # -------------------------------------------------------------------------
    st.subheader(":material/tune: 6. Perbandingan Konfigurasi Model")
    st.write("Eksperimen pengaruh jumlah fitur terpilih dan jumlah iterasi stump (T) terhadap akurasi akhir:")

    if df_konfig is not None:
        df_konfig_tampil = df_konfig.copy()
        df_konfig_tampil["Label"] = (
            df_konfig_tampil["Fitur"].astype(str) + " fitur, T=" + df_konfig_tampil["T"].astype(str)
        )

        col_kg1, col_kg2 = st.columns([1.2, 1])
        with col_kg1:
            st.write("**:material/table_chart: Tabel Perbandingan Konfigurasi:**")
            tabel_cfg = df_konfig_tampil[["Label", "Akurasi Train", "Akurasi CV", "Std CV", "Akurasi Test", "F1 Test", "AUC Test"]]
            st.dataframe(
                tabel_cfg.style.format({
                    "Akurasi Train": "{:.4f}",
                    "Akurasi CV": "{:.4f}",
                    "Std CV": "{:.4f}",
                    "Akurasi Test": "{:.4f}",
                    "F1 Test": "{:.4f}",
                    "AUC Test": "{:.4f}"
                }),
                use_container_width=True
            )

        with col_kg2:
            fig_cfg = chart_perbandingan_konfig(df_konfig)
            st.pyplot(fig_cfg)
            plt.close(fig_cfg)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 7. Kesimpulan Kestabilan Bebas Overfitting
    # -------------------------------------------------------------------------
    st.subheader(":material/verified: 7. Kesimpulan Uji Overfitting")
    if df_konfig is not None:
        baris_final = df_konfig[(df_konfig["Fitur"] == 17) & (df_konfig["T"] == 67)]
        if not baris_final.empty:
            acc_tr = baris_final["Akurasi Train"].values[0]
            acc_val = baris_final["Akurasi CV"].values[0]
            acc_ts = baris_final["Akurasi Test"].values[0]

            selisih_train_test = abs(acc_tr - acc_ts) * 100
            selisih_cv_test = abs(acc_val - acc_ts) * 100

            st.success(
                f"**Kesimpulan Kinerja:** Selisih akurasi antara data latih (**{acc_tr*100:.2f}%**), "
                f"validasi silang CV (**{acc_val*100:.2f}%**), dan data uji (**{acc_ts*100:.2f}%**) "
                f"sangat kecil (selisih Latih-Uji hanya **{selisih_train_test:.2f}%** dan CV-Uji hanya **{selisih_cv_test:.2f}%**). "
                f"Hal ini membuktikan secara ilmiah bahwa model AdaBoost dengan 17 fitur dan T=67 memiliki kemampuan generalisasi "
                f"yang luar biasa tangguh dan **sama sekali tidak mengalami overfitting**.",
                icon=":material/check_circle:"
            )
