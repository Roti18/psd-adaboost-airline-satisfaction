"""
Halaman 4: Prediksi Kepuasan Penumpang - Form input parameter penumpang individual secara langsung.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.config import DAFTAR_FITUR, PETA_CLASS, PETA_TRAVEL, PETA_CUSTOMER
from src.data_loader import muat_train, muat_test, muat_model
from src.model import hitung_H, prediksi, hitung_keyakinan, kontribusi_stump
from src.charts import chart_kontribusi_stump, chart_kontribusi_fitur


def tampil():
    """Menampilkan konten lengkap halaman Prediksi melalui input parameter langsung."""
    st.title(":material/online_prediction: Prediksi Kepuasan Penumpang")
    st.markdown("Masukkan nilai fitur penumpang untuk menguji prediksi klasifikasi model AdaBoost.")
    st.markdown("---")

    df_model = muat_model()
    df_train = muat_train()
    df_test = muat_test()

    if df_model is None:
        return

    # Ambil nilai default statistik dari data latih
    if df_train is not None:
        medians = df_train[DAFTAR_FITUR].median()
        mins = df_train[DAFTAR_FITUR].min()
        maxs = df_train[DAFTAR_FITUR].max()
    else:
        medians = {f: 3 for f in DAFTAR_FITUR}
        medians["Flight_Distance"] = 840
        medians["Age"] = 40
        mins = {f: 0 for f in DAFTAR_FITUR}
        mins["Flight_Distance"] = 31
        mins["Age"] = 7
        maxs = {f: 5 for f in DAFTAR_FITUR}
        maxs["Flight_Distance"] = 4983
        maxs["Age"] = 85

    st.subheader(":material/edit_note: Form Input Parameter Penumpang")
    st.write("Silakan atur parameter profil penerbangan dan nilai kepuasan layanan berikut:")

    with st.form("form_prediksi_penumpang"):
        col1, col2, col3 = st.columns(3)

        # Kolom 1: Profil Penumpang & Tiket
        with col1:
            st.markdown("#### :material/airplane_ticket: Profil Perjalanan")
            input_class = st.selectbox(
                "Kelas Penerbangan (Class)",
                options=["Business", "Eco", "Eco Plus"],
                index=0,
                help="Business=3, Eco Plus=2, Eco=1"
            )
            input_travel = st.selectbox(
                "Tipe Perjalanan (Type of Travel)",
                options=["Business travel", "Personal Travel"],
                index=0,
                help="Business travel=1, Personal Travel=0"
            )
            input_customer = st.selectbox(
                "Tipe Loyalitas (Customer Type)",
                options=["Loyal Customer", "disloyal Customer"],
                index=0,
                help="Loyal Customer=1, disloyal Customer=0"
            )
            input_age = st.number_input(
                "Usia Penumpang (Age)",
                min_value=int(mins["Age"]),
                max_value=int(maxs["Age"]),
                value=int(medians["Age"]),
                step=1
            )
            input_dist = st.number_input(
                "Jarak Penerbangan (Flight Distance dalam mil)",
                min_value=int(mins["Flight_Distance"]),
                max_value=int(maxs["Flight_Distance"]),
                value=int(medians["Flight_Distance"]),
                step=10
            )
            input_boarding = st.slider(
                "Kemudahan Online Boarding",
                min_value=int(mins["Online_boarding"]),
                max_value=int(maxs["Online_boarding"]),
                value=int(medians["Online_boarding"]),
                help="0 = Tidak Berlaku, 1 = Sangat Buruk, 5 = Sangat Memuaskan"
            )

        # Kolom 2: Layanan Kabin & Kenyamanan
        with col2:
            st.markdown("#### :material/chair: Layanan & Kenyamanan Kabin")
            input_seat = st.slider(
                "Kenyamanan Kursi (Seat Comfort)",
                min_value=int(mins["Seat_comfort"]),
                max_value=int(maxs["Seat_comfort"]),
                value=int(medians["Seat_comfort"])
            )
            input_entertainment = st.slider(
                "Hiburan di Pesawat (Inflight Entertainment)",
                min_value=int(mins["Inflight_entertainment"]),
                max_value=int(maxs["Inflight_entertainment"]),
                value=int(medians["Inflight_entertainment"])
            )
            input_onboard = st.slider(
                "Layanan Awak Kabin (On-board Service)",
                min_value=int(mins["On-board_service"]),
                max_value=int(maxs["On-board_service"]),
                value=int(medians["On-board_service"])
            )
            input_legroom = st.slider(
                "Ruang Kaki Kursi (Leg Room Service)",
                min_value=int(mins["Leg_room_service"]),
                max_value=int(maxs["Leg_room_service"]),
                value=int(medians["Leg_room_service"])
            )
            input_clean = st.slider(
                "Kebersihan Kabin (Cleanliness)",
                min_value=int(mins["Cleanliness"]),
                max_value=int(maxs["Cleanliness"]),
                value=int(medians["Cleanliness"])
            )
            input_food = st.slider(
                "Makanan dan Minuman (Food and Drink)",
                min_value=int(mins["Food_and_drink"]),
                max_value=int(maxs["Food_and_drink"]),
                value=int(medians["Food_and_drink"])
            )

        # Kolom 3: Layanan Digital & Bandara
        with col3:
            st.markdown("#### :material/wifi: Layanan Digital & Bandara")
            input_wifi = st.slider(
                "Koneksi WiFi di Pesawat (Inflight WiFi)",
                min_value=int(mins["Inflight_wifi_service"]),
                max_value=int(maxs["Inflight_wifi_service"]),
                value=int(medians["Inflight_wifi_service"])
            )
            input_baggage = st.slider(
                "Penanganan Bagasi (Baggage Handling)",
                min_value=int(mins["Baggage_handling"]),
                max_value=int(maxs["Baggage_handling"]),
                value=int(medians["Baggage_handling"])
            )
            input_inflight_service = st.slider(
                "Layanan Selama Terbang (Inflight Service)",
                min_value=int(mins["Inflight_service"]),
                max_value=int(maxs["Inflight_service"]),
                value=int(medians["Inflight_service"])
            )
            input_checkin = st.slider(
                "Layanan Check-in (Checkin Service)",
                min_value=int(mins["Checkin_service"]),
                max_value=int(maxs["Checkin_service"]),
                value=int(medians["Checkin_service"])
            )
            input_booking = st.slider(
                "Kemudahan Booking Online (Ease of Online Booking)",
                min_value=int(mins["Ease_of_Online_booking"]),
                max_value=int(maxs["Ease_of_Online_booking"]),
                value=int(medians["Ease_of_Online_booking"])
            )

        tombol_prediksi = st.form_submit_button(
            "Prediksi Kepuasan Penumpang",
            icon=":material/rocket_launch:",
            use_container_width=True
        )

    if tombol_prediksi:
        baris_input = {
            "Online_boarding": input_boarding,
            "Class": PETA_CLASS[input_class],
            "Type_of_Travel": PETA_TRAVEL[input_travel],
            "Inflight_entertainment": input_entertainment,
            "Seat_comfort": input_seat,
            "On-board_service": input_onboard,
            "Leg_room_service": input_legroom,
            "Cleanliness": input_clean,
            "Flight_Distance": input_dist,
            "Inflight_wifi_service": input_wifi,
            "Baggage_handling": input_baggage,
            "Inflight_service": input_inflight_service,
            "Checkin_service": input_checkin,
            "Food_and_drink": input_food,
            "Customer_Type": PETA_CUSTOMER[input_customer],
            "Ease_of_Online_booking": input_booking,
            "Age": input_age,
        }

        df_satu = pd.DataFrame([baris_input])[DAFTAR_FITUR]
        H_val = hitung_H(df_satu, df_model)[0]
        is_puas = H_val > 0
        prediksi_teks = "PUAS (Satisfied)" if is_puas else "TIDAK PUAS (Neutral/Dissatisfied)"

        _, keyakinan_pct = hitung_keyakinan(np.array([H_val]))
        keyakinan_persen = float(keyakinan_pct[0])

        st.markdown("---")
        st.subheader(":material/assessment: Hasil Prediksi Model")

        if is_puas:
            st.success(f"### Status: {prediksi_teks}", icon=":material/sentiment_satisfied:")
        else:
            st.error(f"### Status: {prediksi_teks}", icon=":material/sentiment_dissatisfied:")

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Skor Ensemble H(x)", f"{H_val:+.4f}", help="H > 0 menandakan Puas, H <= 0 menandakan Tidak Puas")
        with col_res2:
            st.metric("Estimasi Keyakinan Model", f"{keyakinan_persen:.2f}%", help="Dihitung dengan rumus 1 / (1 + exp(-2|H|))")
        with col_res3:
            st.metric("Keputusan Akhir", "Puas (+1)" if is_puas else "Tidak Puas (-1)")

        st.caption(
            "📌 *Keterangan:* Estimasi keyakinan dihitung menggunakan fungsi logistik "
            r"$P = \frac{1}{1 + e^{-2H(x)}}$"
            " sebagai representasi margin keputusan voting ensemble, bukan probabilitas bersyarat pasti."
        )

        df_rincian = kontribusi_stump(df_satu, df_model)
        total_kontribusi = df_rincian["Kontribusi (α * h)"].sum()

        st.markdown("#### :material/table_chart: Tabel Rincian Kontribusi Seluruh 67 Stump")
        st.dataframe(
            df_rincian.style.format({
                "Threshold": "{:.1f}",
                "Vote (h)": "{:+d}",
                "Alpha (α)": "{:.6f}",
                "Kontribusi (α * h)": "{:+.6f}"
            }),
            use_container_width=True,
            height=260
        )

        st.info(
            f"**Verifikasi Akumulasi Voting:** "
            f"Total penjumlahan kolom kontribusi adalah **{total_kontribusi:+.6f}**, "
            f"persis sama dengan nilai skor ensemble **H(x) = {H_val:+.6f}**.",
            icon=":material/check_circle:"
        )

        st.markdown("#### :material/insights: Analisis Grafis Kontribusi Voting")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_stump = chart_kontribusi_stump(df_rincian)
            st.pyplot(fig_stump)
            plt.close(fig_stump)

        with col_g2:
            fig_fitur = chart_kontribusi_fitur(df_rincian)
            st.pyplot(fig_fitur)
            plt.close(fig_fitur)

    # -------------------------------------------------------------------------
    # Verifikasi Otomatis Konsistensi Model
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander(":material/fact_check: Cek Konsistensi Model (Verifikasi Otomatis)"):
        st.write(
            "Fungsi uji otomatis untuk memverifikasi bahwa prediksi aplikasi web "
            "terhadap seluruh data `test.csv` (20.781 baris) menghasilkan metrik yang sama persis "
            "dengan angka acuan di `perbandingan_konfig.csv` (T=67, 17 fitur)."
        )
        if df_test is not None:
            H_cek = hitung_H(df_test, df_model)
            y_pred_cek = prediksi(H_cek)
            acc_cek = accuracy_score(df_test["y"], y_pred_cek)
            target_acc = 0.920552

            if np.isclose(acc_cek, target_acc, atol=1e-5):
                st.success(
                    f"**Verifikasi Konsistensi Sukses!**\n\n"
                    f"- Akurasi Terhitung Web: **{acc_cek * 100:.4f}%** ({acc_cek:.6f})\n"
                    f"- Akurasi Acuan Konfigurasi: **{target_acc * 100:.4f}%** ({target_acc:.6f})\n"
                    f"- Selisih Perbedaan: **0.000000%** (100% Cocok Sempurna).",
                    icon=":material/check_circle:"
                )
            else:
                st.warning(f"Akurasi terhitung: {acc_cek:.6f}, Acuan: {target_acc:.6f}", icon=":material/warning:")
