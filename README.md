# Klasifikasi Kepuasan Penumpang Maskapai Penerbangan dengan AdaBoost

Aplikasi web berbasis Streamlit untuk klasifikasi tingkat kepuasan penumpang maskapai penerbangan menggunakan algoritma ensemble AdaBoost berbasis 67 Decision Stump. Implementasi algoritma AdaBoost dan fungsi voting ensemble dibangun secara manual menggunakan NumPy murni tanpa modul ensemble dari library scikit-learn.

## Struktur Direktori Proyek

```text
web/
├── data/
│   ├── hasil_cv.csv
│   ├── korelasi.csv
│   ├── kurva_T.csv
│   ├── model_adaboost.csv
│   ├── perbandingan_konfig.csv
│   ├── ringkasan_cv.csv
│   ├── test.csv
│   └── train.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── charts.py
│   └── halaman/
│       ├── __init__.py
│       ├── beranda.py
│       ├── eda.py
│       ├── model_evaluasi.py
│       └── prediksi.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Ringkasan Dataset dan Preprocessing

- Total Data: 103.904 baris data penumpang penerbangan.
- Pembagian Data: 80% data latih (83.123 baris) dan 20% data uji (20.781 baris) dengan stratified split.
- Target Klasifikasi: Biner, yaitu Puas (+1) dan Netral atau Tidak Puas (-1).
- Penanganan Missing Value: 310 nilai kosong pada kolom Arrival_Delay_in_Minutes diimputasi menggunakan nilai median data latih (kolom ini akhirnya tidak terpilih sebagai fitur model karena korelasinya rendah).
- Seleksi Fitur: Dari 22 fitur asli, dipilih 17 fitur dengan nilai mutlak korelasi Pearson (|r|) tertinggi terhadap variabel target pada data latih. 5 fitur dengan korelasi terendah dieliminasi.
- Validasi Silang: 4-Fold Cross Validation (fold bergilir berdasarkan urutan data, proporsi kelas tiap fold seimbang) diterapkan pada 83.123 data latih.

## 17 Fitur Terpilih

1. Online_boarding (Rating 0 sampai 5)
2. Class (Kategori: Eco = 1, Eco Plus = 2, Business = 3)
3. Type_of_Travel (Kategori: Personal Travel = 0, Business travel = 1)
4. Inflight_entertainment (Rating 0 sampai 5)
5. Seat_comfort (Rating 1 sampai 5)
6. On-board_service (Rating 0 sampai 5)
7. Leg_room_service (Rating 0 sampai 5)
8. Cleanliness (Rating 0 sampai 5)
9. Flight_Distance (Numerik kontinu, mil)
10. Inflight_wifi_service (Rating 0 sampai 5)
11. Baggage_handling (Rating 1 sampai 5)
12. Inflight_service (Rating 0 sampai 5)
13. Checkin_service (Rating 1 sampai 5)
14. Food_and_drink (Rating 0 sampai 5)
15. Customer_Type (Kategori: disloyal Customer = 0, Loyal Customer = 1)
16. Ease_of_Online_booking (Rating 0 sampai 5)
17. Age (Numerik kontinu, tahun)

## Logika Model AdaBoost dan Decision Stump

- Decision Stump: Model klasifikasi lemah (weak learner) berupa pohon keputusan dengan kedalaman 1 tingkat. Fungsi prediksi satu stump dirumuskan sebagai:
  h(x) = pol * (+1 jika x[fitur] > thr, selain itu -1)
  dengan pol bernilai +1 atau -1.
- Bobot Suara Stump (Alpha):
  alpha_t = 0.5 * ln((1 - eps_t) / eps_t)
  dengan eps_t adalah weighted error rate pada iterasi ke-t.
- Skor Ensemble H(x):
  H(x) = sum(alpha_t * h_t(x)) untuk t = 1 sampai T (T = 67).
- Keputusan Akhir:
  Prediksi = +1 (Puas) jika H(x) > 0, dan Prediksi = -1 (Tidak Puas) jika H(x) <= 0.
- Estimasi Keyakinan Model:
  P(Puas) = 1 / (1 + exp(-2 * H(x)))
  sebagai representasi margin keputusan voting ensemble.

## Hasil Evaluasi Model (Konfigurasi Terbaik: 17 Fitur, T = 67)

- Akurasi Data Latih: 92.16%
- Akurasi Validasi Silang (Rata-rata 4-Fold CV): 92.11%
- Akurasi Data Uji: 92.06%
- Presisi Data Uji (Kelas Puas): 91.35%
- Recall Data Uji (Kelas Puas): 90.21%
- F1-Score Data Uji: 0.9078
- ROC-AUC Data Uji: 0.9734
- Kestabilan Model: Selisih antara akurasi latih dan uji hanya 0.10%, serta selisih validasi silang dan uji hanya 0.05%. Kinerja yang konsisten pada data latih, validasi, dan data uji menunjukkan tidak ada indikasi overfitting.

## Struktur Modul Kode

- main.py: Titik masuk utama aplikasi Streamlit, mengatur konfigurasi halaman dan navigasi sidebar.
- src/config.py: Menyimpan konstanta direktori data, daftar fitur, palet warna, dan pemetaan encoding.
- src/data_loader.py: Memuat file CSV dengan dekorator caching Streamlit dan penanganan berkas hilang.
- src/model.py: Fungsi inti AdaBoost (hitung_H, prediksi, hitung_keyakinan, kontribusi_stump, hitung_metrik) berbasis NumPy dan Pandas tanpa kode Streamlit.
- src/charts.py: Fungsi pembuatan grafik Matplotlib murni yang mengembalikan objek figure.
- src/halaman/: Modul visualisasi antarmuka per halaman (beranda.py, eda.py, model_evaluasi.py, prediksi.py).

## Panduan Instalasi dan Penggunaan

1. Clone repositori:
```bash
git clone https://github.com/Roti18/psd-adaboost-airline-satisfaction.git
cd psd-adaboost-airline-satisfaction
```

2. Buat dan aktifkan virtual environment:
```bash
python -m venv .venv
# Di Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Di Linux atau macOS:
source .venv/bin/activate
```

3. Pasang dependensi yang dibutuhkan:
```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi Streamlit:
```bash
streamlit run main.py
```
