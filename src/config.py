"""
Konfigurasi global, path data, konstanta warna, pemetaan encoding, dan daftar fitur.
"""

from pathlib import Path

# Path folder data relatif terhadap root proyek
FOLDER_DATA = Path(__file__).resolve().parent.parent / "data"

# Palet warna visualisasi: Hijau = Puas (+1), Merah = Tidak Puas (-1)
WARNA_PUAS = "#27ae60"        # Hijau cerah untuk Puas
WARNA_TIDAK_PUAS = "#e74c3c"  # Merah cerah untuk Netral/Tidak Puas
WARNA_UTAMA = "#2980b9"       # Biru untuk elemen netral atau fitur terpilih
WARNA_ABU = "#95a5a6"         # Abu-abu untuk fitur dieliminasi
WARNA_LATAR = "#f8f9fa"

# Label kelas
LABEL_PUAS = "Puas (Satisfied / +1)"
LABEL_TIDAK_PUAS = "Netral / Tidak Puas (-1)"

# 17 Fitur terpilih sesuai urutan baku dari train.csv
DAFTAR_FITUR = [
    "Online_boarding",
    "Class",
    "Type_of_Travel",
    "Inflight_entertainment",
    "Seat_comfort",
    "On-board_service",
    "Leg_room_service",
    "Cleanliness",
    "Flight_Distance",
    "Inflight_wifi_service",
    "Baggage_handling",
    "Inflight_service",
    "Checkin_service",
    "Food_and_drink",
    "Customer_Type",
    "Ease_of_Online_booking",
    "Age",
]

# Pemetaan nilai kategori input ke numerik
PETA_CLASS = {"Eco": 1, "Eco Plus": 2, "Business": 3}
PETA_TRAVEL = {"Personal Travel": 0, "Business travel": 1}
PETA_CUSTOMER = {"disloyal Customer": 0, "Loyal Customer": 1}
