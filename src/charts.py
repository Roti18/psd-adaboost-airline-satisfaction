"""
Kumpulan fungsi visualisasi grafik matplotlib murni tanpa dependensi Streamlit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import WARNA_PUAS, WARNA_TIDAK_PUAS, WARNA_UTAMA, WARNA_ABU

# Konfigurasi gaya visualisasi matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def chart_distribusi_target(
    n_train_puas: int,
    n_train_tidak: int,
    n_test_puas: int,
    n_test_tidak: int,
) -> plt.Figure:
    """Membuat bar chart perbandingan distribusi kelas target pada data train dan test."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Puas (Satisfied / +1)", "Netral/Tidak Puas (-1)"]
    x = np.arange(len(labels))
    width = 0.35

    bar_train = ax.bar(x - width/2, [n_train_puas, n_train_tidak], width, label="Data Latih (Train)", color="#3498db")
    bar_test = ax.bar(x + width/2, [n_test_puas, n_test_tidak], width, label="Data Uji (Test)", color="#e67e22")

    ax.set_ylabel("Jumlah Penumpang")
    ax.set_title("Distribusi Kelas Target pada Data Latih (80%) dan Data Uji (20%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    total_train = n_train_puas + n_train_tidak
    total_test = n_test_puas + n_test_tidak

    for bar, tot in zip(bar_train, [total_train, total_train]):
        h = bar.get_height()
        pct = (h / tot) * 100 if tot > 0 else 0
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    for bar, tot in zip(bar_test, [total_test, total_test]):
        h = bar.get_height()
        pct = (h / tot) * 100 if tot > 0 else 0
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylim(0, max(n_train_tidak, n_train_puas) * 1.18)
    fig.tight_layout()
    return fig


def chart_korelasi_fitur(df_korelasi: pd.DataFrame) -> plt.Figure:
    """Membuat horizontal bar chart korelasi Pearson fitur terhadap target."""
    df_urut = df_korelasi.sort_values(by="abs_r", ascending=True)
    warna_bar = [WARNA_UTAMA if p <= 17 else WARNA_ABU for p in df_urut["peringkat"]]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars = ax.barh(df_urut.index, df_urut["abs_r"], color=warna_bar)

    for bar, (_, row) in zip(bars, df_urut.iterrows()):
        w = bar.get_width()
        tanda = "+" if row["r"] >= 0 else "-"
        ax.annotate(
            f"|r|={w:.4f} ({tanda})",
            xy=(w, bar.get_y() + bar.get_height()/2),
            xytext=(5, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8.5
        )

    ax.set_xlabel("Nilai Mutlak Korelasi Pearson (|r|)", fontsize=10)
    ax.set_title("Peringkat Korelasi Fitur terhadap Kepuasan (Biru: Terpilih, Abu-abu: Dieliminasi)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, df_urut["abs_r"].max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_persen_puas(
    df_grup: pd.DataFrame,
    rata_puas_global: float,
    nama_fitur: str,
) -> plt.Figure:
    """Membuat bar chart persentase puas per nilai fitur dengan garis rata-rata global."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    warna_batang = [WARNA_PUAS if p >= rata_puas_global else WARNA_TIDAK_PUAS for p in df_grup["persen"]]
    bars = ax.bar(df_grup["label_x"], df_grup["persen"], color=warna_batang, width=0.55, edgecolor="black", linewidth=0.5)

    ax.axhline(rata_puas_global, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Rata-rata Puas Keseluruhan ({rata_puas_global:.2f}%)")

    for bar, total_val, pct_val in zip(bars, df_grup["total"], df_grup["persen"]):
        ax.annotate(
            f"{pct_val:.1f}%\n(n={total_val:,})",
            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    ax.set_ylabel("Persentase Puas (%)", fontsize=10)
    ax.set_xlabel(f"Kategori / Rentang Nilai: {nama_fitur}", fontsize=10)
    ax.set_title(f"Tingkat Kepuasan Penumpang Berdasarkan Nilai: {nama_fitur}", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(df_grup["persen"].max(), rata_puas_global) * 1.25)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_heatmap_korelasi(corr_matrix: pd.DataFrame, daftar_fitur: list) -> plt.Figure:
    """Membuat matriks korelasi heatmap antar 17 fitur terpilih."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(daftar_fitur)))
    ax.set_yticks(range(len(daftar_fitur)))
    ax.set_xticklabels(daftar_fitur, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(daftar_fitur, fontsize=9)

    for i in range(len(daftar_fitur)):
        for j in range(len(daftar_fitur)):
            val = corr_matrix.iloc[i, j]
            warna_teks = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=warna_teks, fontsize=7)

    ax.set_title("Heatmap Korelasi Antar 17 Fitur Terpilih", fontsize=12, fontweight="bold", pad=25)
    fig.tight_layout()
    return fig


def chart_alpha_fitur(df_model: pd.DataFrame) -> plt.Figure:
    """Membuat bar chart total bobot suara alpha per fitur pada model AdaBoost."""
    alpha_per_fitur = df_model.groupby("fitur")["alpha"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(alpha_per_fitur.index, alpha_per_fitur.values, color=WARNA_UTAMA, edgecolor="black", linewidth=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}", xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5)

    ax.set_xlabel("Total Bobot Voting (Jumlah Alpha)", fontsize=10)
    ax.set_title("Fitur Paling Berpengaruh dalam Model AdaBoost (Jumlah Bobot Alpha)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, alpha_per_fitur.max() * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_kurva_T(df_kurva: pd.DataFrame) -> plt.Figure:
    """Membuat kurva pergerakan akurasi latih dan validasi terhadap jumlah stump T."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df_kurva["T"], df_kurva["Latih"] * 100, label="Akurasi Data Latih", color="#2980b9", linewidth=2)
    ax.plot(df_kurva["T"], df_kurva["Validasi"] * 100, label="Akurasi Validasi Silang", color="#27ae60", linewidth=2)

    ax.axvline(x=67, color="#e74c3c", linestyle="--", linewidth=1.8, label="Model Terpilih (T = 67)")

    baris_67 = df_kurva[df_kurva["T"] == 67]
    if not baris_67.empty:
        acc_val_67 = baris_67["Validasi"].values[0] * 100
        ax.scatter([67], [acc_val_67], color="#e74c3c", s=60, zorder=5)
        ax.annotate(f"T=67 ({acc_val_67:.2f}%)", xy=(67, acc_val_67),
                    xytext=(10, -10), textcoords="offset points", fontweight="bold", color="#c0392b")

    ax.set_xlabel("Jumlah Iterasi Stump (T)", fontsize=10)
    ax.set_ylabel("Akurasi (%)", fontsize=10)
    ax.set_title("Kurva Kinerja Model Berdasarkan Jumlah Stump (T)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_cv_fold(df_cv: pd.DataFrame) -> plt.Figure:
    """Membuat bar chart akurasi validasi untuk tiap fold cross validation."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    folds = [f"Fold {f}" for f in df_cv.index]
    acc_val_folds = df_cv["Akurasi Validasi"] * 100
    bars = ax.bar(folds, acc_val_folds, color="#16a085", width=0.5, edgecolor="black", linewidth=0.5)

    rata_cv = acc_val_folds.mean()
    ax.axhline(rata_cv, color="#e67e22", linestyle="--", label=f"Rata-rata ({rata_cv:.2f}%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Akurasi Validasi (%)")
    ax.set_ylim(88, 95)
    ax.set_title("Akurasi Validasi per Fold", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """Membuat visualisasi confusion matrix heatmap 2x2."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    labels_cm = ["Puas (+1)", "Tidak Puas (-1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_cm, fontweight="bold")
    ax.set_yticklabels(labels_cm, fontweight="bold")
    ax.set_xlabel("Prediksi Model", fontweight="bold", fontsize=10)
    ax.set_ylabel("Aktual (Kenyataan)", fontweight="bold", fontsize=10)
    ax.set_title("Confusion Matrix (Data Uji)", fontsize=11, fontweight="bold")

    nama_sel = [["True Positive (TP)", "False Negative (FN)"],
                ["False Positive (FP)", "True Negative (TN)"]]

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total) * 100
            warna_teks = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)\n{nama_sel[i][j]}",
                    ha="center", va="center", color=warna_teks, fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    return fig


def chart_roc(fpr: np.ndarray, tpr: np.ndarray, auc_val: float) -> plt.Figure:
    """Membuat visualisasi kurva ROC dan nilai AUC."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#2980b9", linewidth=2.5, label=f"AdaBoost (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax.set_title("Kurva ROC (Receiver Operating Characteristic)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_perbandingan_konfig(df_konfig: pd.DataFrame) -> plt.Figure:
    """Membuat bar chart perbandingan akurasi test untuk berbagai konfigurasi model."""
    df_plot = df_konfig.copy()
    labels_cfg = df_plot["Fitur"].astype(str) + " fitur, T=" + df_plot["T"].astype(str)
    acc_test_cfg = df_plot["Akurasi Test"] * 100

    warna_cfg = [WARNA_PUAS if (f == 17 and t == 67) else WARNA_UTAMA
                 for f, t in zip(df_plot["Fitur"], df_plot["T"])]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels_cfg, acc_test_cfg, color=warna_cfg, width=0.5, edgecolor="black", linewidth=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Akurasi Test (%)")
    ax.set_ylim(80, 96)
    ax.set_title("Akurasi Test per Konfigurasi", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    return fig


def chart_kontribusi_stump(df_rincian: pd.DataFrame) -> plt.Figure:
    """Membuat bar chart kontribusi individu tiap stump (alpha * h) untuk penumpang tunggal."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    warna_stump = [WARNA_PUAS if val >= 0 else WARNA_TIDAK_PUAS for val in df_rincian["Kontribusi (α * h)"]]
    ax.bar(df_rincian["Iterasi"], df_rincian["Kontribusi (α * h)"], color=warna_stump, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Iterasi Stump (1 s.d. 67)", fontsize=9)
    ax.set_ylabel("Kontribusi (α * h)", fontsize=9)
    ax.set_title("Kontribusi Tiap Stump untuk Penumpang Ini\n(Hijau: Mendukung Puas, Merah: Mendukung Tidak Puas)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def chart_kontribusi_fitur(df_rincian: pd.DataFrame) -> plt.Figure:
    """Membuat horizontal bar chart akumulasi voting tiap fitur untuk satu penumpang."""
    kontribusi_per_fitur = df_rincian.groupby("Fitur")["Kontribusi (α * h)"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    warna_fitur = [WARNA_PUAS if val >= 0 else WARNA_TIDAK_PUAS for val in kontribusi_per_fitur.values]
    bars = ax.barh(kontribusi_per_fitur.index, kontribusi_per_fitur.values, color=warna_fitur, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    for bar in bars:
        w = bar.get_width()
        offset = 4 if w >= 0 else -4
        ha = "left" if w >= 0 else "right"
        ax.annotate(f"{w:+.2f}", xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(offset, 0), textcoords="offset points", ha=ha, va="center", fontsize=8)

    ax.set_xlabel("Total Kontribusi (Jumlah α * h)", fontsize=9)
    ax.set_title("Pengaruh Bersih Tiap Fitur pada Penumpang Ini", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig
