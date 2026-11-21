"""
Logika matematika model AdaBoost, evaluasi metrik, dan perhitungan voting (NumPy & Pandas).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def hitung_H(df_fitur: pd.DataFrame, model: pd.DataFrame) -> np.ndarray:
    """Menghitung skor ensemble H(x) = sum(alpha_t * h_t(x)) untuk semua stump t."""
    H = np.zeros(len(df_fitur))
    for _, s in model.iterrows():
        dasar = np.where(df_fitur[s["fitur"]].to_numpy() > s["thr"], 1, -1)
        H += s["alpha"] * s["pol"] * dasar
    return H


def prediksi(H: np.ndarray) -> np.ndarray:
    """Mengubah skor ensemble H menjadi kelas biner (+1 Puas jika H > 0, selain itu -1)."""
    return np.where(H > 0, 1, -1)


def hitung_keyakinan(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Menghitung estimasi probabilitas puas dan persentase keyakinan keputusan."""
    prob_puas = 1.0 / (1.0 + np.exp(-2.0 * H))
    keyakinan_pct = np.where(H > 0, prob_puas, 1.0 - prob_puas) * 100.0
    return prob_puas, keyakinan_pct


def kontribusi_stump(df_satu: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Menyusun rincian kontribusi voting tiap stump untuk satu penumpang individual."""
    rincian = []
    for _, s in model.iterrows():
        fitur_nama = s["fitur"]
        nilai_input = df_satu.iloc[0][fitur_nama]
        dasar = 1 if nilai_input > s["thr"] else -1
        h = int(s["pol"] * dasar)
        aturan = f"{fitur_nama} > {s['thr']}" if s["pol"] == 1 else f"{fitur_nama} <= {s['thr']}"
        kontribusi = s["alpha"] * h
        rincian.append({
            "Iterasi": int(s["iterasi"]),
            "Fitur": fitur_nama,
            "Nilai Input": nilai_input,
            "Threshold": s["thr"],
            "Aturan": aturan,
            "Vote (h)": h,
            "Alpha (α)": s["alpha"],
            "Kontribusi (α * h)": kontribusi,
        })
    return pd.DataFrame(rincian)


def hitung_metrik(y_true: np.ndarray, y_pred: np.ndarray, H: np.ndarray) -> dict:
    """Menghitung metrik performa klasifikasi (Akurasi, Presisi, Recall, F1, AUC, CM, ROC)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, H)
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    fpr, tpr, _ = roc_curve(y_true, H, pos_label=1)

    return {
        "akurasi": acc,
        "presisi": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }
