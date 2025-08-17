import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from src.utils import BEST_MODEL_PATH
from src.data_prep import load_raw, basic_clean, train_val_test_split

# === Veri seti hedefi (basic_clean sonrası küçük harf) ===
TARGET = "seriousdlqin2yrs"   # GiveMeSomeCredit için
# TARGET = "target"           # HomeCredit kullanıyorsan

# === Politika / Rapor ayarları ===
THRESH = 0.35                 # örnek cut-off (iş mantığına göre ayarlanır)
TOPK = 0.10                   # ilk %10 en riskli
COST_FP = 1.0                 # hatalı red maliyeti (false positive)
COST_FN = 5.0                 # default’u kaçırma maliyeti (false negative)

def topk_recall(y_true: np.ndarray, proba: np.ndarray, k: float) -> float:
    n = len(proba)
    take = max(1, int(np.ceil(k * n)))
    idx = np.argsort(-proba)[:take]
    positives = (y_true == 1).sum()
    if positives == 0:
        return 0.0
    captured = (y_true[idx] == 1).sum()
    return captured / positives

def cost_at_threshold(y_true: np.ndarray, proba: np.ndarray, thresh: float,
                      cost_fp: float, cost_fn: float) -> float:
    y_pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn

def sweep_thresholds(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    best = {"thresh": None, "f1": -1.0, "cost": float("inf")}
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        c = cost_at_threshold(y_true, proba, t, COST_FP, COST_FN)
        # Önce maliyete, eşitse F1'e bak
        if (c < best["cost"]) or (np.isclose(c, best["cost"]) and f1 > best["f1"]):
            best = {"thresh": float(t), "f1": float(f1), "cost": float(c)}
    return best

def main():
    # Model ve veri
    model = joblib.load(BEST_MODEL_PATH)
    df = basic_clean(load_raw(Path("data/raw/credit.csv")))

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, TARGET)
    proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    # --- Global metrikler ---
    roc = roc_auc_score(y_true, proba)
    prauc = average_precision_score(y_true, proba)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {prauc:.4f}")

    # --- Top-K recall ---
    tk = topk_recall(y_true, proba, TOPK)
    print(f"Top-{int(TOPK*100)}% Recall: {tk:.4f}")

    # --- Seçili eşikte rapor ---
    y_pred = (proba >= THRESH).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nCut-off: {THRESH}")
    print(cm)
    print(classification_report(y_true, y_pred, digits=4))

    # --- Maliyet hesabı ---
    cost = cost_at_threshold(y_true, proba, THRESH, COST_FP, COST_FN)
    print(f"Cost@{THRESH}: {cost:.2f}  (FP={COST_FP}, FN={COST_FN})")

    # --- Eşik taraması (öneri) ---
    best = sweep_thresholds(y_true, proba)
    print(f"\n[Suggest] Best threshold (by cost then F1): {best['thresh']:.2f} | "
          f"F1={best['f1']:.4f} | Cost={best['cost']:.2f}")

if __name__ == "__main__":
    main()
