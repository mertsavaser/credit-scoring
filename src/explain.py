import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_prep import load_raw, basic_clean, train_val_test_split
from src.utils import BEST_MODEL_PATH

# === Veri seti hedefi (basic_clean sonrası) ===
TARGET = "seriousdlqin2yrs"   # Give Me Some Credit için

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_dense(X):
    """Sparse ise dense'e çevir."""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def main():
    # Model & veri
    pipe = joblib.load(BEST_MODEL_PATH)
    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["clf"]

    df = basic_clean(load_raw(Path("data/raw/credit.csv")))
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, TARGET)

    # Modelin gördüğü uzaya geçir
    X_trans = pre.transform(X_test)
    X_trans = to_dense(X_trans)

    # Feature isimleri (OHE sonrası)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    # Arka plan örneklemi (hız ve stabilite için)
    bg = shap.utils.sample(X_trans, 200, random_state=42)

    # Explainer: model tipine göre uygun yöntemi seçer (linear/tree/kernel)
    explainer = shap.Explainer(clf, bg, feature_names=feature_names)

    # --- LOCAL: tek müşteriyi açıkla (ilk satır) ---
    sv_local = explainer(X_trans[:1])   # Explanation objesi
    plt.figure()
    shap.plots.waterfall(sv_local[0], show=False, max_display=15)
    plt.tight_layout()
    local_path = OUT_DIR / "shap_local.png"
    plt.savefig(local_path, dpi=160)
    plt.close()

    # --- GLOBAL: genel önem (ilk 1000 örnek üzerinden) ---
    sample_n = min(1000, X_trans.shape[0])
    sv_global = explainer(X_trans[:sample_n])
    plt.figure()
    shap.plots.bar(sv_global, show=False, max_display=20)
    plt.tight_layout()
    global_path = OUT_DIR / "shap_global.png"
    plt.savefig(global_path, dpi=160)
    plt.close()

    print(f"Saved SHAP plots:\n - {local_path}\n - {global_path}")

if __name__ == "__main__":
    main()
