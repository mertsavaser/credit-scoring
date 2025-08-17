# --- Make 'src' importable during unpickle & fix all paths relative to project root ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------- Paths -----------------
MODEL_PATH  = ROOT / "models" / "best_model.pkl"
FEAT_PATH   = ROOT / "models" / "feature_names.json"
POLICY_PATH = ROOT / "models" / "policy.json"

# ----------------- UI -----------------
st.set_page_config(page_title="Credit Scoring Demo", layout="centered")
st.title("📊 Credit Scoring – Risk Tahmin Demo")
st.write("Örnek alanları doldur, risk skorunu ve politikaya göre kararı gör.")

# ----------------- Load artifacts -----------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}. "
            "Önce `python -m src.train --model xgb` ile modeli üret."
        )
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_feature_names():
    if FEAT_PATH.exists():
        return json.loads(FEAT_PATH.read_text(encoding="utf-8"))
    return None

@st.cache_data(show_spinner=False)
def load_policy():
    if POLICY_PATH.exists():
        return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    # varsayılan politika (gerekirse)
    return {"type": "threshold", "threshold": 0.50}

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model yüklenemedi: {e}")
    st.stop()

feature_names = load_feature_names()
policy = load_policy()

# ----------------- Form -----------------
col1, col2 = st.columns(2)
with col1:
    annual_income = st.number_input("Annual Income", min_value=0, value=60000, step=1000)
    dti_pct = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=500.0, value=25.0, step=0.5)
with col2:
    delinquency = st.number_input("Delinquency Count (last 2 years)", min_value=0, value=0, step=1)
    util_pct = st.number_input("Credit Utilization (%)", min_value=0.0, max_value=1000.0, value=30.0, step=0.5)

if st.button("Skoru Hesapla"):
    # 1) Eğitimde beklenen ham kolonlardan tek satırlık DataFrame oluştur
    if feature_names:
        row = pd.DataFrame([{c: np.nan for c in feature_names}])
    else:
        row = pd.DataFrame([{}])  # imputer kalanları halleder

    # 2) Bizim 4 girdi → dataset kolonlarına eşleme (basic_clean sonrası isimler)
    #    Not: model aylık gelir bekliyor; UI yıllık alıyor → /12
    row.loc[0, "monthlyincome"] = float(annual_income) / 12.0

    # yüzdelikler → 0-1 aralığı
    row.loc[0, "debtratio"] = float(dti_pct) / 100.0
    row.loc[0, "revolvingutilizationofunsecuredlines"] = float(util_pct) / 100.0

    # basit gecikme haritalaması
    row.loc[0, "numberoftime30_59dayspastduenotworse"] = int(delinquency)
    row.loc[0, "numberoftime60_89dayspastduenotworse"] = 0
    row.loc[0, "numberoftimes90dayslate"] = 0

    # 3) Skor
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as e:
        st.error(f"Tahmin sırasında hata: {e}")
        st.stop()

    # 4) Politika eşiği
    if policy.get("type") == "topk":
        t = float(policy.get("threshold", 0.5))
        policy_txt = f"Top-K (K={int(policy.get('K', 0.1)*100)}%)"
    else:
        t = float(policy.get("threshold", 0.5))
        policy_txt = "Cost-based threshold"

    # 5) Gösterim
    st.metric("Default Risk Skoru", f"{proba:.3f}")
    st.caption(f"Politika: {policy_txt} | Eşik: {t:.3f}")

    if proba >= t:
        st.error("Karar: REVIEW / REJECT adayı")
    else:
        st.success("Karar: APPROVE")

    with st.expander("🛠 Gönderilen ham özellikler (debug)"):
        st.dataframe(row.T)

    with st.expander("ℹ️ Artefakt yolları"):
        st.write({"MODEL_PATH": str(MODEL_PATH), "FEAT_PATH": str(FEAT_PATH), "POLICY_PATH": str(POLICY_PATH)})
