import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path('models/best_model.pkl')
st.set_page_config(page_title='Credit Scoring Demo', layout='centered')

st.title('📊 Credit Scoring – Risk Tahmin Demo')
st.write('Örnek alanları doldur, risk skorunu gör.')

# Basit örnek alanlar (kendi verine göre güncelle)
col1, col2 = st.columns(2)
with col1:
    income = st.number_input('Annual Income', min_value=0, value=50000, step=1000)
    dti = st.number_input('Debt-to-Income Ratio (%)', min_value=0.0, value=25.0, step=0.5)
with col2:
    delinq = st.number_input('Delinquency Count (last 2 years)', min_value=0, value=0, step=1)
    util = st.number_input('Credit Utilization (%)', min_value=0.0, value=30.0, step=0.5)

if not MODEL_PATH.exists():
    st.warning('Model bulunamadı. Önce python src/train.py ile eğit.')
else:
    model = joblib.load(MODEL_PATH)
    # Form verilerini tek satırlık DataFrame'e çevir
    sample = pd.DataFrame([{
        'annual_income': income,
        'dti': dti,
        'delinquency_count': delinq,
        'credit_utilization': util
    }])

    proba = model.predict_proba(sample)[0, 1]
    st.metric('Default Risk Skoru', f'{proba:.3f}')

    if proba >= 0.35:
        st.error('Karar: RED')
    elif proba >= 0.20:
        st.warning('Karar: MANUAL REVIEW')
    else:
        st.success('Karar: APPROVE')
