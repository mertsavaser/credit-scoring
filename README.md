# Credit Scoring – Risk Tahmin Projesi

**Amaç:** Kredi başvurularında default riskini tahmin eden bir model ve mini demo app.

## Hızlı Başlangıç
1) Sanal ortam ve kurulum:
   \\\ash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   \\\

2) Veri:
   - Kaggle'dan uygun kredi risk setini indir (örn. *Give Me Some Credit*).
   - Ana CSV dosyasını **data/raw/credit.csv** olarak kaydet.

3) Eğitim:
   \\\ash
   python src/train.py
   \\\

4) Değerlendirme:
   \\\ash
   python src/evaluate.py
   \\\

5) Demo (Streamlit):
   \\\ash
   streamlit run app/streamlit_app.py
   \\\

## Klasör Yapısı
\\\
credit-scoring/
├─ data/
│  ├─ raw/                # orijinal csv (credit.csv)
│  ├─ interim/            # temizlenmiş
│  └─ processed/          # feature engineered
├─ notebooks/             # EDA/modelleme defterleri
├─ src/
│  ├─ data_prep.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ explain.py
│  └─ utils.py
├─ app/
│  ├─ streamlit_app.py
│  └─ Dockerfile
├─ models/
│  ├─ best_model.pkl
│  └─ preprocessor.pkl
├─ requirements.txt
├─ .gitignore
└─ README.md
\\\

## Notlar
- Dengesiz sınıf yönetimi (class_weight, SMOTE) ve eşik ayarı ilerleyen commit'lerde.
- SHAP ile açıklanabilirlik örneği explain.py içinde.
