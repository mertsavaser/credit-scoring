from pathlib import Path

# --- Proje kökü: src/utils.py konumundan bir üst klasör ---
ROOT = Path(__file__).resolve().parents[1]

# --- Klasörler ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Klasörleri oluştur (idempotent)
for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Dosya yolları ---
DATA_RAW = RAW_DIR / "credit.csv"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"  # app için eğitimdeki ham kolon listesi

# (opsiyonel) açıklama/rapor çıktıları
SHAP_LOCAL_PATH = MODEL_DIR / "shap_local.png"
SHAP_GLOBAL_PATH = MODEL_DIR / "shap_global.png"
METRICS_JSON = REPORTS_DIR / "metrics.json"

# (opsiyonel) tek noktadan seed
RANDOM_SEED = 42
