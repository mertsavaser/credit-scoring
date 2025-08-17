from pathlib import Path

DATA_RAW = Path('data/raw/credit.csv')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / 'best_model.pkl'
PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.pkl'
