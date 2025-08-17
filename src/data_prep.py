import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Raw data not found at: {path}. Put your CSV as data/raw/credit.csv')
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Örnek: kolon adlarını normalize et
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Basit mantık kontrolleri (örn. negatif yaş/gelir silebilirsin)
    # df = df[df['age'] >= 18]  # varsa
    return df

def train_val_test_split(df: pd.DataFrame, target: str, test_size: float=0.2, val_size: float=0.1, random_state: int=42) -> Tuple[pd.DataFrame, ...]:
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_rel = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_rel, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
