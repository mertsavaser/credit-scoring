import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

# --- helpers ---
def _normalize_col(name: str) -> str:
    s = name.strip().lower()
    s = s.replace(' ', '_').replace('-', '_').replace('%', 'pct')
    s = re.sub(r'[^a-z0-9_]', '', s)   # harf/rakam/altçizgi dışını temizle
    s = re.sub(r'__+', '_', s)
    return s

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    return df

# --- api ---
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Raw data not found at: {path}. Put your CSV as data/raw/credit.csv')
    return pd.read_csv(path)

def basic_clean(
    df: pd.DataFrame,
    drop_id_cols: Optional[List[str]] = None,
    save_to: Optional[Path] = None
) -> pd.DataFrame:
    """
    - Kolon isimlerini normalize eder (lowercase, boşluk/(-)->_).
    - id/unnamed_0/index gibi faydasız sütunları düşer.
    - Basit mantık: age<18 -> çıkar; negatif monthlyincome -> NaN.
    - (opsiyonel) temiz veriyi diske yazar.
    """
    df = _normalize_columns(df)

    # ID/Index benzeri kolonları düş
    drop_cols = {'unnamed_0', 'unnamed:_0', 'id', 'index'}
    if drop_id_cols:
        drop_cols |= {_normalize_col(c) for c in drop_id_cols}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Mantık kontrolleri (varsa)
    if 'age' in df.columns:
        df = df[df['age'].notna()]
        df = df[df['age'] >= 18]
    if 'monthlyincome' in df.columns:
        # Negatif gelir hatalıdır -> NaN bırak, imputer doldurur
        df.loc[df['monthlyincome'] < 0, 'monthlyincome'] = np.nan

    # object kolonlardan sayısala dönebilenleri çevir (etiket yoksa)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='ignore')

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)

    return df

def train_val_test_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Stratified train/val/test split.
    val_size, kalan (1 - test_size) içinde oranlanır.
    """
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` not found. Available: {list(df.columns)[:20]}")

    X = df.drop(columns=[target])
    y = df[target]

    # y tek sınıfa düşmüşse stratify kullanma (güvenlik)
    strat = y if y.nunique(dropna=True) > 1 else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state
    )

    val_rel = val_size / (1 - test_size)
    strat_temp = y_temp if y_temp.nunique(dropna=True) > 1 else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_rel, stratify=strat_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
