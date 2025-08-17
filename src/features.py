import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Tip ayırma ---
def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols

# --- Küçük yardımcı ---
def _winsorize(s: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    ql, qh = s.quantile(low), s.quantile(high)
    return s.clip(ql, qh)

# --- FE: Notebook’ta denediklerimizin prod karşılığı ---
def make_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Delinquency toplamı + bayrak
    cand = [c for c in df.columns if "dayspastdue" in c]
    if cand:
        df["delinq_total"] = df[cand].sum(axis=1).clip(upper=10)
        df["any_delinquency"] = (df["delinq_total"] > 0).astype(int)

    # DebtRatio: winsorize + log1p + extreme bayrak
    if "debtratio" in df.columns:
        df["debtratio_w"] = _winsorize(df["debtratio"])
        df["debtratio_log1p"] = np.log1p(df["debtratio_w"])
        try:
            thr = df["debtratio"].quantile(0.99)
            df["debtratio_extreme"] = (df["debtratio"] > thr).astype(int)
        except Exception:
            pass

    # Revolving utilization: winsorize + log1p + uç bayraklar
    util_col = "revolvingutilizationofunsecuredlines"
    if util_col in df.columns:
        u = _winsorize(df[util_col])
        df["util_w"] = u
        df["util_log1p"] = np.log1p(u)
        df["util_zero"] = (df[util_col] == 0).astype(int)
        try:
            thr_u = df[util_col].quantile(0.95)
            df["util_high"] = (df[util_col] > thr_u).astype(int)
        except Exception:
            pass

    # Gelir eksik bayrağı
    if "monthlyincome" in df.columns:
        df["monthlyincome_missing"] = df["monthlyincome"].isna().astype(int)

    # Yaş bandı
    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=[18, 25, 35, 45, 55, 65, 120],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            include_lowest=True
        )
    return df

# --- Preprocessor ---
def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    scale_numeric: bool = True,
    ohe_min_freq: Optional[float] = None
) -> ColumnTransformer:
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    ohe_kwargs = {'handle_unknown': 'ignore'}
    if ohe_min_freq is not None:
        ohe_kwargs.update({'min_frequency': ohe_min_freq, 'handle_unknown': 'infrequent_if_exist'})

    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(**ohe_kwargs, sparse_output=True))
    ])

    pre = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ],
        remainder='drop',
        n_jobs=None
    )
    return pre

# --- OHE sonrası isimler (opsiyonel) ---
def get_feature_names_out(pre: ColumnTransformer) -> List[str]:
    try:
        names = pre.get_feature_names_out()
        return names.tolist() if hasattr(names, 'tolist') else list(names)
    except Exception:
        return []
