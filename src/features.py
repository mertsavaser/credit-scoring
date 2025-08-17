import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def infer_feature_types(df: pd.DataFrame, target: str):
    num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    pre = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ],
        remainder='drop'
    )
    return pre
