import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data_prep import load_raw, basic_clean, train_val_test_split
from src.features import (
    infer_feature_types,
    build_preprocessor,
    make_custom_features,   # <<< FE'yi pipeline'a ekleyeceğiz
)
from src.utils import DATA_RAW, BEST_MODEL_PATH, PREPROCESSOR_PATH

# === Veri seti hedefi ===
# GiveMeSomeCredit -> basic_clean sonrası:
TARGET = "seriousdlqin2yrs"


def compute_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0:
        return 1.0
    return neg / pos


def build_pre_feat(num_cols, cat_cols, scale_numeric: bool):
    """
    FE (make_custom_features) + Preprocessor'ı tek bir pipeline'da birleştirir.
    Böylece eğitim ve tahminde aynı dönüşümler otomatik uygulanır.
    """
    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=scale_numeric)
    pre_feat = Pipeline(
        steps=[
            ("feat", FunctionTransformer(make_custom_features, validate=False)),
            ("preprocessor", pre),
        ]
    )
    return pre_feat


def train_lr(X_train, y_train, pre_feat):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
    pipe = Pipeline(steps=[("pre_feat", pre_feat), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train, y_train, X_val, y_val, pre_feat):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise ImportError("xgboost yüklü değil. requirements.txt ile kur: xgboost") from e

    # FE+Pre'yi fit edip, early stopping için dönüştürülmüş matrisleri kullan
    pre_feat.fit(X_train, y_train)
    Xtr = pre_feat.transform(X_train)
    Xva = pre_feat.transform(X_val)

    spw = compute_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=0,
    )
    model.fit(
        Xtr,
        y_train,
        eval_set=[(Xva, y_val)],
        early_stopping_rounds=100,
        verbose=False,
    )

    # Son pipeline: FE+Pre + Model
    pipe = Pipeline(steps=[("pre_feat", pre_feat), ("clf", model)])
    return pipe


def train_lgb(X_train, y_train, X_val, y_val, pre_feat):
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        raise ImportError("lightgbm yüklü değil. requirements.txt ile kur: lightgbm") from e

    pre_feat.fit(X_train, y_train)
    Xtr = pre_feat.transform(X_train)
    Xva = pre_feat.transform(X_val)

    spw = compute_pos_weight(y_train)
    model = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        scale_pos_weight=spw,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        Xtr,
        y_train,
        eval_set=[(Xva, y_val)],
        eval_metric="auc",
        early_stopping_rounds=200,
        verbose=False,
    )

    pipe = Pipeline(steps=[("pre_feat", pre_feat), ("clf", model)])
    return pipe


def eval_report(pipe, X, y, split_name: str):
    proba = pipe.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, proba)
    prauc = average_precision_score(y, proba)
    print(f"{split_name.upper()} -> ROC-AUC: {roc:.4f} | PR-AUC (AP): {prauc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["lr", "xgb", "lgb"], default="lr", help="Hangi modeli eğitelim? lr / xgb / lgb"
    )
    args = parser.parse_args()
    model_name = args.model

    # 1) Veri
    df = basic_clean(load_raw(DATA_RAW))
    if TARGET not in df.columns:
        raise ValueError(f"Target column `{TARGET}` not in dataframe. Available: {list(df.columns)[:20]} ...")

    # 2) FE sonrası şema üzerinden feature tiplerini çıkar
    df_for_types = make_custom_features(df.copy())
    num_cols, cat_cols = infer_feature_types(df_for_types, TARGET)
    scale_numeric = model_name == "lr"  # ağaç tabanlılarda scale gerekmez
    pre_feat = build_pre_feat(num_cols, cat_cols, scale_numeric=scale_numeric)

    # 3) Split (ham DF üzerinde; FE pipeline içinde yapılacak)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, TARGET)

    # 4) Model eğitimi
    if model_name == "lr":
        pipe = train_lr(X_train, y_train, pre_feat)
    elif model_name == "xgb":
        pipe = train_xgb(X_train, y_train, X_val, y_val, pre_feat)
    else:  # lgb
        pipe = train_lgb(X_train, y_train, X_val, y_val, pre_feat)

    # 5) Değerlendirme (val & test)
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        eval_report(pipe, X, y, split_name)

    # 6) Kaydet: model + (FE+Pre) bileşeni ayrıca
    joblib.dump(pipe, BEST_MODEL_PATH)
    # preprocessor'ı ayrıca kaydetmek istersen:
    try:
        joblib.dump(pipe.named_steps["pre_feat"].named_steps["preprocessor"], PREPROCESSOR_PATH)
    except Exception:
        pass
    print(f"Saved model to {BEST_MODEL_PATH}")

    # 7) App için eğitimde kullanılan ham kolon isimlerini kaydet (app ham alan gönderir)
    feature_names = X_train.columns.tolist()
    try:
        from src.utils import FEATURE_NAMES_PATH
        with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        print("Saved feature names for app (models/feature_names.json).")
    except Exception:
        pass


if __name__ == "__main__":
    main()
