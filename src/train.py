from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data_prep import load_raw, basic_clean, train_val_test_split
from src.features import infer_feature_types, build_preprocessor
from src.utils import DATA_RAW, BEST_MODEL_PATH, PREPROCESSOR_PATH

TARGET = 'default'  # verine göre değiştir (örn. default, SeriousDlqin2yrs vs.)

def main():
    # Veri
    df = load_raw(DATA_RAW)
    df = basic_clean(df)
    if TARGET not in df.columns:
        raise ValueError(f'Target column {TARGET} not in dataframe. Available: {list(df.columns)[:20]} ...')

    num_cols, cat_cols = infer_feature_types(df, TARGET)
    pre = build_preprocessor(num_cols, cat_cols)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, TARGET)

    # Model (class_weight dengesiz veri için)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None)

    pipe = Pipeline(steps=[
        ('preprocessor', pre),
        ('clf', clf)
    ])

    pipe.fit(X_train, y_train)

    # Değerlendirme
    for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        proba = pipe.predict_proba(X)[:, 1]
        roc = roc_auc_score(y, proba)
        prauc = average_precision_score(y, proba)
        print(f'{split_name.upper()} -> ROC-AUC: {roc:.4f} | PR-AUC: {prauc:.4f}')

    # Kaydet
    joblib.dump(pipe, BEST_MODEL_PATH)
    # Preprocessor pipeline zaten pipe içinde; ayrıca gerekirse:
    joblib.dump(pipe.named_steps['preprocessor'], PREPROCESSOR_PATH)
    print(f'Saved model to {BEST_MODEL_PATH}')

if __name__ == '__main__':
    main()
