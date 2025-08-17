import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import BEST_MODEL_PATH
from src.data_prep import load_raw, basic_clean, train_val_test_split

TARGET = 'default'
THRESH = 0.35  # örnek cut-off

def main():
    model = joblib.load(BEST_MODEL_PATH)
    df = basic_clean(load_raw(Path('data/raw/credit.csv')))
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, TARGET)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESH).astype(int)
    print('Cut-off:', THRESH)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == '__main__':
    main()
