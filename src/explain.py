import shap
import joblib
import pandas as pd
from src.data_prep import load_raw, basic_clean, train_val_test_split
from pathlib import Path

TARGET = 'default'

def main():
    pipe = joblib.load('models/best_model.pkl')
    df = basic_clean(load_raw(Path('data/raw/credit.csv')))
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, TARGET)

    # Pipeline içindeki preprocesstan geçmiş özellikleri SHAP için dönüştür
    pre = pipe.named_steps['preprocessor']
    X_trans = pre.transform(X_test)
    clf = pipe.named_steps['clf']

    # Linear model için KernelExplainer yerine LinearExplainer daha hızlı
    try:
        explainer = shap.LinearExplainer(clf, X_trans, feature_dependence="independent")
    except Exception:
        explainer = shap.KernelExplainer(clf.predict_proba, X_trans[:200])
    shap_values = explainer.shap_values(X_trans[:1])

    # Not: OHE sonrası feature isimlerini almak istersen:
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f'f{i}' for i in range(X_trans.shape[1])]

    shap.force_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                    matplotlib=True)
    print('SHAP local plot generated (matplotlib backend).')

if __name__ == '__main__':
    main()
