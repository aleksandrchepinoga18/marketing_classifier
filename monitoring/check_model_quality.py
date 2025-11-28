# monitoring/check_model_quality.py
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

def check_quality():
    df = pd.read_csv("monitoring/predictions_with_labels.csv")
    y_true = df["actual_response"]   # ← СИМУЛИРОВАННЫЕ (или настоящие) метки
    y_prob = df["prediction_probability"]   # ← предсказания модели
    y_pred = df["prediction_class"]
    
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    
    print(f"ROC-AUC на новых данных: {roc_auc:.4f}")
    print(f"F1 на новых данных: {f1:.4f}")
    
    # Сравниваем с исходным качеством (из metrics.json)
    return roc_auc, f1