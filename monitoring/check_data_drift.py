# monitoring/check_data_drift.py
import pandas as pd
from scipy import stats
from src.data import load_and_clean_data

def detect_drift(threshold=0.05):
    # Исходные данные
    train_df = load_and_clean_data("../ifood_df.csv")
    
    # Новые данные
    new_df = pd.read_csv("monitoring/predictions_with_labels.csv")
    feature_cols = [col for col in new_df.columns if col not in [
        "prediction_probability", "prediction_class", "timestamp", "actual_response"
    ]]
    
    drifts = {}
    for col in feature_cols:
        if col in train_df.columns:
            try:
                _, p = stats.ks_2samp(
                    train_df[col].dropna(),
                    new_df[col].dropna()
                )
                drifts[col] = p < threshold
            except:
                drifts[col] = False
    
    n_drifts = sum(drifts.values())
    print(f"Обнаружено признаков с дрифтом: {n_drifts} из {len(drifts)}")
    return n_drifts > 0