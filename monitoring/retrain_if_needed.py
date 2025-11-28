# monitoring/retrain_if_needed.py
import sys
import os
sys.path.append(".")

from monitoring.check_model_quality import check_quality
from monitoring.check_data_drift import detect_drift
from src.train_final import main as retrain_model

# Пороги деградации
ROC_AUC_THRESHOLD = 0.85
F1_THRESHOLD = 0.45

def main():
    print("🔍 Проверка качества модели...")
    
    # Шаг 1: Проверяем качество
    roc_auc, f1 = check_quality()
    quality_degraded = (roc_auc < ROC_AUC_THRESHOLD) or (f1 < F1_THRESHOLD)
    
    # Шаг 2: Проверяем дрифт
    drift_detected = detect_drift()
    
    if quality_degraded or drift_detected:
        print("⚠️ Обнаружена деградация модели или data drift → запуск ретрейна")
        retrain_model()
        print("✅ Модель успешно переобучена!")
    else:
        print("✅ Модель в порядке. Ретрейн не требуется.")

if __name__ == "__main__":
    main()