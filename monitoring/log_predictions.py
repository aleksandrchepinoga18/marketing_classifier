# monitoring/log_predictions.py
import pandas as pd
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "predictions.csv")
DEBUG_LOG = os.path.join(os.path.dirname(__file__), "debug.log")

def log_prediction(features, prob, will_respond):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"Запуск логирования\n")
            f.write(f"Путь к файлу: {LOG_FILE}\n")
        
        record = features.copy()
        record.update({
            "prediction_probability": prob,
            "prediction_class": will_respond,
            "timestamp": datetime.now().isoformat(),
            "actual_response": None
        })
        
        df = pd.DataFrame([record])
        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)
            
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write("✅ Успешно записано\n")
            
    except Exception as e:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"❌ ОШИБКА: {str(e)}\n")