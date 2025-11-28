# api/app.py
import os
import sys
import json
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.log_predictions import log_prediction

app = Flask(__name__)

# Пути к модели и метрикам
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "reports", "model", "model.txt")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "model", "metrics.json")

# Проверка существования файлов
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"Метрики не найдены: {METRICS_PATH}")

# Загрузка модели и порога
model = lgb.Booster(model_file=MODEL_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)
threshold = metrics["best_threshold"]
expected_features = model.feature_name()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body is required"}), 400

        df = pd.DataFrame([data])
        df = df.reindex(columns=expected_features, fill_value=0)
        prob = model.predict(df)[0]
        will_respond = bool(prob >= threshold)

        # Логирование предсказания
        log_prediction(data, prob, will_respond)

        return jsonify({
            "probability": float(prob),
            "will_respond": will_respond,
            "threshold_used": float(threshold)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)