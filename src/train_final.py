# src/train_final.py
import pandas as pd
import numpy as np
import json
import os
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, average_precision_score,
    classification_report, roc_auc_score
)
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Добавляем src в путь (на случай запуска напрямую)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Используем абсолютный импорт
from data import load_and_clean_data

def main():
    OUTPUT_DIR = "reports/model"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------
    # 1. ЗАГРУЗКА И ОЧИСТКА
    # -------------------------------------------------
    df = load_and_clean_data("ifood_df.csv")
    X = df.drop(columns=['Response'])
    y = df['Response']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

    print(f"Размеры: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Баланс (Response=1): train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    # -------------------------------------------------
    # 2. ПАРАМЕТРЫ
    # -------------------------------------------------
    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "learning_rate": 0.05,
        "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        "max_depth": 5,
        "min_data_in_leaf": 100
    }

    print("✅ Используем финальные параметры для деплоя")
    with open(os.path.join(OUTPUT_DIR, "params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # -------------------------------------------------
    # 3. ОБУЧЕНИЕ
    # -------------------------------------------------
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['valid'],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(50)
        ]
    )

    final_model.save_model(os.path.join(OUTPUT_DIR, "model.txt"))

    # -------------------------------------------------
    # 4. ПОДБОР ПОРОГА НА VAL
    # -------------------------------------------------
    y_val_proba = final_model.predict(X_val)
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_val, (y_val_proba >= th).astype(int)) for th in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n🔢 Лучший порог (по F1 на val): {best_threshold:.4f}")

    # -------------------------------------------------
    # 5. МЕТРИКИ НА ВСЕХ ВЫБОРКАХ (включая ROC-AUC на val!)
    # -------------------------------------------------
    y_train_proba = final_model.predict(X_train)
    y_val_proba = final_model.predict(X_val)
    y_test_proba = final_model.predict(X_test)

    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    # F1
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # ROC-AUC (главное — теперь есть на val!)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # PR-AUC
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    # Переобучение
    overfitting_f1 = (train_f1 - val_f1) > 0.05
    overfitting_roc = (train_roc_auc - val_roc_auc) > 0.02
    overfitting_detected = overfitting_f1 or overfitting_roc

    metrics = {
        "best_threshold": float(best_threshold),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "train_roc_auc": float(train_roc_auc),
        "val_roc_auc": float(val_roc_auc),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "overfitting_detected": bool(overfitting_detected),
        "overfitting_f1": bool(overfitting_f1),
        "overfitting_roc_auc": bool(overfitting_roc),
        "class_balance_test": float(y_test.mean()),
        "best_iteration": int(final_model.best_iteration)
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n📊 Метрики:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # -------------------------------------------------
    # 6. CLASSIFICATION REPORT
    # -------------------------------------------------
    report_str = classification_report(y_test, y_test_pred)
    RUN_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_with_header = f"Модель обучена: {RUN_TIME}\nПорог: {best_threshold:.4f}\n\n" + report_str

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_with_header)

    print("\n" + "="*60)
    print("📋 FULL CLASSIFICATION REPORT (TEST):")
    print("="*60)
    print(report_str)
    print("="*60)
    print(f"\n✅ Отчёт сохранён в: {os.path.abspath(report_path)}")

    # -------------------------------------------------
    # 7. АРТЕФАКТЫ
    # -------------------------------------------------
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 10))
    importance_df.head(20).plot.barh(x='feature', y='importance', legend=False)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (Final Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_val)
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(shap_vals, X_val, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    X_test.sample(5, random_state=42).to_csv(os.path.join(OUTPUT_DIR, "sample_input.csv"), index=False)

    print(f"\n🎉 Финальная модель и все артефакты сохранены в '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()