# src/eda.py
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # важно для сохранения без GUI
import matplotlib.pyplot as plt
from data import load_and_clean_data

def run_eda(data_path: str = "ifood_df.csv", output_dir: str = "reports/eda"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean_data(data_path)

    # 1. Общая информация и статистика
    with open(os.path.join(output_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:
        df.info(buf=f)
        f.write("\n\n=== Описательная статистика ===\n")
        f.write(df.describe().to_string())
        f.write("\n\n=== Пропуски ===\n")
        f.write(df.isnull().sum().to_string())
        f.write(f"\n\n=== Баланс классов ===\n{df['Response'].value_counts().to_string()}")
        f.write(f"\nДоля позитивного класса (1): {df['Response'].mean():.4f}")

    print("✅ Общая информация сохранена в dataset_info.txt")

    # 2. Распределение целевого признака
    plt.figure(figsize=(6, 4))
    df['Response'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
    plt.title('Распределение целевого признака Response')
    plt.xlabel('Response')
    plt.ylabel('Количество')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
    plt.close()

    # 3. Корреляция с целевым признаком
    corr_with_target = df.corr(numeric_only=True)['Response'].drop('Response').sort_values(key=abs, ascending=False)
    corr_with_target.to_csv(os.path.join(output_dir, "correlation_with_target.csv"))

    plt.figure(figsize=(8, 12))
    corr_with_target.plot(kind='barh')
    plt.title('Корреляция признаков с Response')
    plt.xlabel('Корреляция (Пирсон)')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_with_target.png"), dpi=150)
    plt.close()

    # 4. Распределения непрерывных признаков по классам
    continuous_cols = [
        'Income', 'Age', 'Customer_Days', 'Recency',
        'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
        'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]

    for col in continuous_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 4))
        plt.hist(df[df['Response'] == 0][col].dropna(), bins=30, alpha=0.7, label='Response = 0', color='skyblue')
        plt.hist(df[df['Response'] == 1][col].dropna(), bins=30, alpha=0.7, label='Response = 1', color='orange')
        plt.title(f'Распределение {col} по классам Response')
        plt.xlabel(col)
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"), dpi=150)
        plt.close()

    print(f"✅ EDA завершён! Все файлы сохранены в '{output_dir}'")

if __name__ == "__main__":
    run_eda()