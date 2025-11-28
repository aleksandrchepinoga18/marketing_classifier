# src/data.py
import pandas as pd

def load_and_clean_data(data_path: str = "ifood_df.csv"):
    """
    Загружает и очищает данные.
    Возвращает исходный df БЕЗ разделения.
    """
    df = pd.read_csv(data_path)

    # Удаляем известные бесполезные/избыточные признаки
    cols_to_drop = [
        'Z_CostContact', 'Z_Revenue', 'MntTotal',
        'MntRegularProds', 'AcceptedCmpOverall'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    return df