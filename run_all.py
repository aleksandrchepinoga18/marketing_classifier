# run_all.py
import os
import sys

# Добавляем папку src в путь
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Проверка (без эмодзи)
print("sys.path includes src:", src_path in sys.path)
print("Files in src:", os.listdir(src_path) if os.path.exists(src_path) else "NOT FOUND")

try:
    from eda import run_eda
    from train_final import main as train_model
except ImportError as e:
    print(f"ERROR: {e}")
    print("Make sure src/ contains: eda.py, train_final.py, __init__.py")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting full pipeline: EDA -> Model Training")
    print("="*60)
    
    print("\n1. Running EDA...")
    run_eda(data_path="ifood_df.csv", output_dir="reports/eda")
    print("EDA completed\n")
    
    print("2. Training final model...")
    train_model()
    print("\nPipeline finished successfully!")
    print("Results:")
    print("   - EDA: reports/eda/")
    print("   - Model: reports/model/")