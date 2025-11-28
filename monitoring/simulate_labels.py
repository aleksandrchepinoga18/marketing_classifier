# monitoring/simulate_labels.py
import pandas as pd
import numpy as np

def simulate_actual_labels():
    df = pd.read_csv("monitoring/predictions.csv")
    
    np.random.seed(42)
    df["actual_response"] = np.random.choice(
        [0, 1], 
        size=len(df), 
        p=[0.85, 0.15]
    )
    
    df.to_csv("monitoring/predictions_with_labels.csv", index=False)
    print(f"SUCCESS: Simulated {df['actual_response'].sum()} responses out of {len(df)} clients")
    print("FILE SAVED: monitoring/predictions_with_labels.csv")

if __name__ == "__main__":
    simulate_actual_labels()