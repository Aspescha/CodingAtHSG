"""
train.py
--------
One-time script to train the WatchPriceModel and save it to disk.
Run this before starting the Streamlit app.

Usage:
    python train.py
"""

import os
from src.cleaner import WatchDataCleaner
from src.model import WatchPriceModel

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_PATH  = "data/Watches.csv"
MODEL_PATH = "models/watch_price_model.joblib"

# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------

print("Step 1: Loading and cleaning data...")
cleaner = WatchDataCleaner(DATA_PATH)
df = cleaner.clean()
print(f"Dataset ready: {df.shape[0]:,} rows, {df.shape[1]} columns")

print("\nStep 2: Training model (this takes a few minutes)...")
model = WatchPriceModel()
model.fit(df)

print("\nStep 3: Evaluating model...")
results = model.evaluate()

print("\nStep 4: Saving model...")
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

print("\nDone! You can now run the Streamlit app with:")
print("    streamlit run app.py")