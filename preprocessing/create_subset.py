import pandas as pd

print("Loading dataset...")

df = pd.read_csv(
    "dataset/processed/cleaned_dataset.csv"
)

print("Original Shape:", df.shape)

subset = df.sample(
    n=500000,
    random_state=42
)

print("Subset Shape:", subset.shape)

subset.to_csv(
    "dataset/processed/fnn_dataset.csv",
    index=False
)

print("Subset saved successfully!")