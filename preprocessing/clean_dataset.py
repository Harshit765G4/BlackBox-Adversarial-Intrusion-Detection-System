import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_csv("dataset/processed/master_dataset.csv")

print("Original Shape:", df.shape)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Replace infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Count missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Drop rows with NaN
df.dropna(inplace=True)

print("\nAfter NaN Removal:", df.shape)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("After Duplicate Removal:", df.shape)

# Save cleaned dataset
df.to_csv(
    "dataset/processed/cleaned_dataset.csv",
    index=False
)

print("\ncleaned_dataset.csv saved successfully!")