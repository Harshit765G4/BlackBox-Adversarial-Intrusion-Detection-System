import pandas as pd
import os

DATASET_PATH = "dataset/raw"

all_dfs = []

print("\nLoading files...\n")

for file in os.listdir(DATASET_PATH):

    if file.endswith(".csv"):

        path = os.path.join(DATASET_PATH, file)

        print(f"Loading {file}")

        df = pd.read_csv(path)

        all_dfs.append(df)

master_df = pd.concat(all_dfs, ignore_index=True)

print("\nDataset merged successfully!")

print("Shape:", master_df.shape)

# Remove extra spaces from column names
master_df.columns = master_df.columns.str.strip()

# Convert labels
master_df["Label"] = master_df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nLabel Distribution:\n")
print(master_df["Label"].value_counts())

# Save
master_df.to_csv(
    "dataset/processed/master_dataset.csv",
    index=False
)

print("\nmaster_dataset.csv created successfully!")