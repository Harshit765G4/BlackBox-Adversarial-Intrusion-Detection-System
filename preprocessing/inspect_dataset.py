import pandas as pd

file_path = "dataset/raw/Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel Counts:")
print(df[' Label'].value_counts())

print("\nFirst 5 Rows:")
print(df.head())