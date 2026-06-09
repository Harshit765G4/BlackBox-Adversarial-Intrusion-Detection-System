import os

DATASET_PATH = "dataset/raw"

files = os.listdir(DATASET_PATH)

print("\n===== DATASET FILES =====\n")

for file in files:
    print(file)

print(f"\nTotal Files Found: {len(files)}")