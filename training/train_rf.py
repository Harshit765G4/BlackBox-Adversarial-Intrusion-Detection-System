import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Loading cleaned dataset...")

df = pd.read_csv(
    "dataset/processed/cleaned_dataset.csv"
)

# Remove spaces
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)

# Features
X = df.drop("Label", axis=1)

# Target
y = df["Label"]

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Predicting...")

y_pred = rf.predict(X_test)

print("\nResults\n")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred))

print("Recall:",
      recall_score(y_test, y_pred))

print("F1 Score:",
      f1_score(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(
    rf,
    "models/rf_model.pkl"
)

print("\nModel saved successfully!")