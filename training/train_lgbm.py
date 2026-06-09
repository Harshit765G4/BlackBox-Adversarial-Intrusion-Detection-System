import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("Loading dataset...")

df = pd.read_csv(
    "dataset/processed/cleaned_dataset.csv"
)

df.columns = df.columns.str.strip()

X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nResults\n")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1:",
    f1_score(y_test, y_pred)
)

print(
    "\nConfusion Matrix:\n",
    confusion_matrix(y_test, y_pred)
)

joblib.dump(
    model,
    "models/lgbm_model.pkl"
)

print("\nModel saved successfully!")