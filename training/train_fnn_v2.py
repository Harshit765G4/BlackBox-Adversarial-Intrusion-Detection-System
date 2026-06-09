import pandas as pd
import numpy as np
import joblib

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

# =====================================
# Load Dataset
# =====================================

print("Loading dataset...")

df = pd.read_csv(
    "dataset/processed/fnn_dataset.csv"
)

df.columns = df.columns.str.strip()

X = df.drop("Label", axis=1)
y = df["Label"]

print(f"Dataset Shape: {df.shape}")

# =====================================
# Train/Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# =====================================
# Feature Scaling
# =====================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for FGSM / PGD attacks
joblib.dump(
    scaler,
    "models/fnn_scaler.pkl"
)

print("Scaler saved successfully!")

# =====================================
# Tensor Conversion
# =====================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)

# =====================================
# DataLoader
# =====================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True
)

# =====================================
# Class Weights
# =====================================

attack_count = y_train.sum().item()

benign_count = (
    len(y_train) - attack_count
)

pos_weight = torch.tensor(
    [benign_count / attack_count],
    dtype=torch.float32
)

print(f"\nAttack Samples : {attack_count}")
print(f"Benign Samples : {benign_count}")
print(f"Positive Weight: {pos_weight.item():.4f}")

# =====================================
# Neural Network
# =====================================

class FNN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = FNN(
    X_train.shape[1]
)

# =====================================
# Training Setup
# =====================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

best_loss = float("inf")

# =====================================
# Training
# =====================================

print("\nTraining FNN v2...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=False
    )

    for batch_x, batch_y in loop:

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.6f}"
    )

    # Save best model
    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "models/fnn_v2_best.pth"
        )

# =====================================
# Evaluation
# =====================================

print("\nEvaluating model...\n")

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    probs = torch.sigmoid(outputs)

    preds = (
        probs > 0.5
    ).float()

y_pred = preds.numpy()

acc = accuracy_score(
    y_test.numpy(),
    y_pred
)

precision = precision_score(
    y_test.numpy(),
    y_pred
)

recall = recall_score(
    y_test.numpy(),
    y_pred
)

f1 = f1_score(
    y_test.numpy(),
    y_pred
)

cm = confusion_matrix(
    y_test.numpy(),
    y_pred
)

print("\n========== RESULTS ==========\n")

print(f"Accuracy : {acc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")

print("\nConfusion Matrix:\n")
print(cm)

# =====================================
# Save Final Model
# =====================================

torch.save(
    model.state_dict(),
    "models/fnn_v2_model.pth"
)

# Save Metrics
metrics = {
    "accuracy": float(acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

joblib.dump(
    metrics,
    "results/fnn_v2_metrics.pkl"
)

joblib.dump(X_test, "artifacts/X_test.pkl")
joblib.dump(y_test, "artifacts/y_test.pkl")

print("\nModel saved successfully!")
print("Scaler saved successfully!")
print("Metrics saved successfully!")