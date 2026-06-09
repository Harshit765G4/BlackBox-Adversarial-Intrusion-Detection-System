import pandas as pd
import numpy as np
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

# ---------------------------
# Load Dataset
# ---------------------------

print("Loading dataset...")

df = pd.read_csv(
    "dataset/processed/fnn_dataset.csv"
)

df.columns = df.columns.str.strip()

X = df.drop("Label", axis=1)
y = df["Label"]

# ---------------------------
# Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# Feature Scaling
# ---------------------------

print("Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Convert to Tensor
# ---------------------------

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
).view(-1,1)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1,1)

# ---------------------------
# Neural Network
# ---------------------------

class FNN(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1),
            predictions = torch.sigmoid(
    model(X_test)
)
        )

    def forward(self,x):
        return self.network(x)

model = FNN(X_train.shape[1])

# ---------------------------
# Training Setup
# ---------------------------

attack_count = y_train.sum().item()
benign_count = len(y_train) - attack_count

pos_weight = torch.tensor(
    [benign_count / attack_count],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

print("\nTraining FNN...\n")

for epoch in tqdm(range(epochs)):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

# ---------------------------
# Evaluation
# ---------------------------

model.eval()

with torch.no_grad():

    predictions = model(X_test)

    predictions = (
        predictions > 0.5
    ).float()

y_pred = predictions.numpy()

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

print("\nResults\n")

print("Accuracy :", acc)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test.numpy(),
        y_pred
    )
)

# ---------------------------
# Save Model
# ---------------------------

torch.save(
    model.state_dict(),
    "models/fnn_model.pth"
)

print("\nModel saved successfully!")