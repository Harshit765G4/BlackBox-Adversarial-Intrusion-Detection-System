import joblib
import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# Model Definition
# =====================================

class FNN(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim,256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.network(x)

# =====================================
# Load Artifacts
# =====================================

print("Loading artifacts...")

X_test = joblib.load(
    "artifacts/X_test.pkl"
)

y_test = joblib.load(
    "artifacts/y_test.pkl"
)

X_test = X_test.clone().detach().float()

y_test = (
    y_test.clone()
    .detach()
    .float()
    .view(-1,1)
)

# =====================================
# Load Model
# =====================================

model = FNN(
    X_test.shape[1]
)

model.load_state_dict(
    torch.load(
        "models/fnn_v2_model.pth",
        weights_only=True
    )
)

model.eval()

# =====================================
# Clean Accuracy
# =====================================

with torch.no_grad():

    outputs = model(X_test)

    probs = torch.sigmoid(outputs)

    clean_preds = (
        probs > 0.5
    ).float()

clean_acc = accuracy_score(
    y_test.numpy(),
    clean_preds.numpy()
)

print(f"\nClean Accuracy: {clean_acc:.4f}")

# =====================================
# FGSM Experiment
# =====================================

epsilons = [
    0.01,
    0.03,
    0.05,
    0.07,
    0.10
]

results = []

criterion = nn.BCEWithLogitsLoss()

for epsilon in epsilons:

    print(f"\nRunning FGSM (epsilon={epsilon})")

    X_adv = (
        X_test.clone()
        .detach()
        .requires_grad_(True)
    )

    outputs = model(X_adv)

    loss = criterion(
        outputs,
        y_test
    )

    model.zero_grad()

    loss.backward()

    gradient = X_adv.grad.data

    X_adv = (
        X_adv
        + epsilon
        * gradient.sign()
    )

    with torch.no_grad():

        outputs_adv = model(X_adv)

        probs_adv = torch.sigmoid(
            outputs_adv
        )

        adv_preds = (
            probs_adv > 0.5
        ).float()

    acc = accuracy_score(
        y_test.numpy(),
        adv_preds.numpy()
    )

    precision = precision_score(
        y_test.numpy(),
        adv_preds.numpy()
    )

    recall = recall_score(
        y_test.numpy(),
        adv_preds.numpy()
    )

    f1 = f1_score(
        y_test.numpy(),
        adv_preds.numpy()
    )

    drop = clean_acc - acc

    results.append({
        "epsilon": epsilon,
        "clean_accuracy": clean_acc,
        "attack_accuracy": acc,
        "accuracy_drop": drop,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

# =====================================
# Save Results
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/fgsm_results.csv",
    index=False
)

print("\nResults Saved!")

print("\n========================")
print(results_df)
print("========================")