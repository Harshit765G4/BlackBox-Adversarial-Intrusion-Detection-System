"""
attacks/whitebox/pgd.py

Projected Gradient Descent – TARGETED EVASION ATTACK

ORIGINAL BUG
------------
Same issue as fgsm.py: the loss was computed w.r.t. true labels
(untargeted), degrading both attack and benign accuracy.

PGD IMPROVEMENTS
----------------
1. Targeted objective: push P(attack) toward 0.
2. Only perturb attack samples (realistic threat model).
3. Random restarts (n_restarts=5) to escape local optima.
4. L∞ projection enforces a physically-motivated perturbation budget.
5. Reports per-class metrics and evasion rate.
"""

import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from core.config import PATHS, PGD_CONFIG
from core.model_registry import FNN


# ─── Load ─────────────────────────────────────────────────────────────────────

print("Loading artifacts...")
X_test = joblib.load(PATHS["X_test"]).clone().detach().float()
y_test = joblib.load(PATHS["y_test"]).clone().detach().float().view(-1, 1)

model = FNN(X_test.shape[1])
model.load_state_dict(torch.load(PATHS["fnn_model"], weights_only=True))
model.eval()

criterion  = nn.BCEWithLogitsLoss()
epsilon    = PGD_CONFIG["epsilon"]
alpha      = PGD_CONFIG["alpha"]
iterations = PGD_CONFIG["iterations"]
n_restarts = 5                         # random restarts

attack_mask = (y_test == 1).float()   # only perturb real attacks


# ─── Clean accuracy ───────────────────────────────────────────────────────────

with torch.no_grad():
    clean_preds = (torch.sigmoid(model(X_test)) > 0.5).float()

clean_acc = accuracy_score(y_test.numpy(), clean_preds.numpy())
print(f"Clean Accuracy : {clean_acc:.4f}\n")
print(f"Running PGD (ε={epsilon}, α={alpha}, iters={iterations}, restarts={n_restarts})...")


# ─── PGD helper ───────────────────────────────────────────────────────────────

def pgd_attack(X_orig: torch.Tensor) -> torch.Tensor:
    """Return best adversarial sample across restarts."""
    target     = torch.zeros_like(y_test)   # target: benign
    best_adv   = X_orig.clone()
    best_loss  = torch.full((X_orig.shape[0], 1), -1e9)

    for restart in range(n_restarts):
        # Random initialisation within ε-ball
        delta = torch.zeros_like(X_orig)
        delta.uniform_(-epsilon, epsilon)
        delta = delta * attack_mask        # only attack rows
        X_adv = (X_orig + delta).detach()

        for _ in range(iterations):
            X_adv.requires_grad_(True)

            outputs = model(X_adv)
            loss    = criterion(outputs, target)   # targeted loss

            model.zero_grad()
            loss.backward()

            grad  = X_adv.grad.data
            # Step toward lower P(attack)
            X_adv = X_adv + alpha * grad.sign() * attack_mask
            # Project into ε-ball
            eta   = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = (X_orig + eta * attack_mask).detach()

        # Track best adversarial by per-sample loss
        with torch.no_grad():
            r_loss = criterion(model(X_adv), target).item()
            out    = model(X_adv)
            per_loss = torch.abs(out)              # proxy: dist from 0 logit
        
        improve = per_loss > best_loss
        best_adv  = torch.where(improve.expand_as(best_adv), X_adv, best_adv)
        best_loss = torch.where(improve, per_loss, best_loss)

    return best_adv.detach()


# ─── Run attack ───────────────────────────────────────────────────────────────

X_adv = pgd_attack(X_test)

with torch.no_grad():
    adv_preds = (torch.sigmoid(model(X_adv)) > 0.5).float()

adv_acc = accuracy_score(y_test.numpy(), adv_preds.numpy())

attack_idx   = (y_test.squeeze() == 1)
evasion_rate = (
    (adv_preds.squeeze()[attack_idx] == 0).float().mean().item()
    if attack_idx.sum() > 0 else 0.0
)

print("\n╔══════ PGD RESULTS ══════╗")
print(f"  ε={epsilon}  α={alpha}  iters={iterations}  restarts={n_restarts}")
print(f"  Clean Accuracy   : {clean_acc:.4f}")
print(f"  Attack Accuracy  : {adv_acc:.4f}  (drop: {clean_acc-adv_acc:.4f})")
print(f"  Evasion Rate     : {evasion_rate:.4f}  (attacks → benign)")
print()
print(classification_report(
    y_test.numpy(), adv_preds.numpy(),
    target_names=["Benign", "Attack"], digits=4
))
