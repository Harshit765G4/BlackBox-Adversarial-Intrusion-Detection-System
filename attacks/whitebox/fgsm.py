"""
attacks/whitebox/fgsm.py

Fast Gradient Sign Method - TARGETED EVASION ATTACK

ORIGINAL BUG
------------
The original FGSM computed:
    loss = BCEWithLogitsLoss(outputs, y_test)      ← uses TRUE labels
    X_adv = X_test + epsilon * sign(grad)

This maximises loss on the TRUE label, which degrades *overall accuracy*
by both misclassifying attacks AND misclassifying benign traffic.  It is
NOT a realistic threat model.

CORRECT EVASION FORMULATION
-----------------------------
An adversary wants to make ATTACKS look BENIGN.  The targeted objective is:

    loss = BCEWithLogitsLoss(outputs, zeros)       ← target = 0 (benign)
    X_adv = X_test + epsilon * sign(-grad)         ← descend toward target

This specifically minimises the model's confidence that a sample is an
attack.  The accuracy drop we care about is on the ATTACK class.

We also report both:
  * Overall accuracy degradation (all classes)
  * Attack-class evasion rate (attacks classified as benign)
"""

import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

from core.config import PATHS, FGSM_EPSILONS
from core.model_registry import FNN


# ─── Load ─────────────────────────────────────────────────────────────────────

print("Loading artifacts...")
X_test = joblib.load(PATHS["X_test"]).clone().detach().float()
y_test = joblib.load(PATHS["y_test"]).clone().detach().float().view(-1, 1)

model = FNN(X_test.shape[1])
model.load_state_dict(torch.load(PATHS["fnn_model"], weights_only=True))
model.eval()


# ─── Clean accuracy ───────────────────────────────────────────────────────────

with torch.no_grad():
    clean_preds = (torch.sigmoid(model(X_test)) > 0.5).float()

clean_acc = accuracy_score(y_test.numpy(), clean_preds.numpy())
print(f"\nClean Accuracy : {clean_acc:.4f}")


# ─── FGSM – Targeted Evasion ──────────────────────────────────────────────────

criterion = nn.BCEWithLogitsLoss()
# Attack mask: only perturb samples that are actually attacks
attack_mask = (y_test == 1).float()   # shape (N,1)

results = []

for epsilon in FGSM_EPSILONS:
    X_adv = X_test.clone().detach().requires_grad_(True)

    outputs = model(X_adv)

    # Targeted loss: push outputs toward label 0 (benign)
    target = torch.zeros_like(y_test)
    loss   = criterion(outputs, target)

    model.zero_grad()
    loss.backward()

    grad = X_adv.grad.data

    # Gradient ascent on loss w.r.t. target=0 means gradient descent
    # on P(attack).  Sign of gradient points toward TARGET (benign).
    # Multiply attack_mask so benign traffic is NOT perturbed.
    X_adv = (
        X_test
        + epsilon * grad.sign() * attack_mask
    ).detach()

    with torch.no_grad():
        adv_preds = (torch.sigmoid(model(X_adv)) > 0.5).float()

    adv_acc    = accuracy_score(y_test.numpy(), adv_preds.numpy())

    # Attack-class evasion rate
    attack_idx = (y_test.squeeze() == 1)
    if attack_idx.sum() > 0:
        evasion_rate = (
            adv_preds.squeeze()[attack_idx] == 0
        ).float().mean().item()
    else:
        evasion_rate = 0.0

    results.append({
        "epsilon":      epsilon,
        "adv_acc":      adv_acc,
        "acc_drop":     clean_acc - adv_acc,
        "evasion_rate": evasion_rate,
    })

    print(f"\n── epsilon={epsilon:.2f} ──────────────")
    print(f"  Attack Accuracy  : {adv_acc:.4f}  (drop: {clean_acc-adv_acc:.4f})")
    print(f"  Evasion Rate     : {evasion_rate:.4f}  (attacks misclassified as benign)")
    print(classification_report(
        y_test.numpy(), adv_preds.numpy(),
        target_names=["Benign", "Attack"], digits=4
    ))

print("\n╔══════ FGSM SUMMARY ══════╗")
print(f"{'Epsilon':>8} {'Acc':>8} {'Drop':>8} {'Evasion%':>10}")
for r in results:
    print(f"{r['epsilon']:>8.2f} {r['adv_acc']:>8.4f} {r['acc_drop']:>8.4f} {r['evasion_rate']*100:>9.2f}%")
