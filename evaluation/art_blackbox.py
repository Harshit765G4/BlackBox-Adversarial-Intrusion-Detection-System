# In a new file: evaluation/art_blackbox.py
import numpy as np
import joblib
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

from core.model_registry import get_rf
from core.preprocessor import preprocess
import pandas as pd

rf = get_rf()
classifier = SklearnClassifier(model=rf, clip_values=(0, 1))

# Load attack samples only
import pandas as pd
df = pd.read_csv("dataset/processed/fnn_dataset.csv")
df.columns = df.columns.str.strip()
X_attack = preprocess(df[df["Label"] == 1].drop("Label", axis=1))

# Run HopSkipJump (query-based, no gradients)
attack = HopSkipJump(
    classifier=classifier,
    targeted=False,
    max_iter=50,
    max_eval=1000,
    init_eval=100,
)
X_adv = attack.generate(X_attack.values[:100])  # start with 100 samples

# Evaluate
preds = rf.predict(X_adv)
evasion_rate = (preds == 0).mean()
print(f"HopSkipJump evasion rate: {evasion_rate:.4f}")