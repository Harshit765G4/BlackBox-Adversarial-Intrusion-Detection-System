import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/fgsm_results.csv"
)

plt.figure(figsize=(8,5))

plt.plot(
    df["epsilon"],
    df["attack_accuracy"],
    marker="o"
)

plt.title(
    "FGSM Attack Strength vs IDS Accuracy"
)

plt.xlabel("Epsilon")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(
    "results/fgsm_accuracy_curve.png"
)

plt.show()

print(
    "Graph saved successfully!"
)