import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Chargement dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")

# Prétraitement rapide (à adapter selon ton dataset)
df = df.dropna()
X = df.drop("bank_account", axis=1)
y = df["bank_account"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde pickle
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "columns": list(X.columns)
    }, f)

print("Modèle enregistré sous model.pkl")
