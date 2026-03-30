"""
preprocessing.py
----------------
Chargement, nettoyage et feature engineering du dataset
Cybersecurity Intrusion Detection.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Colonnes numériques à normaliser
NUM_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
]

# Colonnes catégorielles à encoder
CAT_COLS = ["protocol_type", "encryption_used", "browser_type"]


def load_data(filepath: str) -> pd.DataFrame:
    """Charge le CSV brut."""
    df = pd.read_csv(filepath)
    print(f"[load] {df.shape[0]} lignes, {df.shape[1]} colonnes chargées.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage de base :
    - Supprime session_id (identifiant sans valeur prédictive)
    - Impute les valeurs manquantes de encryption_used par 'None'
    """
    df = df.drop(columns=["session_id"])
    missing = df["encryption_used"].isnull().sum()
    df["encryption_used"] = df["encryption_used"].fillna("None")
    print(f"[clean] {missing} valeurs manquantes dans encryption_used → remplacées par 'None'.")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding des colonnes catégorielles."""
    df = pd.get_dummies(df, columns=CAT_COLS)
    print(f"[encode] {df.shape[1]} colonnes après encodage.")
    return df


def split_features_target(df: pd.DataFrame):
    """Sépare X et y."""
    X = df.drop(columns=["attack_detected"])
    y = df["attack_detected"]
    print(f"[split] Features : {X.shape[1]} colonnes | Cible : {y.value_counts().to_dict()}")
    return X, y


def normalize(X_train: pd.DataFrame, X_test: pd.DataFrame, save_path: str = "models/scaler.pkl"):
    """
    Ajuste un StandardScaler sur X_train et transforme X_train + X_test.
    Sauvegarde le scaler pour l'API Flask.
    """
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[normalize] Scaler sauvegardé → {save_path}")
    return X_train, X_test, scaler


def run_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline complet : charge → nettoie → encode → split → normalise.
    Retourne X_train, X_test, y_train, y_test prêts pour l'entraînement.
    """
    df = load_data(filepath)
    df = clean(df)
    df = encode(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split] Train : {len(X_train)} | Test : {len(X_test)}")

    X_train, X_test, _ = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_pipeline("data/cybersecurity_intrusion_data.csv")
    print("\nPrétraitement terminé.")
    print("X_train shape :", X_train.shape)
    print("X_test  shape :", X_test.shape)