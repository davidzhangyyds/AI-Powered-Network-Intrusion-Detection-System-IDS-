"""
train.py
--------
Entraînement des modèles ML, comparaison et sérialisation
du meilleur modèle pour le déploiement.
"""

import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from preprocessing import run_pipeline


# ── Paramètres ──────────────────────────────────────────────────────────────

DATA_PATH    = "data/cybersecurity_intrusion_data.csv"
MODEL_PATH   = "models/model.pkl"
EXPERIMENT   = "cybersecurity_intrusion_detection"


# ── Définition des modèles ───────────────────────────────────────────────────

def build_models():
    dt     = DecisionTreeClassifier(max_depth=10, random_state=42)
    rf     = RandomForestClassifier(n_estimators=100, random_state=42)
    voting = VotingClassifier(estimators=[("rf", rf), ("dt", dt)], voting="soft")
    return [
        ("Decision Tree",    dt,     {"max_depth": 10}),
        ("Random Forest",    rf,     {"n_estimators": 100}),
        ("Voting Ensemble",  voting, {"voting": "soft", "estimators": "rf+dt"}),
    ]


# ── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc":      round(roc_auc_score(y_test, y_proba), 4),
    }


# ── Boucle d'entraînement avec MLflow ────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test):
    mlflow.set_experiment(EXPERIMENT)
    best_model  = None
    best_auc    = 0.0
    best_name   = ""

    for name, model, params in build_models():
        print(f"\n[train] {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            # Log MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            #Save model for futur Docker deployment
            mlflow.sklearn.log_model(model, name=name.replace(" ", "_"))

            print(f"  accuracy : {metrics['accuracy']:.4f}")
            print(f"  f1_score : {metrics['f1_score']:.4f}")
            print(f"  auc      : {metrics['auc']:.4f}")

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                best_model = model
                best_name  = name

    print(f"\n[train] Meilleur modèle : {best_name} (AUC = {best_auc:.4f})")
    return best_model, best_name


# ── Sauvegarde ───────────────────────────────────────────────────────────────

def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[save] Modèle sauvegardé → {path}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Pipeline d'entraînement — Intrusion Detection")
    print("=" * 50)

    X_train, X_test, y_train, y_test = run_pipeline(DATA_PATH)
    best_model, best_name = train_all(X_train, X_test, y_train, y_test)
    save_model(best_model)

    print("\nEntraînement terminé.")
    print("Lance 'mlflow ui' pour visualiser les expériences.")
