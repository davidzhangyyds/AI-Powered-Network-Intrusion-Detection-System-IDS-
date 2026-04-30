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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, fbeta_score, precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import run_pipeline

# ── Paramètres ──────────────────────────────────────────────────────────────

DATA_PATH    = "data/cybersecurity_intrusion_data.csv"
MODEL_PATH   = "models/model.pkl"
EXPERIMENT   = "cybersecurity_intrusion_detection"




# ── Définition des modèles ───────────────────────────────────────────────────

def build_models():
    # Cost-sensitive learning: penalize false negatives (missed attacks) 2x more than false positives
    # This reduces false negatives while maintaining reasonable false positives
    dt     = DecisionTreeClassifier(max_depth=10, class_weight={0: 1, 1: 2}, random_state=42)
    rf     = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 2}, random_state=42)
    knn    = KNeighborsClassifier(17)  # KNN doesn't support class_weight
    model_lr = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 2})
    dt_plus_rf = VotingClassifier(estimators=[("rf", rf), ("dt", dt)], voting="soft")
    dt_plus_knn = VotingClassifier(estimators=[("dt", dt), ("knn", knn)], voting="soft")
    dt_plus_knn_plus_lr = VotingClassifier(estimators=[("dt", dt), ("knn", knn), ("lr", model_lr)], voting="soft")
    return [
        ("Decision Tree (Balanced)",    dt,     {"max_depth": 10, "class_weight": {0: 1, 1: 2}}),
        ("Random Forest (Balanced)",    rf,     {"n_estimators": 100, "class_weight": {0: 1, 1: 2}}),
        ("Voting Ensemble rf+dt",  dt_plus_rf, {"voting": "soft", "estimators": "rf+dt", "class_weight": {0: 1, 1: 2}}),
        ("Voting Ensemble dt+knn",  dt_plus_knn, {"voting": "soft", "estimators": "dt+knn"}),
        ("Voting Ensemble dt+knn+lr",  dt_plus_knn_plus_lr, {"voting": "soft", "estimators": "dt+knn+lr", "class_weight": {0: 1, 1: 2}}),
    ]

# Force le modèle à punir plus sévèrement les erreurs sur les attaques (classe 1)
model_lr = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 2})
dt = DecisionTreeClassifier(max_depth=10, class_weight={0: 1, 1: 2}, random_state=42)
# ── Évaluation ───────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba, metric='f2', min_recall=0.80):
    """Find optimal threshold that achieves min_recall while maximizing precision.
    
    Strategy:
    1. Filter thresholds that achieve at least min_recall (80%)
    2. From those, pick the one with highest precision (lowest FP)
    3. Fallback to F2-score if no threshold meets min_recall
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find thresholds that achieve at least min_recall
    valid_recall_idx = np.where(recalls[:-1] >= min_recall)[0]
    
    if len(valid_recall_idx) > 0:
        # From thresholds with recall >= 80%, pick one with highest precision
        best_idx = valid_recall_idx[np.argmax(precisions[:-1][valid_recall_idx])]
        optimal_threshold = thresholds[best_idx]
    else:
        # Fallback: use F2-score if recall target can't be met
        # F2 weights recall 2x higher than precision
        f2_scores = 5 * (precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-10)
        idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[idx]
    
    return optimal_threshold


def evaluate(model, X_test, y_test, threshold=None, find_threshold=True) -> dict:
    """Evaluate with intelligent threshold selection."""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold if not provided
    if threshold is None and find_threshold:
        threshold = find_optimal_threshold(y_test, y_proba, metric='f1')
    elif threshold is None:
        threshold = 0.5
    
    y_pred_custom = (y_proba >= threshold).astype(int)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred_custom), 4),
        "recall":   round(recall_score(y_test, y_pred_custom), 4),
        "precision": round(precision_score(y_test, y_pred_custom), 4),
        "f1_score": round(f1_score(y_test, y_pred_custom), 4),
        "f2_score": round(fbeta_score(y_test, y_pred_custom, beta=2), 4),
        "auc":      round(roc_auc_score(y_test, y_proba), 4),
        "threshold": round(threshold, 4),
    }


# ── Boucle d'entraînement avec MLflow ────────────────────────────────────────

#Finding the optimal K for KNN
def find_k(X_train, y_train):
    # Je pars du principe que tu as déjà X_train, X_test, y_train, y_test

    erreurs = []
    # On teste les k impairs de 1 à 100 (pour encadrer ton fameux 85)
    k_range = range(1, 100, 2) 

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # On calcule l'erreur (1 - précision)
        erreur = 1 - accuracy_score(y_test, y_pred)
        erreurs.append(erreur)

    # Affichage du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, erreurs, marker='o', linestyle='dashed', color='blue', markerfacecolor='red')
    plt.title('Taux d\'erreur en fonction de la valeur de K')
    plt.xlabel('Valeur de K')
    plt.ylabel('Erreur moyenne')
    plt.show()


#Training with cross-validation and logging to MLflow
def train_all(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    best_model  = None
    best_recall    = 0.0
    best_name   = ""

    # Configuration de la validation croisée stratifiée (5 sous-ensembles)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model, params in build_models():
        print(f"\n[train] {name}...")
        with mlflow.start_run(run_name=name):
            
            # --- 1. ÉVALUATION ROBUSTE : Stratified K-Fold Cross Validation ---
            print("  -> Lancement de la validation croisée (Stratified 5-Fold)...")
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv, 
                scoring={'accuracy': 'accuracy', 
                         'f1': 'f1', 
                         'auc': 'roc_auc',
                         'recall': 'recall',},
                n_jobs=-1 # Utilise tous les cœurs du processeur pour aller plus vite
            )
            
            # Calcul de la moyenne des scores sur les 5 entraînements
            cv_mean_acc = np.mean(cv_results['test_accuracy'])
            cv_mean_f1  = np.mean(cv_results['test_f1'])
            cv_mean_auc = np.mean(cv_results['test_auc'])
            cv_mean_recall = np.mean(cv_results['test_recall'])

            # --- 2. ENTRAÎNEMENT FINAL sur l'ensemble de X_train ---
            model.fit(X_train, y_train)
            
            # --- 3. ÉVALUATION FINALE sur le vrai X_test isolé ---
            metrics = evaluate(model, X_test, y_test)

            # --- 4. LOGS MLFLOW (On loggue les résultats CV et les résultats finaux) ---
            mlflow.log_params(params)
            mlflow.log_metrics({
                "test_accuracy": metrics["accuracy"],
                "test_recall": metrics["recall"],   # CRITICAL: Minimize false negatives
                "test_precision": metrics["precision"],  # Track false positives
                "test_f1_score": metrics["f1_score"],
                "test_f2_score": metrics["f2_score"],
                "test_auc": metrics["auc"],
                "optimal_threshold": metrics["threshold"],
                "cv_mean_recall": round(cv_mean_recall, 4),
                "cv_mean_accuracy": round(cv_mean_acc, 4),
                "cv_mean_f1_score": round(cv_mean_f1, 4),
                "cv_mean_auc": round(cv_mean_auc, 4)
            })

            # Save model for future Docker deployment
            mlflow.sklearn.log_model(model, artifact_path=name.replace(" ", "_"))

            # Affichage dans la console
            print(f"  [CV 5-Fold] Mean Accuracy : {cv_mean_acc:.4f}")
            print(f"  [CV 5-Fold] Mean F1-Score : {cv_mean_f1:.4f}")
            print(f"  [CV 5-Fold] Mean AUC      : {cv_mean_auc:.4f}")
            print(f"  [CV 5-Fold] Mean Recall   : {cv_mean_recall:.4f}")
            print(f"  [Hold-out]  Test Accuracy : {metrics['accuracy']:.4f}")
            print(f"  [Hold-out]  Test Recall   : {metrics['recall']:.4f}")
            print(f"  [Hold-out]  Test Precision: {metrics['precision']:.4f}")
            print(f"  [Hold-out]  Test F1-Score : {metrics['f1_score']:.4f}")
            print(f"  [Hold-out]  Optimal Threshold: {metrics['threshold']:.4f}")

            # L'élection du meilleur modèle se fait sur la métrique la plus robuste (CV_AUC)
            if cv_mean_recall > best_recall:
                best_recall = cv_mean_recall
                best_model = model
                best_name  = name

    print(f"\n[train] Meilleur modèle (élu par validation croisée) : {best_name} (Mean Recall = {best_recall:.4f})")
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
