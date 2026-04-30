"""
evaluate.py
-----------
Calcul et affichage de toutes les métriques d'évaluation
pour chaque modèle entraîné : rapport de classification,
courbe ROC, matrice de confusion.
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # mode sans interface graphique

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,         
)

from preprocessing import run_pipeline

DATA_PATH  = "data/cybersecurity_intrusion_data.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_DIR = "outputs/"


# ── Chargement ───────────────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[load] Modèle chargé depuis {path}")
    return model


# ── Métriques textuelles ─────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba, min_recall=0.80):
    import numpy as np
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # On s'assure de ne pas prendre le dernier élément (qui n'a pas de seuil associé)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Masque pour les seuils respectant le rappel minimum
    valid_recall_mask = recalls >= min_recall
    
    if np.any(valid_recall_mask):
        # On filtre et on prend l'indice du max de précision parmi ceux valides
        best_idx = np.argmax(precisions[valid_recall_mask])
        # Attention : l'index 'best_idx' est relatif au masque, il faut retrouver l'index réel
        optimal_threshold = thresholds[valid_recall_mask][best_idx]
    else:
        # Fallback F2 plus robuste
        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
        optimal_threshold = thresholds[np.argmax(f2_scores)]
    
    return optimal_threshold


def print_report(model, X_test, y_test, threshold=None):
    """Print classification report using optimal threshold to balance FN and FP."""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_test, y_proba)
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    print("\n" + "=" * 50)
    print("  Rapport de classification")
    print("=" * 50)
    print(f"  Optimal Threshold: {threshold:.4f}")
    print("=" * 50)
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal (0)", "Attaque (1)"]
    ))
    print(f"AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    return y_pred, y_proba, threshold


# ── Courbe ROC ───────────────────────────────────────────────────────────────

def plot_roc(model, X_test, y_test, save_path: str = f"{OUTPUT_DIR}roc_curve.png"):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    ax.set_title("Courbe ROC — Meilleur modèle")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Courbe ROC sauvegardée → {save_path}")


# ── Matrice de confusion ─────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, threshold, save_path: str = f"{OUTPUT_DIR}confusion_matrix.png"):
    """Plot confusion matrix using the optimal threshold."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Attaque"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (Threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Matrice de confusion sauvegardée → {save_path}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _, X_test, _, y_test = run_pipeline(DATA_PATH)
    model = load_model()

    y_pred, y_proba, threshold = print_report(model, X_test, y_test)
    plot_roc(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test, threshold)

    print("\nÉvaluation terminée. Fichiers sauvegardés dans outputs/")
