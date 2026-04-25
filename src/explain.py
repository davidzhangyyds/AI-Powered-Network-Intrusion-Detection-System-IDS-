"""
explain.py
----------
Interprétabilité du modèle via SHAP (SHapley Additive exPlanations).
Génère un summary plot global et un waterfall plot pour
une prédiction individuelle.
"""

import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance

from preprocessing import run_pipeline

DATA_PATH  = "data/cybersecurity_intrusion_data.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_DIR = "outputs/"


# ── Chargement ───────────────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Summary plot (importance globale) ────────────────────────────────────────

def plot_shap_summary(model, X_test: pd.DataFrame, n_samples: int = 200,
                      save_path: str = f"{OUTPUT_DIR}shap_summary.png"):
    """
    Affiche l'importance globale de chaque feature.
    Pour VotingClassifier: utilise permutation importance (rapide).
    Pour les autres modèles: utilise SHAP TreeExplainer.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_sample  = X_test.iloc[:n_samples]
    
    # Pour VotingClassifier, utilise permutation importance (beaucoup plus rapide)
    if isinstance(model, VotingClassifier):
        print("  [explain] VotingClassifier détecté → utilisation de Permutation Importance (rapide)...")
        
        # Use dummy target for permutation importance
        dummy_target = np.zeros(len(X_sample))
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X_sample, dummy_target,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create feature importance dataframe
        importances = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 7))
        top_n = min(15, len(importances))
        top_features = importances.head(top_n)
        plt.barh(range(top_n), top_features['importance'])
        plt.yticks(range(top_n), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title('Permutation Importance - Features pour la classe Attaque')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[explain] Feature importance sauvegardé → {save_path}")
        print("\n[explain] Top 5 features les plus influentes :")
        for idx, row in importances.head(5).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")
    else:
        # Use TreeExplainer for tree-based models
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"  [explain] TreeExplainer failed: {e}")
            return
        
        # Extract SHAP values for attack class (class 1)
        if isinstance(shap_vals, list):
            shap_attack = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_attack = shap_vals[:, :, 1]
        else:
            shap_attack = shap_vals

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_attack, X_sample, show=False)
        plt.title("SHAP - Importance des features (classe : Attaque)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explain] SHAP Summary plot sauvegardé → {save_path}")

        # Display top 5
        mean_abs = pd.Series(
            np.abs(shap_attack).mean(axis=0),
            index=X_sample.columns
        ).sort_values(ascending=False)
        print("\n[explain] Top 5 features les plus influentes :")
        for feat, val in mean_abs.head(5).items():
            print(f"  {feat:<35} {val:.4f}")


# ── Waterfall plot (explication individuelle) ─────────────────────────────────

def plot_shap_waterfall(model, X_test: pd.DataFrame, sample_idx: int = 0,
                        save_path: str = f"{OUTPUT_DIR}shap_waterfall.png"):
    """
    Explique une prédiction individuelle avec visualisation simple.
    Pour VotingClassifier: utilise un graphique de features importantes.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample = X_test.iloc[[sample_idx]]
    
    # Pour VotingClassifier, crée une visualisation simple des contributions
    if isinstance(model, VotingClassifier):
        # Get prediction for this sample
        pred_proba = model.predict_proba(sample)[0]
        attack_prob = pred_proba[1]
        
        # Get coefficients or importance from underlying estimators
        plt.figure(figsize=(10, 6))
        
        # Show feature values for this sample
        feature_vals = sample.iloc[0].values
        feature_names = sample.columns
        
        sorted_idx = np.argsort(np.abs(feature_vals))[-10:]
        colors = ['red' if feature_vals[i] > feature_vals.mean() else 'blue' for i in sorted_idx]
        
        plt.barh(range(len(sorted_idx)), feature_vals[sorted_idx], color=colors)
        plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
        plt.xlabel('Feature Value')
        plt.title(f'Feature Values for Session #{sample_idx} (Predicted: {attack_prob:.2%} attack)')
        plt.axvline(feature_vals.mean(), color='green', linestyle='--', label='Mean')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explain] Feature visualization sauvegardé → {save_path}")
    else:
        # Use SHAP waterfall for tree-based models
        try:
            explainer = shap.TreeExplainer(model)
            shap_exp = explainer(sample)
        except Exception as e:
            print(f"  [explain] TreeExplainer failed: {e}")
            return
        
        # Extract for attack class (class 1)
        if isinstance(shap_exp, list):
            shap_vals = shap_exp[1] if len(shap_exp) > 1 else shap_exp[0]
        elif hasattr(shap_exp, 'values'):
            if shap_exp.values.ndim == 3:
                shap_vals = shap_exp[:, :, 1]
            else:
                shap_vals = shap_exp.values
        else:
            shap_vals = shap_exp

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.title(f"SHAP - Explication session #{sample_idx}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explain] SHAP Waterfall plot sauvegardé → {save_path}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _, X_test, _, y_test = run_pipeline(DATA_PATH)
    model = load_model()

    print("[shap] Calcul des valeurs SHAP (peut prendre ~30 sec)...")
    plot_shap_summary(model, X_test)
    plot_shap_waterfall(model, X_test, sample_idx=0)

    print("\nAnalyse SHAP terminée. Fichiers sauvegardés dans outputs/")
