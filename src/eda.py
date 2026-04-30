"""
eda.py
------
Analyse exploratoire des données (EDA) :
visualisations de la distribution des features,
corrélations, et patterns liés aux attaques.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH  = "data/cybersecurity_intrusion_data.csv"
OUTPUT_DIR = "outputs/"

NUM_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
]


def load(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["encryption_used"] = df["encryption_used"].fillna("None")
    print(f"[eda] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ── 1. Distribution de la cible ──────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    counts = df["attack_detected"].value_counts()
    labels = ["Normal", "Attaque"]
    colors = ["steelblue", "tomato"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=10)
    ax.set_title("Distribution des classes")
    ax.set_ylabel("Nombre de sessions")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}eda_classes.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eda] Distribution des classes → {path}")


# ── 2. Heatmap de corrélation ────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame):
    corr = df[NUM_COLS + ["attack_detected"]].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation between numerical features")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}eda_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eda] Heatmap → {path}")


# ── 3. Boxplots par classe ───────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    palette = {0: "steelblue", 1: "tomato"}

    for i, col in enumerate(NUM_COLS):
        sns.boxplot(
            data=df, x="attack_detected", y=col,
            hue="attack_detected", palette=palette,
            ax=axes[i], width=0.5, legend=False,
        )
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Normal", "Attaque"])
        axes[i].set_xlabel("")
        axes[i].set_title(col.replace("_", " ").title())

    # Supprimer le 6e subplot vide
    fig.delaxes(axes[5])
    fig.suptitle("Distribution des features numériques par classe", fontsize=13, y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}eda_boxplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eda] Boxplots → {path}")


# ── 4. Features catégorielles vs attaque ─────────────────────────────────────

def plot_categorical(df: pd.DataFrame):
    cat_cols = ["protocol_type", "encryption_used", "browser_type"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, cat_cols):
        ct = df.groupby([col, "attack_detected"]).size().unstack(fill_value=0)
        ct.columns = ["Normal", "Attaque"]
        ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"],
                width=0.6, rot=0)
        ax.set_title(f"Attacks by {col.replace('_', ' ')}")
        ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}eda_categorical.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eda] Features catégorielles → {path}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load(DATA_PATH)

    print("\n--- Statistiques descriptives ---")
    print(df[NUM_COLS].describe().round(2))
    print("\n--- Valeurs manquantes ---")
    print(df.isnull().sum())

    plot_class_distribution(df)
    plot_heatmap(df)
    plot_boxplots(df)
    plot_categorical(df)

    print("\nEDA terminée. Visualisations dans outputs/")