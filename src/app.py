"""
app.py
------
API REST Flask pour servir les prédictions du modèle
de détection d'intrusion en temps réel.

Routes :
  GET  /health    → API status
  POST /predict   → prediction on a network session
"""

import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Chargement du modèle et du scaler ────────────────────────────────────────

with open("models/model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

# Colonnes numériques à normaliser (même ordre que preprocessing.py)
NUM_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
]

# Toutes les colonnes attendues après get_dummies (même ordre que X_train)
EXPECTED_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
    "unusual_time_access",
    "protocol_type_ICMP",
    "protocol_type_TCP",
    "protocol_type_UDP",
    "encryption_used_AES",
    "encryption_used_DES",
    "encryption_used_None",
    "browser_type_Chrome",
    "browser_type_Edge",
    "browser_type_Firefox",
    "browser_type_Safari",
    "browser_type_Unknown",
]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Checks if the API is operational."""
    return jsonify({"status": "ok", "model": type(MODEL).__name__}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts whether a network session is an attack.

    Expected JSON body (example) :
    {
        "network_packet_size": 599,
        "login_attempts": 3,
        "session_duration": 120.5,
        "ip_reputation_score": 0.85,
        "failed_logins": 2,
        "unusual_time_access": 1,
        "protocol_type": "TCP",
        "encryption_used": "AES",
        "browser_type": "Chrome"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Corps JSON manquant."}), 400

    try:
        # Construction du DataFrame à partir des valeurs brutes
        df = pd.DataFrame([{
            "network_packet_size": data.get("network_packet_size", 0),
            "login_attempts":      data.get("login_attempts", 0),
            "session_duration":    data.get("session_duration", 0.0),
            "ip_reputation_score": data.get("ip_reputation_score", 0.0),
            "failed_logins":       data.get("failed_logins", 0),
            "unusual_time_access": data.get("unusual_time_access", 0),
            "protocol_type":       data.get("protocol_type", "TCP"),
            "encryption_used":     data.get("encryption_used", "None"),
            "browser_type":        data.get("browser_type", "Unknown"),
        }])

        # One-hot encoding (même logique que preprocessing.py)
        df = pd.get_dummies(df, columns=["protocol_type", "encryption_used", "browser_type"])

        # Ajouter les colonnes manquantes avec 0 (si une catégorie est absente)
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0

        # Réordonner les colonnes dans le bon ordre
        df = df[EXPECTED_COLS]

        # Normalisation des colonnes numériques
        df[NUM_COLS] = SCALER.transform(df[NUM_COLS])

        # Prédiction
        prediction = int(MODEL.predict(df)[0])
        probability = float(MODEL.predict_proba(df)[0][1])
        message = "Attaque détectée" if prediction == 1 else "Session normale"

        return jsonify({
            "prediction":  prediction,
            "message":     message,
            "probability": round(probability, 4),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Lancement ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[api] Démarrage de l'API sur http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)