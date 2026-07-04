"""API de prediction : charge le meilleur modele et produit une prediction
pour un nouveau profil client (utilise par app.py, page "Prediction en temps reel")."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from etl import CAT_COLS, ENCODERS_PATH, FEATURE_COLUMNS, NUM_COLS, SCALER_PATH, feature_engineering, transform_features

ROOT_DIR = Path(__file__).resolve().parent.parent
BEST_MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"


def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, encoders


def build_feature_row(client_profile: dict, encoders, scaler) -> pd.DataFrame:
    """client_profile doit contenir les colonnes brutes (age, sexe, taille_cm, ...)."""
    df = pd.DataFrame([client_profile])
    df = feature_engineering(df)
    df = transform_features(df, encoders, scaler)
    return df[FEATURE_COLUMNS]

def predict_client(client_profile: dict, model=None, scaler=None, encoders=None):
    if model is None or scaler is None or encoders is None:
        model, scaler, encoders = load_artifacts()

    X = build_feature_row(client_profile, encoders, scaler)
    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)

    if proba >= 0.70:
        interpretation = "Profil favorable"
    elif proba >= 0.40:
        interpretation = "Profil a risque, ajuster le programme"
    else:
        interpretation = "Profil critique, revoir les bases"

    return {"proba": proba, "prediction": prediction, "interpretation": interpretation}


if __name__ == "__main__":
    example = {
        "age": 29,
        "sexe": "H",
        "taille_cm": 178,
        "poids_initial_kg": 85,
        "poids_cible_kg": 78,
        "objectif": "seche",
        "niveau": "intermediaire",
        "frequence_entrainement_semaine": 4,
        "calories_quotidiennes": 2200,
        "proteines_g_par_jour": 170,
        "heures_sommeil": 7,
        "semaines_suivi_prevues": 12,
        "adherence_programme_pct": 80,
    }
    print(predict_client(example))
