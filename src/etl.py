"""
Pipeline ETL - Prediction de progression coaching fitness (@builtbyarthur)

Finalite (RGPD) : ce traitement vise a estimer la probabilite qu'un client
de coaching en ligne atteigne son objectif physique (seche / prise de masse /
recomposition), afin d'adapter l'accompagnement propose par le coach.

Principe de pseudonymisation (art. 4(5) RGPD) : le prenom du client (donnee
d'identification, utilisee uniquement pour l'affichage cote coach dans le
dashboard) est stocke dans une table d'identite separee
(data/raw/annuaire_clients.csv), qui ne fait JAMAIS partie du dataset servant
a l'entrainement ou a la prediction (data/processed/dataset_final.csv).
Le pipeline ML ne manipule que l'identifiant pseudonymise (CLIENT_XXX). Dans
un contexte reel (donnees non synthetiques), cette table d'identite devrait
etre stockee separement (acces restreint, hors depot de code partage) : voir
docs/RGPD_AI_ACT.md pour le detail des mesures et de l'analyse AI Act.

Etapes :
    1. Extraction   : generation / chargement des donnees brutes + inventaire
    2. Transformation : feature engineering, encodage, normalisation
    3. Chargement   : export du dataset final et des artefacts de pretraitement
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "clients_raw.csv"
ANNUAIRE_PATH = ROOT_DIR / "data" / "raw" / "annuaire_clients.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "dataset_final.csv"
SCALER_PATH = ROOT_DIR / "data" / "processed" / "scaler.pkl"
ENCODERS_PATH = ROOT_DIR / "data" / "processed" / "label_encoders.pkl"

PRENOMS_H = [
    "Lucas", "Hugo", "Nathan", "Thomas", "Maxime", "Antoine", "Julien", "Alexandre",
    "Kevin", "Romain", "Quentin", "Baptiste", "Mathieu", "Adrien", "Florian",
    "Guillaume", "Clement", "Vincent", "Simon", "Theo", "Gabriel", "Arthur",
    "Benjamin", "Nicolas", "Yanis", "Rayan", "Mehdi", "Karim", "Sofiane", "Bilal",
]
PRENOMS_F = [
    "Emma", "Lea", "Chloe", "Manon", "Camille", "Sarah", "Laura", "Julie",
    "Marion", "Pauline", "Charlotte", "Ines", "Lucie", "Justine", "Amandine",
    "Melanie", "Celia", "Oceane", "Margaux", "Elise", "Anais", "Clara",
    "Sofia", "Yasmine", "Nora", "Sabrina", "Lina", "Alicia", "Emilie", "Marine",
]
NOMS_FAMILLE = [
    "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Petit", "Durand", "Leroy",
    "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand",
    "Roux", "Vincent", "Fournier", "Morel", "Girard", "Andre", "Lefevre", "Mercier",
    "Dupont", "Lambert", "Bonnet", "Francois", "Martinez", "Legrand", "Garnier", "Faure",
    "Rousseau", "Blanc", "Guerin", "Muller", "Henry", "Roussel", "Nicolas", "Perrin",
]

CAT_COLS = ["sexe", "objectif", "niveau"]
NUM_COLS = [
    "age",
    "taille_cm",
    "poids_initial_kg",
    "poids_cible_kg",
    "frequence_entrainement_semaine",
    "calories_quotidiennes",
    "proteines_g_par_jour",
    "heures_sommeil",
    "semaines_suivi_prevues",
    "adherence_programme_pct",
    "imc",
    "ratio_proteines_poids",
    "besoin_calorique_estime",
    "deficit_calorique",
    "score_mode_de_vie",
]
TARGET_COL = "objectif_atteint"
FEATURE_COLUMNS = [f"{c}_scaled" for c in NUM_COLS] + [f"{c}_encoded" for c in CAT_COLS]


def _activity_factor(freq_par_semaine: np.ndarray) -> np.ndarray:
    """Facteur d'activite approximatif (NEAT + entrainement) selon la frequence hebdo."""
    return 1.25 + 0.055 * np.clip(freq_par_semaine, 0, 6)


def generate_raw_data(n_clients: int = 600, seed: int = 42) -> pd.DataFrame:
    """Genere un dataset synthetique realiste de n_clients clients de coaching fitness."""
    rng = np.random.default_rng(seed)

    client_id = [f"CLIENT_{i + 1:03d}" for i in range(n_clients)]
    sexe = rng.choice(["H", "F"], size=n_clients, p=[0.55, 0.45])
    age = np.clip(rng.normal(32, 8, n_clients), 18, 65).round().astype(int)

    taille_cm = np.where(
        sexe == "H",
        np.clip(rng.normal(178, 6.5, n_clients), 158, 202),
        np.clip(rng.normal(165, 6, n_clients), 148, 190),
    ).round(1)

    objectif = rng.choice(
        ["seche", "prise_masse", "recomposition"], size=n_clients, p=[0.42, 0.33, 0.25]
    )
    niveau = rng.choice(
        ["debutant", "intermediaire", "avance"], size=n_clients, p=[0.40, 0.40, 0.20]
    )

    poids_base = np.where(
        sexe == "H", rng.normal(84, 13, n_clients), rng.normal(68, 12, n_clients)
    )
    poids_initial_kg = np.clip(poids_base, 45, 160).round(1)

    poids_cible_pct = np.select(
        [objectif == "seche", objectif == "prise_masse", objectif == "recomposition"],
        [
            rng.uniform(-0.16, -0.05, n_clients),
            rng.uniform(0.03, 0.10, n_clients),
            rng.uniform(-0.03, 0.03, n_clients),
        ],
    )
    poids_cible_kg = np.clip(
        poids_initial_kg * (1 + poids_cible_pct), 40, 170
    ).round(1)

    frequence_entrainement_semaine = np.clip(
        rng.poisson(3.4, n_clients), 1, 6
    ).astype(int)

    bmr = (
        10 * poids_initial_kg
        + 6.25 * taille_cm
        - 5 * age
        + np.where(sexe == "H", 5, -161)
    )
    besoin_calorique_estime = (bmr * _activity_factor(frequence_entrainement_semaine)).round(0)

    calorie_offset_cible = np.select(
        [objectif == "seche", objectif == "prise_masse", objectif == "recomposition"],
        [
            -rng.normal(500, 180, n_clients),
            rng.normal(380, 160, n_clients),
            rng.normal(0, 220, n_clients),
        ],
    )
    calories_quotidiennes = np.clip(
        besoin_calorique_estime + calorie_offset_cible, 1200, 4500
    ).round(0)

    proteines_g_par_jour = np.clip(
        poids_initial_kg * rng.uniform(1.2, 2.4, n_clients), 60, 320
    ).round(0)

    heures_sommeil = np.clip(rng.normal(6.8, 1.1, n_clients), 3.5, 10).round(1)
    semaines_suivi_prevues = rng.integers(4, 25, n_clients)
    adherence_programme_pct = np.clip(
        rng.beta(5, 2.2, n_clients) * 100, 10, 100
    ).round(1)

    df = pd.DataFrame(
        {
            "client_id": client_id,
            "age": age,
            "sexe": sexe,
            "taille_cm": taille_cm,
            "poids_initial_kg": poids_initial_kg,
            "poids_cible_kg": poids_cible_kg,
            "objectif": objectif,
            "niveau": niveau,
            "frequence_entrainement_semaine": frequence_entrainement_semaine,
            "calories_quotidiennes": calories_quotidiennes,
            "proteines_g_par_jour": proteines_g_par_jour,
            "heures_sommeil": heures_sommeil,
            "semaines_suivi_prevues": semaines_suivi_prevues,
            "adherence_programme_pct": adherence_programme_pct,
        }
    )

    df = _add_target(df, besoin_calorique_estime, rng)
    return df


def _add_target(df: pd.DataFrame, besoin_calorique_estime: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Genere objectif_atteint avec une logique realiste (adherence, calories, frequence, sommeil)."""
    deficit = besoin_calorique_estime - df["calories_quotidiennes"].to_numpy()

    ideal_deficit = np.select(
        [df["objectif"] == "seche", df["objectif"] == "prise_masse", df["objectif"] == "recomposition"],
        [500.0, -380.0, 0.0],
    )
    tolerance = np.select(
        [df["objectif"] == "seche", df["objectif"] == "prise_masse", df["objectif"] == "recomposition"],
        [900.0, 900.0, 700.0],
    )
    alignement_calorique = 1 - np.clip(np.abs(deficit - ideal_deficit) / tolerance, 0, 1)

    adherence_norm = df["adherence_programme_pct"].to_numpy() / 100
    freq_norm = (df["frequence_entrainement_semaine"].to_numpy() - 3) / 3
    sommeil_norm = (df["heures_sommeil"].to_numpy() - 6.5) / 2.5
    suivi_norm = np.clip((df["semaines_suivi_prevues"].to_numpy() - 4) / 20, 0, 1)

    linear = (
        -2.5
        + 6.0 * (adherence_norm - 0.5)
        + 4.2 * (alignement_calorique - 0.5)
        + 1.6 * freq_norm
        + 0.9 * sommeil_norm
        + 1.0 * suivi_norm
        + rng.normal(0, 0.45, len(df))
    )
    proba = 1 / (1 + np.exp(-linear))
    df["objectif_atteint"] = rng.binomial(1, proba)
    return df


def generate_annuaire(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Genere la table d'identite (prenom, nom) associee aux client_id, separee du
    dataset ML. Utilise un generateur aleatoire independant (seed+1) afin de ne pas
    perturber la sequence de tirages de generate_raw_data (donnees numeriques et
    cible inchangees, pas de reentrainement necessaire)."""
    rng_identite = np.random.default_rng(seed + 1)
    n = len(df)

    prenoms = np.where(
        df["sexe"].to_numpy() == "H",
        rng_identite.choice(PRENOMS_H, size=n),
        rng_identite.choice(PRENOMS_F, size=n),
    )
    noms = rng_identite.choice(NOMS_FAMILLE, size=n)

    return pd.DataFrame(
        {
            "client_id": df["client_id"],
            "prenom": prenoms,
            "nom": noms,
        }
    )


def inspect_raw_data(df: pd.DataFrame) -> dict:
    """Inventaire des donnees brutes : types, valeurs manquantes, doublons."""
    inventory = {
        "n_lignes": len(df),
        "n_colonnes": df.shape[1],
        "types": df.dtypes.astype(str).to_dict(),
        "valeurs_manquantes": df.isna().sum().to_dict(),
        "doublons_client_id": int(df["client_id"].duplicated().sum()),
        "doublons_lignes": int(df.duplicated().sum()),
        "taux_cible": df["objectif_atteint"].value_counts(normalize=True).to_dict(),
    }
    return inventory


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features derivees : IMC, ratio proteines/poids, deficit calorique, score mode de vie."""
    df = df.copy()

    df["imc"] = (df["poids_initial_kg"] / (df["taille_cm"] / 100) ** 2).round(1)
    df["ratio_proteines_poids"] = (df["proteines_g_par_jour"] / df["poids_initial_kg"]).round(2)

    bmr = (
        10 * df["poids_initial_kg"]
        + 6.25 * df["taille_cm"]
        - 5 * df["age"]
        + np.where(df["sexe"] == "H", 5, -161)
    )
    df["besoin_calorique_estime"] = (bmr * _activity_factor(df["frequence_entrainement_semaine"].to_numpy())).round(0)
    df["deficit_calorique"] = (df["besoin_calorique_estime"] - df["calories_quotidiennes"]).round(0)

    sommeil_score = np.clip(df["heures_sommeil"] / 9, 0, 1)
    adherence_score = np.clip(df["adherence_programme_pct"] / 100, 0, 1)
    df["score_mode_de_vie"] = ((0.5 * sommeil_score + 0.5 * adherence_score) * 100).round(1)

    return df


def fit_transform_features(df: pd.DataFrame):
    """Encode les variables categorielles (LabelEncoder) et normalise le numerique (StandardScaler)."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CAT_COLS:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[NUM_COLS])
    for i, col in enumerate(NUM_COLS):
        df[f"{col}_scaled"] = scaled_values[:, i]

    return df, encoders, scaler


def transform_features(df: pd.DataFrame, encoders: dict, scaler: StandardScaler) -> pd.DataFrame:
    """Applique des encodeurs/scaler deja entraines a de nouvelles donnees (ex: predict.py)."""
    df = df.copy()
    for col in CAT_COLS:
        le = encoders[col]
        df[f"{col}_encoded"] = le.transform(df[col])

    scaled_values = scaler.transform(df[NUM_COLS])
    for i, col in enumerate(NUM_COLS):
        df[f"{col}_scaled"] = scaled_values[:, i]

    return df


def run_etl(n_clients: int = 600, seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extraction
    raw_df = generate_raw_data(n_clients=n_clients, seed=seed)
    inventory = inspect_raw_data(raw_df)
    raw_df.to_csv(RAW_PATH, index=False)

    annuaire_df = generate_annuaire(raw_df, seed=seed)
    annuaire_df.to_csv(ANNUAIRE_PATH, index=False)

    if verbose:
        print("=== Inventaire des donnees brutes ===")
        print(json.dumps(inventory, indent=2, ensure_ascii=False, default=str))

    # 2. Transformation
    engineered_df = feature_engineering(raw_df)
    processed_df, encoders, scaler = fit_transform_features(engineered_df)

    # 3. Chargement
    processed_df.to_csv(PROCESSED_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    if verbose:
        print(f"\nDataset brut       -> {RAW_PATH}")
        print(f"Annuaire (identite) -> {ANNUAIRE_PATH}  (jamais fusionne au dataset ML)")
        print(f"Dataset final       -> {PROCESSED_PATH}")
        print(f"Scaler              -> {SCALER_PATH}")
        print(f"Encoders            -> {ENCODERS_PATH}")
        print(f"\nColonnes de features utilisees pour le ML ({len(FEATURE_COLUMNS)}):")
        print(FEATURE_COLUMNS)

    return processed_df


if __name__ == "__main__":
    run_etl()
