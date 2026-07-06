"""Reentrainement periodique du modele en integrant les vrais clients labellises
(objectif_atteint connu), avec une garde anti-regression : les artefacts de
production (best_model.pkl, scaler, encoders, dataset_final.csv) ne sont ecrases
que si le nouveau score composite est au moins aussi bon que l'ancien.

Usage : depuis l'app ("Mes clients" > Reentrainement) ou en ligne de commande
(python src/retrain_with_real_data.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import db
from etl import (
    ENCODERS_PATH, FEATURE_COLUMNS,
    PROCESSED_PATH, RAW_PATH, SCALER_PATH, TARGET_COL,
    feature_engineering, fit_transform_features,
)
from select_model import RESULTS_PATH as PROD_RESULTS_PATH
from select_model import WEIGHTS, compute_composite_scores, generate_report
from train import MODELS_DIR, RANDOM_STATE, run_training

ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = MODELS_DIR / "candidate"
CANDIDATE_RESULTS_PATH = MODELS_DIR / "retrain_candidate_results.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
DOCS_DIR = ROOT_DIR / "docs"
SELECTION_REPORT_PATH = DOCS_DIR / "MODEL_SELECTION_REPORT.md"

RAW_FEATURE_COLS = [
    "client_id", "age", "sexe", "taille_cm", "poids_initial_kg", "poids_cible_kg",
    "objectif", "niveau", "frequence_entrainement_semaine", "calories_quotidiennes",
    "proteines_g_par_jour", "heures_sommeil", "semaines_suivi_prevues",
    "adherence_programme_pct", "objectif_atteint",
]

MIN_REAL_CLIENTS = 5


def build_combined_raw_df() -> tuple[pd.DataFrame, int]:
    synthetic_df = pd.read_csv(RAW_PATH)[RAW_FEATURE_COLS]
    real_df = db.get_labelled_clients()

    if real_df.empty:
        return synthetic_df, 0

    real_df = real_df[RAW_FEATURE_COLS].copy()
    real_df["objectif_atteint"] = real_df["objectif_atteint"].astype(int)
    combined = pd.concat([synthetic_df, real_df], ignore_index=True)
    return combined, len(real_df)


def _get_previous_winner_score() -> float:
    if not PROD_RESULTS_PATH.exists():
        return -1.0
    with open(PROD_RESULTS_PATH, "r", encoding="utf-8") as f:
        prod_results = json.load(f)
    winner_key = prod_results["winner"]
    for row in prod_results["ranking"]:
        if row["key"] == winner_key:
            return row["composite_score"]
    return -1.0


def run_retrain(min_real_clients: int = MIN_REAL_CLIENTS) -> dict:
    combined_raw_df, n_real = build_combined_raw_df()

    if n_real < min_real_clients:
        return {
            "status": "skipped",
            "reason": f"Seulement {n_real} client(s) reel(s) labellise(s) "
                      f"(objectif_atteint connu) - minimum requis : {min_real_clients}. "
                      "Reentrainement trop instable sur un si petit volume, annule.",
            "n_real_clients": n_real,
        }

    engineered_df = feature_engineering(combined_raw_df)
    processed_df, encoders, scaler = fit_transform_features(engineered_df)

    X = processed_df[FEATURE_COLUMNS]
    y = processed_df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    candidate_train_results = run_training(
        X_train, X_test, y_train, y_test,
        models_dir=CANDIDATE_DIR, results_path=CANDIDATE_RESULTS_PATH,
    )

    ranking = compute_composite_scores(candidate_train_results)
    new_winner_key = ranking.iloc[0]["key"]
    new_score = float(ranking.iloc[0]["composite_score"])
    old_score = _get_previous_winner_score()

    promoted = new_score >= old_score

    summary = {
        "status": "promoted" if promoted else "rejected",
        "n_real_clients": n_real,
        "n_total_clients": len(combined_raw_df),
        "old_score": old_score,
        "new_score": new_score,
        "new_winner_label": ranking.iloc[0]["label"],
        "ranking": ranking.to_dict(orient="records"),
    }

    if promoted:
        shutil.copyfile(CANDIDATE_DIR / f"{new_winner_key}.pkl", BEST_MODEL_PATH)
        for key in candidate_train_results:
            shutil.copyfile(CANDIDATE_DIR / f"{key}.pkl", MODELS_DIR / f"{key}.pkl")

        processed_df.to_csv(PROCESSED_PATH, index=False)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoders, ENCODERS_PATH)

        output = {
            "ranking": ranking.to_dict(orient="records"),
            "winner": new_winner_key,
            "weights": WEIGHTS,
            "full_results": candidate_train_results,
        }
        with open(PROD_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        report = generate_report(ranking, new_winner_key, candidate_train_results)
        report += (
            f"\n\n## Reentrainement avec donnees reelles\n\n"
            f"Ce modele integre {n_real} client(s) reel(s) labellise(s) en plus du "
            f"dataset synthetique ({len(combined_raw_df)} clients au total). "
            f"Score composite precedent : {old_score:.4f} -> nouveau : {new_score:.4f}."
        )
        SELECTION_REPORT_PATH.write_text(report, encoding="utf-8")
    else:
        with open(CANDIDATE_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"ranking": ranking.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    result = run_retrain()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
