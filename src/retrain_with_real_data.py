"""Reentrainement periodique du modele en integrant les vrais clients labellises
(objectif_atteint connu), avec une garde anti-regression : les artefacts de
production (best_model.pkl, scaler, encoders, dataset_final.csv) ne sont ecrases
que si le nouveau score composite depasse l'ancien d'au moins une marge de
promotion minimale (voir _get_previous_winner_stats/run_retrain) - un simple
">=" aurait pu promouvoir un modele sur une amelioration nulle ou infime,
indiscernable du bruit statistique normal de mesure. Cette marge n'est pas une
constante inventee : elle est derivee de l'ecart-type de validation croisee
(cv_std) deja mesure pour le modele en production, pondere par le poids de
l'AUC-ROC dans le score composite - voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md.

Note methodologique (absence de fuite de donnees) : le scaler/encoders utilises
pour evaluer les 4 modeles sont ajustes uniquement sur un split d'entrainement
(80%), jamais sur l'integralite des donnees combinees - voir train.py. Le modele
promu est ensuite reentraine avec les memes hyperparametres sur 100% des donnees
combinees (synthetiques + reelles), pratique standard une fois la methodologie
validee, pas une fuite.

Chaque tentative (promue, rejetee ou ignoree faute de volume) est journalisee
dans models/retrain_history.json, pour permettre de detecter une derive du
modele dans le temps (Bloc 3 - maintenabilite).

Usage : depuis l'app ("Mes clients" > Reentrainement) ou en ligne de commande
(python src/retrain_with_real_data.py)."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import db
from etl import (
    ENCODERS_PATH, FEATURE_COLUMNS,
    PROCESSED_PATH, RAW_PATH, SCALER_PATH, TARGET_COL,
    feature_engineering, fit_transform_features, transform_features,
)
from select_model import RESULTS_PATH as PROD_RESULTS_PATH
from select_model import WEIGHTS, compute_composite_scores, generate_report
from train import MODEL_SPECS, MODELS_DIR, RANDOM_STATE, run_training

ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = MODELS_DIR / "candidate"
CANDIDATE_RESULTS_PATH = MODELS_DIR / "retrain_candidate_results.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RETRAIN_HISTORY_PATH = MODELS_DIR / "retrain_history.json"
DOCS_DIR = ROOT_DIR / "docs"
SELECTION_REPORT_PATH = DOCS_DIR / "MODEL_SELECTION_REPORT.md"

RAW_FEATURE_COLS = [
    "client_id", "age", "sexe", "taille_cm", "poids_initial_kg", "poids_cible_kg",
    "objectif", "niveau", "frequence_entrainement_semaine", "calories_quotidiennes",
    "proteines_g_par_jour", "heures_sommeil", "semaines_suivi_prevues",
    "adherence_programme_pct", "objectif_atteint",
]

# Seuil minimal usuel en apprentissage automatique pour qu'une re-estimation sur un
# sous-groupe reste statistiquement significative plutot que du bruit (regle
# empirique communement citee : au moins 5 a 10 observations par classe avant de
# tirer une conclusion) - voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md.
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


def _get_previous_winner_stats() -> tuple[float, float]:
    """Retourne (composite_score, cv_std) du modele actuellement en production.
    cv_std est l'ecart-type de l'AUC-ROC sur la validation croisee 5-fold - une
    mesure deja calculee du bruit statistique normal de ce modele, utilisee comme
    marge de promotion (voir run_retrain ci-dessous)."""
    if not PROD_RESULTS_PATH.exists():
        return -1.0, 0.0
    with open(PROD_RESULTS_PATH, "r", encoding="utf-8") as f:
        prod_results = json.load(f)
    winner_key = prod_results["winner"]
    for row in prod_results["ranking"]:
        if row["key"] == winner_key:
            return row["composite_score"], row["cv_std"]
    return -1.0, 0.0


def get_retrain_history() -> list[dict]:
    if not RETRAIN_HISTORY_PATH.exists():
        return []
    with open(RETRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _log_retrain_attempt(summary: dict):
    """Ajoute une entree a l'historique des tentatives de reentrainement, quel que
    soit le resultat (promu/rejete/ignore) - permet de suivre l'evolution du score
    du modele dans le temps et de reperer une eventuelle derive."""
    history = get_retrain_history()
    history.append({
        "date": date.today().isoformat(),
        "status": summary.get("status"),
        "n_real_clients": summary.get("n_real_clients"),
        "old_score": summary.get("old_score"),
        "new_score": summary.get("new_score"),
        "promotion_margin": summary.get("promotion_margin"),
        "winner_label": summary.get("new_winner_label"),
    })
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def run_retrain(min_real_clients: int = MIN_REAL_CLIENTS) -> dict:
    combined_raw_df, n_real = build_combined_raw_df()

    if n_real < min_real_clients:
        result = {
            "status": "skipped",
            "reason": f"Seulement {n_real} client(s) reel(s) labellise(s) "
                      f"(objectif_atteint connu) - minimum requis : {min_real_clients}. "
                      "Reentrainement trop instable sur un si petit volume, annule.",
            "n_real_clients": n_real,
        }
        _log_retrain_attempt(result)
        return result

    engineered_df = feature_engineering(combined_raw_df)

    # Split avant normalisation : le test set n'influence jamais le scaler/encoders
    # utilises pour evaluer les modeles (voir note methodologique en tete de fichier).
    train_raw, test_raw = train_test_split(
        engineered_df, test_size=0.2, stratify=engineered_df[TARGET_COL], random_state=RANDOM_STATE
    )
    train_processed, eval_encoders, eval_scaler = fit_transform_features(train_raw)
    test_processed = transform_features(test_raw, eval_encoders, eval_scaler)

    X_train, y_train = train_processed[FEATURE_COLUMNS], train_processed[TARGET_COL]
    X_test, y_test = test_processed[FEATURE_COLUMNS], test_processed[TARGET_COL]

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    candidate_train_results = run_training(
        X_train, X_test, y_train, y_test,
        models_dir=CANDIDATE_DIR, results_path=CANDIDATE_RESULTS_PATH,
    )

    ranking = compute_composite_scores(candidate_train_results)
    new_winner_key = ranking.iloc[0]["key"]
    new_score = float(ranking.iloc[0]["composite_score"])
    old_score, old_cv_std = _get_previous_winner_stats()

    # Marge de promotion : exiger une amelioration strictement superieure a 0 ne
    # suffit pas a distinguer un vrai gain d'une simple fluctuation de mesure. La
    # marge minimale exigee est derivee de l'ecart-type de validation croisee
    # (cv_std) deja mesure pour le modele actuellement en production, pondere par
    # le poids de l'AUC-ROC dans le score composite (WEIGHTS["auc"] = 0.4) - pas
    # une constante inventee, mais le bruit statistique reellement observe sur ce
    # modele, ramene a l'echelle du score composite. Voir
    # docs/JUSTIFICATIONS_METHODOLOGIQUES.md.
    promotion_margin = WEIGHTS["auc"] * old_cv_std
    promoted = new_score >= old_score + promotion_margin

    summary = {
        "status": "promoted" if promoted else "rejected",
        "n_real_clients": n_real,
        "n_total_clients": len(combined_raw_df),
        "old_score": old_score,
        "promotion_margin": promotion_margin,
        "new_score": new_score,
        "new_winner_label": ranking.iloc[0]["label"],
        "ranking": ranking.to_dict(orient="records"),
    }

    if promoted:
        # Reentrainement final avec les memes hyperparametres sur l'integralite des
        # donnees combinees (synthetiques + reelles), une fois la methodologie
        # validee sur le split ci-dessus - maximise les donnees pour le modele
        # deploye, sans jamais fuiter le test set dans une metrique rapportee.
        full_processed_df, full_encoders, full_scaler = fit_transform_features(engineered_df)
        X_full = full_processed_df[FEATURE_COLUMNS]
        y_full = full_processed_df[TARGET_COL]

        best_params = candidate_train_results[new_winner_key]["best_params"]
        final_model = clone(MODEL_SPECS[new_winner_key]["estimator"]).set_params(**best_params)
        final_model.fit(X_full, y_full)
        joblib.dump(final_model, BEST_MODEL_PATH)

        for key in candidate_train_results:
            shutil.copyfile(CANDIDATE_DIR / f"{key}.pkl", MODELS_DIR / f"{key}.pkl")

        full_processed_df.to_csv(PROCESSED_PATH, index=False)
        joblib.dump(full_scaler, SCALER_PATH)
        joblib.dump(full_encoders, ENCODERS_PATH)

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
            f"Score composite precedent : {old_score:.4f} -> nouveau : {new_score:.4f} "
            f"(marge de promotion exigee : {promotion_margin:.4f}, derivee du bruit de "
            f"validation croisee du modele precedent)."
        )
        SELECTION_REPORT_PATH.write_text(report, encoding="utf-8")
    else:
        with open(CANDIDATE_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"ranking": ranking.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)

    _log_retrain_attempt(summary)
    return summary


if __name__ == "__main__":
    result = run_retrain()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
