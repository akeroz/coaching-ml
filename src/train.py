"""
Entrainement et evaluation de 4 modeles de classification pour predire
l'atteinte d'objectif des clients (coaching fitness @builtbyarthur).

Modeles : Regression logistique, Foret aleatoire, XGBoost, Reseau de neurones (MLP).
Chaque modele est optimise par GridSearchCV puis evalue sur un jeu de test
(20%, split stratifie), avec validation croisee 5-fold et courbes d'apprentissage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from etl import FEATURE_COLUMNS, RAW_PATH, TARGET_COL, feature_engineering, fit_transform_features, transform_features

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_PATH = MODELS_DIR / "train_results.json"

RANDOM_STATE = 42
CV_FOLDS = 5


def load_data():
    """Charge les donnees et les prepare SANS fuite : le split train/test est fait
    sur les donnees brutes (feature engineering pur, sans normalisation), puis le
    StandardScaler/LabelEncoder est ajuste uniquement sur le train set. Le test set
    n'influence jamais la normalisation - contrairement a une version precedente qui
    relisait dataset_final.csv (deja normalise sur 100% des donnees avant le split).

    Le scaler/encoders "production" utilises par predict.py (data/processed/scaler.pkl,
    label_encoders.pkl, generes par etl.py sur l'integralite du dataset) restent
    volontairement distincts : une fois la methodologie validee par ce split honnete,
    le modele final deploye beneficie d'etre entraine sur toutes les donnees
    disponibles - pratique standard, pas une fuite, puisqu'aucune metrique n'est
    rapportee a partir de ce refit final."""
    raw_df = pd.read_csv(RAW_PATH)
    engineered_df = feature_engineering(raw_df)

    train_raw, test_raw = train_test_split(
        engineered_df, test_size=0.2, stratify=engineered_df[TARGET_COL], random_state=RANDOM_STATE
    )
    train_processed, encoders, scaler = fit_transform_features(train_raw)
    test_processed = transform_features(test_raw, encoders, scaler)

    X_train, y_train = train_processed[FEATURE_COLUMNS], train_processed[TARGET_COL]
    X_test, y_test = test_processed[FEATURE_COLUMNS], test_processed[TARGET_COL]
    return X_train, X_test, y_train, y_test


MODEL_SPECS = {
    "logistic_regression": {
        "label": "Regression logistique",
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
    },
    "random_forest": {
        "label": "Foret aleatoire",
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
        },
    },
    "xgboost": {
        "label": "XGBoost",
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    },
    "mlp": {
        "label": "Reseau de neurones (MLP)",
        "estimator": MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
        },
    },
}


def train_one_model(key: str, spec: dict, X_train, X_test, y_train, y_test, models_dir: Path = MODELS_DIR) -> dict:
    print(f"\n=== Entrainement : {spec['label']} ===")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    start = time.perf_counter()
    grid = GridSearchCV(
        spec["estimator"], spec["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    training_time_sec = time.perf_counter() - start

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="roc_auc")

    train_sizes, train_scores, val_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring="roc_auc",
        n_jobs=-1,
    )

    results = {
        "key": key,
        "label": spec["label"],
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "training_time_sec": training_time_sec,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "y_test": y_test.tolist(),
        "y_proba": y_proba.tolist(),
        "learning_curve": {
            "train_sizes": train_sizes.tolist(),
            "train_scores_mean": train_scores.mean(axis=1).tolist(),
            "val_scores_mean": val_scores.mean(axis=1).tolist(),
        },
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / f"{key}.pkl")

    print(f"  Meilleurs parametres : {grid.best_params_}")
    print(f"  Accuracy={results['accuracy']:.3f}  F1={results['f1_weighted']:.3f}  "
          f"AUC={results['auc_roc']:.3f}  Temps={training_time_sec:.1f}s")

    return results


def run_training(X_train=None, X_test=None, y_train=None, y_test=None,
                  models_dir: Path = MODELS_DIR, results_path: Path = RESULTS_PATH) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    if X_train is None:
        X_train, X_test, y_train, y_test = load_data()

    all_results = {}
    for key, spec in MODEL_SPECS.items():
        all_results[key] = train_one_model(key, spec, X_train, X_test, y_train, y_test, models_dir=models_dir)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResultats d'entrainement sauvegardes -> {results_path}")
    return all_results


if __name__ == "__main__":
    run_training()
