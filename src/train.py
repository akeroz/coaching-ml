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

from etl import FEATURE_COLUMNS, PROCESSED_PATH, TARGET_COL

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_PATH = MODELS_DIR / "train_results.json"

RANDOM_STATE = 42
CV_FOLDS = 5


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


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


def train_one_model(key: str, spec: dict, X_train, X_test, y_train, y_test) -> dict:
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

    joblib.dump(best_model, MODELS_DIR / f"{key}.pkl")

    print(f"  Meilleurs parametres : {grid.best_params_}")
    print(f"  Accuracy={results['accuracy']:.3f}  F1={results['f1_weighted']:.3f}  "
          f"AUC={results['auc_roc']:.3f}  Temps={training_time_sec:.1f}s")

    return results


def run_training() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    all_results = {}
    for key, spec in MODEL_SPECS.items():
        all_results[key] = train_one_model(key, spec, X_train, X_test, y_train, y_test)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResultats d'entrainement sauvegardes -> {RESULTS_PATH}")
    return all_results


if __name__ == "__main__":
    run_training()
