"""Tests unitaires de la logique metier (src/coaching_logic.py)."""

import pandas as pd
import pytest

from coaching_logic import (
    build_profile_from_client_row, compute_progress_status,
    local_feature_importance, predict_for_client, progress_pct,
)
from etl import FEATURE_COLUMNS
from predict import build_feature_row, load_artifacts

CLIENT_SECHE = pd.Series({
    "age": 25, "sexe": "H", "taille_cm": 180.0,
    "poids_initial_kg": 90.0, "poids_cible_kg": 80.0,
    "objectif": "seche", "niveau": "intermediaire",
    "frequence_entrainement_semaine": 4, "calories_quotidiennes": 2200.0,
    "proteines_g_par_jour": 160.0, "heures_sommeil": 7.0,
    "semaines_suivi_prevues": 12, "adherence_programme_pct": 80.0,
})


def test_compute_progress_status_progression():
    # poids initial 90, cible 80 -> se rapprocher (ex: 85) doit etre une progression
    status = compute_progress_status(CLIENT_SECHE, poids_actuel=85.0)
    assert status["libelle"] == "En progression"


def test_compute_progress_status_regression():
    # s'eloigner de la cible (poids qui monte alors que l'objectif est de perdre)
    status = compute_progress_status(CLIENT_SECHE, poids_actuel=93.0)
    assert status["libelle"] == "En regression"


def test_compute_progress_status_stable():
    status = compute_progress_status(CLIENT_SECHE, poids_actuel=90.05)
    assert status["libelle"] == "Stable"


def test_compute_progress_status_works_for_prise_masse():
    """Meme logique doit fonctionner sans configuration specifique pour la prise de masse."""
    client = pd.Series({**CLIENT_SECHE, "poids_initial_kg": 65.0, "poids_cible_kg": 75.0, "objectif": "prise_masse"})
    assert compute_progress_status(client, poids_actuel=70.0)["libelle"] == "En progression"
    assert compute_progress_status(client, poids_actuel=60.0)["libelle"] == "En regression"


def test_progress_pct_halfway():
    # 90 -> 80 (delta -10), poids actuel 85 = mi-chemin = 50%
    assert progress_pct(CLIENT_SECHE, poids_actuel=85.0) == pytest.approx(50.0)


def test_progress_pct_at_goal():
    assert progress_pct(CLIENT_SECHE, poids_actuel=80.0) == pytest.approx(100.0)


def test_progress_pct_no_target_change():
    client = pd.Series({**CLIENT_SECHE, "poids_cible_kg": CLIENT_SECHE["poids_initial_kg"]})
    assert progress_pct(client, poids_actuel=999.0) == pytest.approx(100.0)


def test_build_profile_from_client_row_maps_all_fields():
    profile = build_profile_from_client_row(CLIENT_SECHE)
    assert profile["age"] == 25
    assert isinstance(profile["age"], int)
    assert profile["objectif"] == "seche"
    assert profile["poids_initial_kg"] == 90.0
    assert set(profile.keys()) == {
        "age", "sexe", "taille_cm", "poids_initial_kg", "poids_cible_kg",
        "objectif", "niveau", "frequence_entrainement_semaine", "calories_quotidiennes",
        "proteines_g_par_jour", "heures_sommeil", "semaines_suivi_prevues",
        "adherence_programme_pct",
    }


@pytest.fixture(scope="module")
def model_artifacts():
    return load_artifacts()


def test_predict_for_client_returns_valid_probability(model_artifacts):
    model, scaler, encoders = model_artifacts
    result = predict_for_client(CLIENT_SECHE, model, scaler, encoders)
    assert 0.0 <= result["proba"] <= 1.0
    assert result["interpretation"] in {
        "Profil favorable", "Profil a risque, ajuster le programme", "Profil critique, revoir les bases",
    }


def test_predict_for_client_interpretation_matches_thresholds(model_artifacts):
    model, scaler, encoders = model_artifacts
    result = predict_for_client(CLIENT_SECHE, model, scaler, encoders)
    proba = result["proba"]
    if proba > 0.70:
        assert result["interpretation"] == "Profil favorable"
    elif proba >= 0.40:
        assert result["interpretation"] == "Profil a risque, ajuster le programme"
    else:
        assert result["interpretation"] == "Profil critique, revoir les bases"


def test_local_feature_importance_covers_all_features(model_artifacts):
    model, scaler, encoders = model_artifacts
    profile = build_profile_from_client_row(CLIENT_SECHE)
    X_row = build_feature_row(profile, encoders, scaler)
    importance = local_feature_importance(model, X_row)
    assert set(importance.index) == set(FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
