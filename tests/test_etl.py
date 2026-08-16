"""Tests unitaires du pipeline ETL (src/etl.py)."""

import numpy as np
import pandas as pd
import pytest

from etl import (
    CAT_COLS, FEATURE_COLUMNS, NUM_COLS, TARGET_COL,
    feature_engineering, fit_transform_features, generate_raw_data, transform_features,
)


@pytest.fixture(scope="module")
def raw_df():
    return generate_raw_data(n_clients=120, seed=42)


def test_generate_raw_data_shape_and_types(raw_df):
    assert len(raw_df) == 120
    assert raw_df["client_id"].is_unique
    assert set(raw_df["sexe"].unique()) <= {"H", "F"}
    assert set(raw_df["objectif"].unique()) <= {"seche", "prise_masse", "recomposition"}
    assert set(raw_df["niveau"].unique()) <= {"debutant", "intermediaire", "avance"}
    assert set(raw_df[TARGET_COL].unique()) <= {0, 1}


def test_generate_raw_data_no_missing_values(raw_df):
    assert raw_df.isna().sum().sum() == 0


def test_generate_raw_data_plausible_ranges(raw_df):
    assert raw_df["age"].between(18, 65).all()
    assert raw_df["poids_initial_kg"].between(40, 170).all()
    assert raw_df["adherence_programme_pct"].between(10, 100).all()
    assert raw_df["frequence_entrainement_semaine"].between(1, 6).all()


def test_generate_raw_data_is_deterministic():
    df1 = generate_raw_data(n_clients=50, seed=7)
    df2 = generate_raw_data(n_clients=50, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_raw_data_different_seeds_differ():
    df1 = generate_raw_data(n_clients=50, seed=1)
    df2 = generate_raw_data(n_clients=50, seed=2)
    assert not df1["poids_initial_kg"].equals(df2["poids_initial_kg"])


def test_feature_engineering_imc_formula():
    df = pd.DataFrame({
        "poids_initial_kg": [80.0], "taille_cm": [200.0], "age": [30],
        "sexe": ["H"], "frequence_entrainement_semaine": [4],
        "calories_quotidiennes": [2500.0], "proteines_g_par_jour": [160.0],
        "heures_sommeil": [7.0], "adherence_programme_pct": [80.0],
    })
    out = feature_engineering(df)
    # 80 / (2.0 ** 2) = 20.0
    assert out["imc"].iloc[0] == pytest.approx(20.0)


def test_feature_engineering_ratio_proteines_poids():
    df = pd.DataFrame({
        "poids_initial_kg": [100.0], "taille_cm": [180.0], "age": [25],
        "sexe": ["F"], "frequence_entrainement_semaine": [3],
        "calories_quotidiennes": [2000.0], "proteines_g_par_jour": [150.0],
        "heures_sommeil": [7.0], "adherence_programme_pct": [70.0],
    })
    out = feature_engineering(df)
    assert out["ratio_proteines_poids"].iloc[0] == pytest.approx(1.5)


def test_feature_engineering_score_mode_de_vie_bounds(raw_df):
    out = feature_engineering(raw_df)
    assert out["score_mode_de_vie"].between(0, 100).all()


def test_feature_engineering_deficit_calorique_sign():
    """Un client qui mange beaucoup moins que son besoin doit avoir un deficit positif."""
    df = pd.DataFrame({
        "poids_initial_kg": [90.0], "taille_cm": [180.0], "age": [30],
        "sexe": ["H"], "frequence_entrainement_semaine": [3],
        "calories_quotidiennes": [1200.0],  # tres bas par rapport au besoin estime
        "proteines_g_par_jour": [150.0], "heures_sommeil": [7.0],
        "adherence_programme_pct": [80.0],
    })
    out = feature_engineering(df)
    assert out["deficit_calorique"].iloc[0] > 0


def test_fit_transform_features_adds_expected_columns(raw_df):
    engineered = feature_engineering(raw_df)
    processed, encoders, scaler = fit_transform_features(engineered)
    for col in FEATURE_COLUMNS:
        assert col in processed.columns
    assert set(encoders.keys()) == set(CAT_COLS)


def test_fit_transform_features_scaled_columns_are_standardized(raw_df):
    engineered = feature_engineering(raw_df)
    processed, _, _ = fit_transform_features(engineered)
    for col in NUM_COLS:
        scaled = processed[f"{col}_scaled"]
        assert scaled.mean() == pytest.approx(0, abs=1e-8)
        assert scaled.std(ddof=0) == pytest.approx(1, abs=1e-6)


def test_transform_features_matches_fit_on_same_data(raw_df):
    """transform_features() applique a des donnees deja vues doit reproduire
    exactement les valeurs produites par fit_transform_features()."""
    engineered = feature_engineering(raw_df)
    processed, encoders, scaler = fit_transform_features(engineered)

    one_row = engineered.iloc[[0]].copy()
    replayed = transform_features(one_row, encoders, scaler)

    for col in FEATURE_COLUMNS:
        assert replayed[col].iloc[0] == pytest.approx(processed[col].iloc[0])


def test_transform_features_unknown_category_raises(raw_df):
    engineered = feature_engineering(raw_df)
    _, encoders, scaler = fit_transform_features(engineered)

    bad_row = engineered.iloc[[0]].copy()
    bad_row["objectif"] = "categorie_inexistante"
    with pytest.raises(ValueError):
        transform_features(bad_row, encoders, scaler)
