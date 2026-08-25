"""Tests d'integration de la couche de persistance (src/db.py).

Ces tests s'executent contre la vraie base configuree via DATABASE_URL (ex.
Supabase) : ils sont automatiquement ignores si DATABASE_URL n'est pas
definie (ex. checkout frais sans .env), pour ne jamais faire echouer la
suite faute de credentials. Un client de test dedie est cree puis
systematiquement supprime en fin de test (teardown), sans jamais toucher
aux vrais clients ou aux profils de demonstration existants."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="DATABASE_URL non configuree - tests d'integration base de donnees ignores",
)

import db  # noqa: E402

TEST_PROFILE = {
    "prenom": "TestUnitaire", "nom": "Pytest", "age": 30, "sexe": "H", "taille_cm": 180.0,
    "poids_initial_kg": 80.0, "poids_cible_kg": 75.0, "objectif": "seche",
    "niveau": "intermediaire", "frequence_entrainement_semaine": 4,
    "calories_quotidiennes": 2200.0, "proteines_g_par_jour": 160.0, "heures_sommeil": 7.0,
    "semaines_suivi_prevues": 12, "adherence_programme_pct": 80.0,
}


@pytest.fixture
def test_client_id():
    """Cree un client de test et garantit sa suppression, meme si le test echoue."""
    client_id = db.add_client(dict(TEST_PROFILE))
    yield client_id
    db.delete_client(client_id)


def test_add_client_creates_retrievable_record(test_client_id):
    client = db.get_client(test_client_id)
    assert client is not None
    assert client["prenom"] == "TestUnitaire"
    assert client["nom"] == "Pytest"
    assert client["actif"] == 1
    assert pd_isna(client["objectif_atteint"])


def test_add_client_creates_initial_weigh_in(test_client_id):
    weigh_ins = db.get_weigh_ins(test_client_id)
    assert len(weigh_ins) == 1
    assert weigh_ins.iloc[0]["poids"] == TEST_PROFILE["poids_initial_kg"]


def test_add_weigh_in_appends_history(test_client_id):
    db.add_weigh_in(test_client_id, 78.5, note="Semaine 2")
    weigh_ins = db.get_weigh_ins(test_client_id)
    assert len(weigh_ins) == 2
    assert weigh_ins.iloc[-1]["poids"] == 78.5


def test_update_client_modifies_fields(test_client_id):
    updated = dict(TEST_PROFILE)
    updated["poids_cible_kg"] = 73.0
    updated["adherence_programme_pct"] = 95.0
    db.update_client(test_client_id, updated)

    client = db.get_client(test_client_id)
    assert client["poids_cible_kg"] == 73.0
    assert client["adherence_programme_pct"] == 95.0


def test_update_client_status_closes_tracking(test_client_id):
    db.update_client_status(test_client_id, actif=False, objectif_atteint=1)
    client = db.get_client(test_client_id)
    assert client["actif"] == 0
    assert client["objectif_atteint"] == 1


def test_get_labelled_clients_includes_closed_client(test_client_id):
    db.update_client_status(test_client_id, actif=False, objectif_atteint=0)
    labelled = db.get_labelled_clients()
    assert test_client_id in labelled["client_id"].values


def test_delete_client_removes_record_and_weigh_ins():
    client_id = db.add_client(dict(TEST_PROFILE))
    db.add_weigh_in(client_id, 79.0)

    db.delete_client(client_id)

    assert db.get_client(client_id) is None
    assert db.get_weigh_ins(client_id).empty


def test_add_client_without_consent_is_not_recorded_as_consenting():
    client_id = db.add_client(dict(TEST_PROFILE))
    try:
        client = db.get_client(client_id)
        assert client["consentement_recueilli"] == 0
        assert pd_isna(client["date_consentement"])
    finally:
        db.delete_client(client_id)


def test_add_client_with_consent_is_timestamped():
    client_id = db.add_client(dict(TEST_PROFILE), consentement_recueilli=True)
    try:
        client = db.get_client(client_id)
        assert client["consentement_recueilli"] == 1
        assert not pd_isna(client["date_consentement"])
    finally:
        db.delete_client(client_id)


def test_add_weigh_in_stores_energie_and_tour_taille(test_client_id):
    db.add_weigh_in(test_client_id, 78.0, energie=4, tour_taille_cm=85.5)
    weigh_ins = db.get_weigh_ins(test_client_id)
    last_row = weigh_ins.iloc[-1]
    assert last_row["energie"] == 4
    assert last_row["tour_taille_cm"] == 85.5


def test_add_weigh_in_optional_fields_default_to_null(test_client_id):
    db.add_weigh_in(test_client_id, 77.0)
    last_row = db.get_weigh_ins(test_client_id).iloc[-1]
    assert pd_isna(last_row["energie"])
    assert pd_isna(last_row["tour_taille_cm"])


def test_export_all_data_includes_test_client(test_client_id):
    export = db.export_all_data()
    assert test_client_id in export["clients"]["client_id"].values
    assert test_client_id in export["suivis_hebdo"]["client_id"].values


def pd_isna(value) -> bool:
    import pandas as pd
    return pd.isna(value)
