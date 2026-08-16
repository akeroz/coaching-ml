"""Tests unitaires de l'export PDF (src/report.py)."""

import pandas as pd

from report import build_client_pdf

CLIENT = pd.Series({
    "prenom": "Test", "nom": "Unitaire", "objectif": "seche", "niveau": "intermediaire",
    "poids_initial_kg": 90.0, "poids_cible_kg": 80.0,
})


def test_build_client_pdf_returns_valid_pdf_bytes():
    weigh_ins = pd.DataFrame({
        "date_saisie": ["2026-01-01", "2026-01-08"],
        "poids": [90.0, 88.5],
        "note": ["Poids de depart", ""],
    })
    prediction = {"proba": 0.73, "interpretation": "Profil favorable"}

    pdf_bytes = build_client_pdf(CLIENT, weigh_ins, prediction)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_build_client_pdf_without_weigh_ins_or_prediction():
    """Ne doit pas planter si le client n'a pas encore de pesee ni de prediction calculee."""
    pdf_bytes = build_client_pdf(CLIENT, pd.DataFrame(columns=["date_saisie", "poids", "note"]), None)
    assert pdf_bytes.startswith(b"%PDF")
