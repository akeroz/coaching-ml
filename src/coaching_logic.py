"""Logique metier pure (sans dependance a Streamlit), extraite de app.py pour
etre testable independamment du runtime de l'application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predict import build_feature_row


def local_feature_importance(model, X_row: pd.DataFrame) -> pd.Series:
    """Contribution de chaque feature a la prediction (approche generique multi-modeles)."""
    if hasattr(model, "coef_"):
        contrib = model.coef_[0] * X_row.iloc[0].to_numpy()
        return pd.Series(contrib, index=X_row.columns).sort_values(key=abs, ascending=False)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=X_row.columns).sort_values(ascending=False)
    return pd.Series(np.zeros(len(X_row.columns)), index=X_row.columns)


def build_profile_from_client_row(client: pd.Series) -> dict:
    return {
        "age": int(client["age"]), "sexe": client["sexe"], "taille_cm": client["taille_cm"],
        "poids_initial_kg": client["poids_initial_kg"], "poids_cible_kg": client["poids_cible_kg"],
        "objectif": client["objectif"], "niveau": client["niveau"],
        "frequence_entrainement_semaine": int(client["frequence_entrainement_semaine"]),
        "calories_quotidiennes": client["calories_quotidiennes"],
        "proteines_g_par_jour": client["proteines_g_par_jour"],
        "heures_sommeil": client["heures_sommeil"],
        "semaines_suivi_prevues": int(client["semaines_suivi_prevues"]),
        "adherence_programme_pct": client["adherence_programme_pct"],
    }


PROBA_SEUIL_FAVORABLE = 0.70
PROBA_SEUIL_RISQUE = 0.40


def predict_for_client(client: pd.Series, model, scaler, encoders) -> dict:
    """Seuils de decision bases sur la convention RAG (Red-Amber-Green), un standard
    largement utilise dans les tableaux de bord de risque et l'aide a la decision :
    un seuil de decision naturel a 0.5 (defaut de la classification binaire) est
    entoure d'une zone tampon symetrique (+/- 0.10-0.15) ou le jugement humain du
    coach doit completer la prediction, plutot qu'un seuil unique tranchant. Voir
    docs/JUSTIFICATIONS_METHODOLOGIQUES.md pour le detail."""
    profile = build_profile_from_client_row(client)
    X_row = build_feature_row(profile, encoders, scaler)
    proba = float(model.predict_proba(X_row)[0, 1])
    interpretation = (
        "Profil favorable" if proba > PROBA_SEUIL_FAVORABLE
        else "Profil a risque, ajuster le programme" if proba >= PROBA_SEUIL_RISQUE
        else "Profil critique, revoir les bases"
    )
    return {"proba": proba, "interpretation": interpretation}


SEUIL_STABILITE_KG = 0.15


def compute_progress_status(client: pd.Series, poids_actuel: float) -> dict:
    """Statut de progression base sur le rapprochement (ou l'eloignement) du poids
    cible, quel que soit l'objectif (seche/prise_masse/recomposition) - fonctionne
    de facon uniforme sans logique specifique par type d'objectif.

    Le seuil de 0.15 kg pour "stable" n'est pas arbitraire : une pesee quotidienne
    varie naturellement de plusieurs centaines de grammes a plusieurs kg d'un jour
    a l'autre (hydratation, digestion, cycle) - un ecart hebdomadaire sous ce seuil
    est dans le bruit de mesure normal d'un pese-personne grand public, pas un
    signal de progression reelle. Voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md."""
    poids_initial = client["poids_initial_kg"]
    poids_cible = client["poids_cible_kg"]
    distance_initiale = abs(poids_initial - poids_cible)
    distance_actuelle = abs(poids_actuel - poids_cible)
    ecart = distance_initiale - distance_actuelle  # positif = rapprochement du but

    if abs(ecart) < SEUIL_STABILITE_KG:
        return {"icone": "⚪", "libelle": "Stable", "couleur": "gray"}
    if ecart > 0:
        return {"icone": "🟢", "libelle": "En progression", "couleur": "green"}
    return {"icone": "🔴", "libelle": "En regression", "couleur": "red"}


def progress_pct(client: pd.Series, poids_actuel: float) -> float:
    """Pourcentage de progression vers l'objectif (0 = poids de depart, 100 = objectif
    atteint), independant de la direction (perte ou prise de poids). Peut depasser 100
    ou etre negatif si le client s'est eloigne au-dela de son point de depart."""
    poids_initial = client["poids_initial_kg"]
    poids_cible = client["poids_cible_kg"]
    denom = (poids_cible - poids_initial)
    if abs(denom) < 1e-6:
        return 100.0
    return float((poids_actuel - poids_initial) / denom * 100)
