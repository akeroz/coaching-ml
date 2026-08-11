"""Profils clients fictifs varies, utilises pour peupler data/coaching.db a des fins
de demonstration (interface "Mes clients" : fiche client, indicateurs de progression,
export PDF, cloture) - jamais de personne reelle.

Utilisation :
- En script ponctuel : python src/seed_demo_clients.py
- Importe automatiquement par app.py sur l'instance Streamlit Cloud (DEMO_MODE=true
  dans les secrets), pour repeupler une base vide apres un redemarrage de conteneur
  (le stockage de Streamlit Cloud n'est pas garanti persistant)."""

from datetime import date, timedelta

from sqlalchemy import text

import db

TODAY = date.today()


def d(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


# (profil, poids releves sur plusieurs semaines - le dernier point determine la tendance affichee)
PROFILES = [
    {
        "profile": {
            "prenom": "Lucas", "nom": "Meyer", "age": 19, "sexe": "H", "taille_cm": 179,
            "poids_initial_kg": 88.0, "poids_cible_kg": 78.0, "objectif": "seche",
            "niveau": "debutant", "frequence_entrainement_semaine": 4,
            "calories_quotidiennes": 2100, "proteines_g_par_jour": 160, "heures_sommeil": 7.5,
            "semaines_suivi_prevues": 14, "adherence_programme_pct": 85,
        },
        "note": "Etudiant, emploi du temps flexible, bonne adherence",
        "poids_history": [88.0, 87.2, 86.3, 85.1, 84.0],  # en bonne voie -> vert
    },
    {
        "profile": {
            "prenom": "Nathan", "nom": "Roche", "age": 22, "sexe": "H", "taille_cm": 174,
            "poids_initial_kg": 65.0, "poids_cible_kg": 73.0, "objectif": "prise_masse",
            "niveau": "intermediaire", "frequence_entrainement_semaine": 5,
            "calories_quotidiennes": 3100, "proteines_g_par_jour": 150, "heures_sommeil": 6.5,
            "semaines_suivi_prevues": 16, "adherence_programme_pct": 80,
        },
        "note": "Travailleur temps plein, entraine le soir",
        "poids_history": [65.0, 66.1, 67.3, 68.6, 69.8],  # se rapproche -> vert
    },
    {
        "profile": {
            "prenom": "Hugo", "nom": "Blanchard", "age": 24, "sexe": "H", "taille_cm": 182,
            "poids_initial_kg": 80.0, "poids_cible_kg": 78.0, "objectif": "recomposition",
            "niveau": "avance", "frequence_entrainement_semaine": 5,
            "calories_quotidiennes": 2500, "proteines_g_par_jour": 180, "heures_sommeil": 6.0,
            "semaines_suivi_prevues": 12, "adherence_programme_pct": 55,
        },
        "note": "Alternance, stress eleve, adherence irreguliere",
        "poids_history": [80.0, 80.8, 81.5, 82.3, 83.0],  # s'eloigne -> rouge
    },
    {
        "profile": {
            "prenom": "Adrien", "nom": "Fontaine", "age": 18, "sexe": "H", "taille_cm": 176,
            "poids_initial_kg": 95.0, "poids_cible_kg": 82.0, "objectif": "seche",
            "niveau": "debutant", "frequence_entrainement_semaine": 2,
            "calories_quotidiennes": 2600, "proteines_g_par_jour": 130, "heures_sommeil": 6.0,
            "semaines_suivi_prevues": 18, "adherence_programme_pct": 40,
        },
        "note": "Premiere annee d'etudes, faible adherence au programme",
        "poids_history": [95.0, 95.6, 96.4, 97.1, 97.8],  # regresse -> rouge
    },
    {
        "profile": {
            "prenom": "Karim", "nom": "Belkacem", "age": 21, "sexe": "H", "taille_cm": 170,
            "poids_initial_kg": 62.0, "poids_cible_kg": 70.0, "objectif": "prise_masse",
            "niveau": "debutant", "frequence_entrainement_semaine": 4,
            "calories_quotidiennes": 2900, "proteines_g_par_jour": 140, "heures_sommeil": 7.0,
            "semaines_suivi_prevues": 16, "adherence_programme_pct": 90,
        },
        "note": "Livreur, tres actif physiquement, tres bonne adherence",
        "poids_history": [62.0, 63.0, 64.1, 65.3, 66.5],  # se rapproche -> vert
    },
    {
        "profile": {
            "prenom": "Bilal", "nom": "Ouazzani", "age": 25, "sexe": "H", "taille_cm": 177,
            "poids_initial_kg": 79.0, "poids_cible_kg": 78.5, "objectif": "recomposition",
            "niveau": "intermediaire", "frequence_entrainement_semaine": 3,
            "calories_quotidiennes": 2400, "proteines_g_par_jour": 155, "heures_sommeil": 7.0,
            "semaines_suivi_prevues": 10, "adherence_programme_pct": 70,
        },
        "note": "Travail de bureau sedentaire, poids proche de l'objectif",
        "poids_history": [79.0, 78.8, 78.9, 78.6, 78.7],  # stable -> gris/vert leger
    },
]


def seed_demo_clients(verbose: bool = True) -> list[str]:
    """Cree les profils fictifs de demonstration. Retourne la liste des client_id crees."""
    created = []
    for entry in PROFILES:
        profile = entry["profile"]
        client_id = db.add_client(profile)

        # add_client() insere deja une premiere pesee "aujourd'hui" avec le poids initial ;
        # on la remplace par un historique etale sur les semaines precedentes.
        with db.get_connection() as conn:
            conn.execute(text("DELETE FROM suivis_hebdo WHERE client_id = :client_id"), {"client_id": client_id})

        n_points = len(entry["poids_history"])
        for i, poids in enumerate(entry["poids_history"]):
            days_ago = (n_points - 1 - i) * 7  # une pesee par semaine, la plus recente = aujourd'hui
            db.add_weigh_in(client_id, poids, note=entry["note"] if i == 0 else "", date_saisie=d(days_ago))

        created.append(client_id)
        if verbose:
            print(f"Cree : {client_id} - {profile['prenom']} {profile['nom']} ({entry['note']})")

    if verbose:
        print(f"\n{len(created)} clients de demonstration crees dans data/coaching.db")

    return created


if __name__ == "__main__":
    seed_demo_clients()
