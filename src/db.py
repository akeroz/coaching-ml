"""Couche de persistance pour les VRAIS clients du coach (distincte du dataset
synthetique utilise pour l'entrainement du modele).

Base de donnees hebergee (Postgres, ex. Supabase) plutot que fichier local : c'est
ce qui permet d'acceder aux memes donnees depuis n'importe quel appareil (telephone,
autre ordinateur). La chaine de connexion (DATABASE_URL) est un secret : jamais
committee, jamais codee en dur - lue depuis une variable d'environnement (fichier
.env local, gitignore) ou depuis les secrets Streamlit Cloud en production. Voir
docs/RGPD_AI_ACT.md pour le detail des mesures de securite (acces restreint a
l'application, chiffrement au repos assure par l'hebergeur, aucune donnee
d'identite dans le dataset d'entrainement)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    prenom TEXT NOT NULL,
    nom TEXT NOT NULL,
    age INTEGER NOT NULL,
    sexe TEXT NOT NULL,
    taille_cm REAL NOT NULL,
    poids_initial_kg REAL NOT NULL,
    poids_cible_kg REAL NOT NULL,
    objectif TEXT NOT NULL,
    niveau TEXT NOT NULL,
    frequence_entrainement_semaine INTEGER NOT NULL,
    calories_quotidiennes REAL NOT NULL,
    proteines_g_par_jour REAL NOT NULL,
    heures_sommeil REAL NOT NULL,
    semaines_suivi_prevues INTEGER NOT NULL,
    adherence_programme_pct REAL NOT NULL,
    date_creation TEXT NOT NULL,
    objectif_atteint INTEGER,
    actif INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS suivis_hebdo (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients (client_id),
    date_saisie TEXT NOT NULL,
    poids REAL NOT NULL,
    note TEXT
);
"""

# Migrations additives (idempotentes) pour les colonnes ajoutees apres la creation
# initiale des tables - evite d'avoir a recreer la base a chaque evolution du schema.
MIGRATIONS = """
ALTER TABLE clients ADD COLUMN IF NOT EXISTS consentement_recueilli INTEGER NOT NULL DEFAULT 0;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS date_consentement TEXT;
ALTER TABLE suivis_hebdo ADD COLUMN IF NOT EXISTS energie INTEGER;
ALTER TABLE suivis_hebdo ADD COLUMN IF NOT EXISTS tour_taille_cm REAL;
"""

_engine: Engine | None = None
_db_initialized: bool = False


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL n'est pas configuree. En local, definissez-la dans un "
            "fichier .env (voir .env.example). Sur Streamlit Cloud, ajoutez-la dans "
            "les Secrets de l'application."
        )
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
    return _engine


@contextmanager
def get_connection():
    with get_engine().begin() as conn:
        yield conn


def init_db():
    """Cree les tables/colonnes si besoin. N'execute les commandes DDL qu'une seule
    fois par processus (evite un aller-retour reseau superflu a chaque lecture, qui
    ralentissait sensiblement chaque interaction avec l'application)."""
    global _db_initialized
    if _db_initialized:
        return
    with get_connection() as conn:
        for statement in SCHEMA.strip().split(";\n\n"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        for statement in MIGRATIONS.strip().split(";\n"):
            statement = statement.strip().rstrip(";")
            if statement:
                conn.execute(text(statement))
    _db_initialized = True


def _next_client_id(conn) -> str:
    count = conn.execute(text("SELECT COUNT(*) FROM clients")).scalar()
    return f"REAL_{count + 1:03d}"


def add_client(profile: dict, consentement_recueilli: bool = False) -> str:
    """Insere un nouveau client reel. Retourne le client_id genere.

    consentement_recueilli : a cocher explicitement par le coach lorsque le client
    a ete informe de la collecte de ses donnees (voir docs/RGPD_AI_ACT.md, art. 15-20
    RGPD - preuve horodatee du recueil du consentement)."""
    init_db()
    with get_connection() as conn:
        client_id = _next_client_id(conn)
        conn.execute(
            text("""INSERT INTO clients (
                client_id, prenom, nom, age, sexe, taille_cm, poids_initial_kg,
                poids_cible_kg, objectif, niveau, frequence_entrainement_semaine,
                calories_quotidiennes, proteines_g_par_jour, heures_sommeil,
                semaines_suivi_prevues, adherence_programme_pct, date_creation,
                objectif_atteint, actif, consentement_recueilli, date_consentement
            ) VALUES (
                :client_id, :prenom, :nom, :age, :sexe, :taille_cm, :poids_initial_kg,
                :poids_cible_kg, :objectif, :niveau, :frequence_entrainement_semaine,
                :calories_quotidiennes, :proteines_g_par_jour, :heures_sommeil,
                :semaines_suivi_prevues, :adherence_programme_pct, :date_creation,
                NULL, 1, :consentement_recueilli, :date_consentement
            )"""),
            {
                **profile, "client_id": client_id, "date_creation": date.today().isoformat(),
                "consentement_recueilli": int(consentement_recueilli),
                "date_consentement": date.today().isoformat() if consentement_recueilli else None,
            },
        )
        conn.execute(
            text("INSERT INTO suivis_hebdo (client_id, date_saisie, poids, note) "
                 "VALUES (:client_id, :date_saisie, :poids, :note)"),
            {"client_id": client_id, "date_saisie": date.today().isoformat(),
             "poids": profile["poids_initial_kg"], "note": "Poids de depart"},
        )
        return client_id


EDITABLE_FIELDS = [
    "prenom", "nom", "age", "sexe", "taille_cm", "poids_initial_kg", "poids_cible_kg",
    "objectif", "niveau", "frequence_entrainement_semaine", "calories_quotidiennes",
    "proteines_g_par_jour", "heures_sommeil", "semaines_suivi_prevues", "adherence_programme_pct",
]


def update_client(client_id: str, profile: dict):
    """Met a jour les informations d'un client existant (correction d'une erreur de saisie)."""
    init_db()
    with get_connection() as conn:
        set_clause = ", ".join(f"{field} = :{field}" for field in EDITABLE_FIELDS)
        params = {field: profile[field] for field in EDITABLE_FIELDS}
        params["client_id"] = client_id
        conn.execute(text(f"UPDATE clients SET {set_clause} WHERE client_id = :client_id"), params)


def update_client_status(client_id: str, actif: bool = True, objectif_atteint: int | None = None):
    init_db()
    with get_connection() as conn:
        if objectif_atteint is not None:
            conn.execute(
                text("UPDATE clients SET actif = :actif, objectif_atteint = :objectif_atteint "
                     "WHERE client_id = :client_id"),
                {"actif": int(actif), "objectif_atteint": objectif_atteint, "client_id": client_id},
            )
        else:
            conn.execute(
                text("UPDATE clients SET actif = :actif WHERE client_id = :client_id"),
                {"actif": int(actif), "client_id": client_id},
            )


def delete_client(client_id: str):
    init_db()
    with get_connection() as conn:
        conn.execute(text("DELETE FROM suivis_hebdo WHERE client_id = :client_id"), {"client_id": client_id})
        conn.execute(text("DELETE FROM clients WHERE client_id = :client_id"), {"client_id": client_id})


def add_weigh_in(
    client_id: str, poids: float, note: str = "", date_saisie: str | None = None,
    energie: int | None = None, tour_taille_cm: float | None = None,
):
    """energie : ressenti du client sur une echelle de 1 (epuise) a 5 (en pleine forme),
    optionnel. tour_taille_cm : mensuration optionnelle, complementaire au poids seul."""
    init_db()
    with get_connection() as conn:
        conn.execute(
            text("INSERT INTO suivis_hebdo (client_id, date_saisie, poids, note, energie, tour_taille_cm) "
                 "VALUES (:client_id, :date_saisie, :poids, :note, :energie, :tour_taille_cm)"),
            {"client_id": client_id, "date_saisie": date_saisie or date.today().isoformat(),
             "poids": poids, "note": note, "energie": energie, "tour_taille_cm": tour_taille_cm},
        )


def get_all_clients() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(text("SELECT * FROM clients ORDER BY date_creation DESC"), conn)


def get_client(client_id: str) -> pd.Series | None:
    df = get_all_clients()
    match = df[df["client_id"] == client_id]
    return match.iloc[0] if not match.empty else None


def get_weigh_ins(client_id: str) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            text("SELECT * FROM suivis_hebdo WHERE client_id = :client_id ORDER BY date_saisie"),
            conn, params={"client_id": client_id},
        )


def get_labelled_clients() -> pd.DataFrame:
    """Clients reels dont l'issue est connue (objectif_atteint renseigne) - utilisables
    pour le reentrainement du modele."""
    df = get_all_clients()
    return df[df["objectif_atteint"].notna()]


def get_all_weigh_ins() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(text("SELECT * FROM suivis_hebdo ORDER BY client_id, date_saisie"), conn)


def export_all_data() -> dict[str, pd.DataFrame]:
    """Sauvegarde complete des vraies donnees (clients + historique des suivis),
    a la demande du coach - filet de securite independant de l'hebergeur."""
    return {"clients": get_all_clients(), "suivis_hebdo": get_all_weigh_ins()}
