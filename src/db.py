"""Couche de persistance pour les VRAIS clients du coach (distincte du dataset
synthetique utilise pour l'entrainement du modele).

Le fichier data/coaching.db contient des donnees personnelles reelles des lors
qu'il est utilise en production : il est deliberement exclu du depot Git
(.gitignore) et ne doit jamais etre partage publiquement (voir docs/RGPD_AI_ACT
pour les principes appliques, notamment la minimisation et la separation des
donnees d'entrainement / donnees d'identite)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "coaching.db"

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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    date_saisie TEXT NOT NULL,
    poids REAL NOT NULL,
    note TEXT,
    FOREIGN KEY (client_id) REFERENCES clients (client_id)
);
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def _next_client_id(conn) -> str:
    row = conn.execute("SELECT COUNT(*) FROM clients").fetchone()
    return f"REAL_{row[0] + 1:03d}"


def add_client(profile: dict) -> str:
    """Insere un nouveau client reel. Retourne le client_id genere."""
    init_db()
    with get_connection() as conn:
        client_id = _next_client_id(conn)
        conn.execute(
            """INSERT INTO clients (
                client_id, prenom, nom, age, sexe, taille_cm, poids_initial_kg,
                poids_cible_kg, objectif, niveau, frequence_entrainement_semaine,
                calories_quotidiennes, proteines_g_par_jour, heures_sommeil,
                semaines_suivi_prevues, adherence_programme_pct, date_creation,
                objectif_atteint, actif
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1)""",
            (
                client_id, profile["prenom"], profile["nom"], profile["age"],
                profile["sexe"], profile["taille_cm"], profile["poids_initial_kg"],
                profile["poids_cible_kg"], profile["objectif"], profile["niveau"],
                profile["frequence_entrainement_semaine"], profile["calories_quotidiennes"],
                profile["proteines_g_par_jour"], profile["heures_sommeil"],
                profile["semaines_suivi_prevues"], profile["adherence_programme_pct"],
                date.today().isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO suivis_hebdo (client_id, date_saisie, poids, note) VALUES (?, ?, ?, ?)",
            (client_id, date.today().isoformat(), profile["poids_initial_kg"], "Poids de depart"),
        )
        return client_id


def update_client_status(client_id: str, actif: bool = True, objectif_atteint: int | None = None):
    init_db()
    with get_connection() as conn:
        if objectif_atteint is not None:
            conn.execute(
                "UPDATE clients SET actif = ?, objectif_atteint = ? WHERE client_id = ?",
                (int(actif), objectif_atteint, client_id),
            )
        else:
            conn.execute("UPDATE clients SET actif = ? WHERE client_id = ?", (int(actif), client_id))


def delete_client(client_id: str):
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM suivis_hebdo WHERE client_id = ?", (client_id,))
        conn.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))


def add_weigh_in(client_id: str, poids: float, note: str = ""):
    init_db()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO suivis_hebdo (client_id, date_saisie, poids, note) VALUES (?, ?, ?, ?)",
            (client_id, date.today().isoformat(), poids, note),
        )


def get_all_clients() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM clients ORDER BY date_creation DESC", conn)


def get_client(client_id: str) -> pd.Series | None:
    df = get_all_clients()
    match = df[df["client_id"] == client_id]
    return match.iloc[0] if not match.empty else None


def get_weigh_ins(client_id: str) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM suivis_hebdo WHERE client_id = ? ORDER BY date_saisie",
            conn, params=(client_id,),
        )


def get_labelled_clients() -> pd.DataFrame:
    """Clients reels dont l'issue est connue (objectif_atteint renseigne) - utilisables
    pour le reentrainement du modele."""
    df = get_all_clients()
    return df[df["objectif_atteint"].notna()]
