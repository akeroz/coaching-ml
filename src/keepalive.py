"""Ping minimal de la base de donnees, execute periodiquement (GitHub Actions,
voir .github/workflows/keepalive.yml) pour eviter la mise en pause automatique
du projet Supabase gratuit apres 7 jours d'inactivite. Ne modifie aucune
donnee, se contente d'une lecture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

if __name__ == "__main__":
    clients = db.get_all_clients()
    print(f"Ping Supabase OK - {len(clients)} client(s) en base.")
