"""Configuration pytest partagee : rend src/ importable depuis les tests
(meme mecanisme que app.py, qui insere src/ dans sys.path au demarrage) et
charge .env avant que les tests evaluent os.environ (ex. skipif DATABASE_URL)."""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()
