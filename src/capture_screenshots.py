"""Script ponctuel (hors pipeline) pour capturer des screenshots propres des 6 pages
de l'application Streamlit
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "docs" / "screenshots"
APP_URL = "http://localhost:8501"

PAGES = [
    ("1_presentation.png", "Presentation du projet"),
    ("2_pipeline_etl.png", "Pipeline ETL"),
    ("3_comparaison_modeles.png", "Comparaison des modeles"),
    ("4_prediction_temps_reel.png", "Prediction en temps reel"),
    ("5_dashboard_suivi_clients.png", "Dashboard de suivi clients"),
    ("6_gestion_de_projet.png", "Gestion de projet"),
]


def click_label_containing(page, text):
    page.locator("label", has_text=text).first.click()
    page.wait_for_timeout(1500)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        for filename, label_text in PAGES:
            click_label_containing(page, label_text)

            if label_text == "Prediction en temps reel":
                page.get_by_role("button", name="Predire").click()
                page.wait_for_timeout(2000)

            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT_DIR / filename), full_page=True)
            print(f"Capture sauvegardee : {filename}")

        browser.close()


if __name__ == "__main__":
    main()
