"""Generation d'un resume PDF par client (a envoyer au client ou a archiver),
sans exposer le dashboard complet du coach."""

from __future__ import annotations

from datetime import date

import pandas as pd
from fpdf import FPDF


def build_client_pdf(client: pd.Series, weigh_ins: pd.DataFrame, prediction: dict | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Suivi de progression - Coaching @builtbyarthur", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Genere le {date.today().isoformat()}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"{client['prenom']} {client['nom']}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Objectif : {client['objectif']}  |  Niveau : {client['niveau']}", ln=True)
    pdf.cell(0, 7, f"Poids initial : {client['poids_initial_kg']} kg  ->  Poids cible : {client['poids_cible_kg']} kg", ln=True)
    pdf.ln(4)

    if not weigh_ins.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Historique des pesees", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, row in weigh_ins.iterrows():
            note = f" - {row['note']}" if row.get("note") else ""
            pdf.cell(0, 6, f"{row['date_saisie']} : {row['poids']} kg{note}", ln=True)
        pdf.ln(4)

        poids_actuel = weigh_ins.iloc[-1]["poids"]
        delta = poids_actuel - client["poids_initial_kg"]
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"Variation depuis le depart : {delta:+.1f} kg", ln=True)
        pdf.ln(2)

    if prediction is not None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Estimation du modele (aide a la decision)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            0, 6,
            f"Probabilite estimee d'atteinte de l'objectif : {prediction['proba']:.0%}\n"
            f"Interpretation : {prediction['interpretation']}\n\n"
            "Cette estimation est une aide statistique a la decision du coach, "
            "pas une decision automatique ni un avis medical.",
        )

    return bytes(pdf.output())
