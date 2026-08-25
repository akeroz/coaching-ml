"""Dashboard Streamlit - Prediction de progression coaching fitness (@builtbyarthur)."""

import json
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import db  # noqa: E402
from coaching_logic import (  # noqa: E402
    build_profile_from_client_row, compute_progress_status,
    local_feature_importance, predict_for_client, progress_pct,
)
from etl import CAT_COLS, FEATURE_COLUMNS, NUM_COLS, TARGET_COL  # noqa: E402
from predict import build_feature_row, load_artifacts  # noqa: E402
from report import build_client_pdf  # noqa: E402
from retrain_with_real_data import MIN_REAL_CLIENTS, run_retrain  # noqa: E402
from seed_demo_clients import seed_demo_clients  # noqa: E402

st.set_page_config(page_title="Coaching ML - builtbyarthur", page_icon="💪", layout="wide")

RAW_PATH = ROOT_DIR / "data" / "raw" / "clients_raw.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "dataset_final.csv"
RESULTS_PATH = ROOT_DIR / "models" / "results.json"
ARCHITECTURE_PNG = ROOT_DIR / "docs" / "architecture_diagram.png"
SELECTION_REPORT = ROOT_DIR / "docs" / "MODEL_SELECTION_REPORT.md"
RGPD_AI_ACT_DOC = ROOT_DIR / "docs" / "RGPD_AI_ACT.md"

_secrets_file_exists = (ROOT_DIR / ".streamlit" / "secrets.toml").exists()
if _secrets_file_exists:
    if "DATABASE_URL" in st.secrets:
        os.environ.setdefault("DATABASE_URL", st.secrets["DATABASE_URL"])
    DEMO_MODE = bool(st.secrets.get("DEMO_MODE", False))
else:
    DEMO_MODE = False

if "DATABASE_URL" not in os.environ:
    st.error(
        "DATABASE_URL n'est pas configuree. En local, creez un fichier .env "
        "(voir .env.example). Sur Streamlit Cloud, ajoutez DATABASE_URL dans les Secrets."
    )
    st.stop()

if DEMO_MODE and db.get_all_clients().empty:
    seed_demo_clients(verbose=False)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --violet: #6E11F4;
        --violet-dark: #4C0FB0;
        --violet-tint: #F3EBFF;
        --ink: #1A1A1A;
    }

    .app-header {
        background: linear-gradient(135deg, var(--violet), var(--violet-dark));
        padding: 1.5rem 1.8rem; margin-bottom: 1.6rem; border-radius: 16px;
        box-shadow: 0 8px 24px rgba(110,17,244,0.18);
    }
    .app-header .tag {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
        letter-spacing: 0.14em; text-transform: uppercase; color: rgba(255,255,255,0.7);
        display: block; margin-bottom: 0.35rem;
    }
    .app-header h1 {
        font-family: 'Playfair Display', serif; font-weight: 800; font-size: 2rem;
        color: #fff; margin: 0; line-height: 1.05;
    }
    .app-header p { margin: 0.45rem 0 0 0; color: rgba(255,255,255,0.85); font-size: 0.92rem; }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem; margin-bottom: 1.6rem; }
    .kpi-card {
        background: #fff; border: 1px solid #ECE3FC; border-radius: 14px;
        padding: 1.15rem 1.3rem; box-shadow: 0 2px 8px rgba(110,17,244,0.05);
    }
    .kpi-card .kpi-icon { font-size: 1.2rem; margin-bottom: 0.4rem; }
    .kpi-card .kpi-value {
        font-family: 'Playfair Display', serif; font-weight: 800;
        font-size: 2rem; line-height: 1.1; color: var(--ink);
    }
    .kpi-card .kpi-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.07em;
        text-transform: uppercase; color: var(--violet); margin-top: 0.35rem; font-weight: 600;
    }

    .client-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem; margin-bottom: 1.6rem; }
    .client-card {
        background: #fff; border: 1px solid #ECE3FC; border-radius: 14px;
        padding: 1.1rem 1.2rem; box-shadow: 0 2px 8px rgba(110,17,244,0.05);
    }
    .client-card-head { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.75rem; }
    .avatar {
        width: 42px; height: 42px; min-width: 42px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; color: #fff; font-size: 0.92rem;
    }
    .client-name { font-weight: 700; font-size: 0.95rem; color: var(--ink); }
    .client-meta {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
        letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.55;
    }
    .client-status { font-size: 0.82rem; margin-top: 0.6rem; color: var(--ink); }
    .progress-track { background: #F1ECFB; border-radius: 6px; height: 7px; margin-top: 0.6rem; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 6px; }

    .activity-feed { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
    .activity-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.6rem 0.9rem; background: #fff; border-radius: 10px;
        font-size: 0.85rem; border: 1px solid #ECE3FC; box-shadow: 0 1px 4px rgba(110,17,244,0.04);
    }
    .activity-item .activity-date {
        font-family: 'JetBrains Mono', monospace; opacity: 0.5; font-size: 0.72rem;
        white-space: nowrap; margin-left: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AVATAR_PALETTE = ["#6E11F4", "#9747FF", "#4C0FB0", "#C026D3", "#DB2777", "#059669", "#D97706", "#0EA5E9"]
STATUS_COLOR = {"green": "#059669", "red": "#DC2626", "gray": "#9CA3AF"}


def render_header(title: str, subtitle: str = "", tag: str = "COACHING ML"):
    st.markdown(
        f"""
        <div class="app-header">
            <span class="tag">{tag}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def avatar_color(seed: str) -> str:
    return AVATAR_PALETTE[sum(ord(c) for c in seed) % len(AVATAR_PALETTE)]


def render_kpi_cards(cards: list[dict]):
    html = ['<div class="kpi-grid">']
    for c in cards:
        html.append(
            f'<div class="kpi-card"><div class="kpi-icon">{c["icon"]}</div>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-label">{c["label"]}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_client_cards(rows: list[dict]):
    html = ['<div class="client-grid">']
    for r in rows:
        initials = (r["prenom"][:1] + r["nom"][:1]).upper()
        color = avatar_color(r["prenom"] + r["nom"])
        pct = max(0, min(100, r["progress_pct"]))
        bar_color = STATUS_COLOR[r["couleur"]]
        html.append(
            f'<div class="client-card">'
            f'<div class="client-card-head">'
            f'<div class="avatar" style="background:{color}">{initials}</div>'
            f'<div><div class="client-name">{r["prenom"]} {r["nom"]}</div>'
            f'<div class="client-meta">{r["objectif"]} · {r["niveau"]}</div></div>'
            f'</div>'
            f'<div class="client-status">{r["icone"]} {r["libelle"]} — {r["poids_actuel"]:.1f} kg / {r["poids_cible"]:.1f} kg cible</div>'
            f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{bar_color}"></div></div>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_activity_feed(items: list[dict]):
    if not items:
        st.caption("Aucune activite recente.")
        return
    html = ['<div class="activity-feed">']
    for it in items:
        html.append(
            f'<div class="activity-item"><span>{it["text"]}</span>'
            f'<span class="activity-date">{it["date"]}</span></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


@st.cache_data
def load_raw():
    return pd.read_csv(RAW_PATH)


@st.cache_data
def load_processed():
    return pd.read_csv(PROCESSED_PATH)


@st.cache_data
def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model_artifacts():
    return load_artifacts()


@st.cache_data(ttl=30)
def cached_clients() -> pd.DataFrame:
    return db.get_all_clients()


@st.cache_data(ttl=30)
def cached_weigh_ins_by_client() -> dict:
    """Toutes les pesees en un seul aller-retour reseau vers la base, regroupees par
    client - remplace une requete Supabase separee par client (N+1), qui ralentissait
    sensiblement l'app des que le nombre de clients augmentait."""
    all_weigh_ins = db.get_all_weigh_ins()
    if all_weigh_ins.empty:
        return {}
    return {cid: grp.reset_index(drop=True) for cid, grp in all_weigh_ins.groupby("client_id")}


EMPTY_WEIGH_INS = pd.DataFrame(columns=["id", "client_id", "date_saisie", "poids", "note", "energie", "tour_taille_cm"])


def invalidate_client_caches():
    """A appeler apres tout ajout/modification/suppression de client ou de pesee,
    pour que l'affichage reflete immediatement le changement."""
    cached_clients.clear()
    cached_weigh_ins_by_client.clear()


def refresh_after_write(message: str):
    """Invalide le cache, programme un toast de confirmation et relance l'affichage
    immediatement (au lieu de demander a l'utilisateur de recharger la page a la main)."""
    invalidate_client_caches()
    st.session_state["_toast_message"] = message
    st.rerun()


ESPACE_COACH = ["Accueil", "Mes clients", "Prediction en temps reel"]
ESPACE_TECHNIQUE = [
    "Presentation du projet",
    "Pipeline ETL",
    "Comparaison des modeles",
    "Demo dashboard (donnees simulees)",
    "Reentrainement (avance)",
    "Gestion de projet",
]

PAGE_ICONS = {
    "Accueil": "🏠  Accueil",
    "Mes clients": "👥  Mes clients",
    "Prediction en temps reel": "🔮  Prediction en temps reel",
    "Presentation du projet": "📄  Presentation du projet",
    "Pipeline ETL": "🔧  Pipeline ETL",
    "Comparaison des modeles": "📊  Comparaison des modeles",
    "Demo dashboard (donnees simulees)": "🎭  Demo dashboard (simulee)",
    "Reentrainement (avance)": "🔁  Reentrainement (avance)",
    "Gestion de projet": "📋  Gestion de projet",
}
ESPACE_ICONS = {"Espace coach": "🧑‍💼  Espace coach", "Documentation technique": "🛠️  Documentation technique"}

st.sidebar.markdown(
    "<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;letter-spacing:0.1em;"
    "text-transform:uppercase;color:#6E11F4;font-weight:600;margin-bottom:0.3rem;'>Coaching ML</div>",
    unsafe_allow_html=True,
)
espace = st.sidebar.radio(
    "Espace", ["Espace coach", "Documentation technique"], format_func=lambda x: ESPACE_ICONS.get(x, x)
)
st.sidebar.markdown("---")
if espace == "Espace coach":
    page = st.sidebar.radio("Navigation", ESPACE_COACH, format_func=lambda x: PAGE_ICONS.get(x, x))
else:
    page = st.sidebar.radio("Navigation", ESPACE_TECHNIQUE, format_func=lambda x: PAGE_ICONS.get(x, x))
    st.sidebar.caption("Pages de documentation / certification / maintenance avancee.")

# ----------------------------------------------------------------------------
# ACCUEIL
# ----------------------------------------------------------------------------
if page == "Accueil":
    render_header("Accueil", "Vue d'ensemble de votre activite de coaching", tag="TABLEAU DE BORD")
    if DEMO_MODE:
        st.caption(
            "🎭 Mode demo : les clients affiches sont des profils fictifs generes pour "
            "illustrer le fonctionnement de l'application, pas de vrais clients."
        )

    clients_df = cached_clients()
    if clients_df.empty:
        st.info(
            "Aucun client enregistre pour le moment. Rendez-vous dans \"Mes clients\" "
            "pour ajouter votre premier client."
        )
    else:
        actifs_df = clients_df[clients_df["actif"] == 1]
        clotures_df = clients_df[clients_df["objectif_atteint"].notna()]
        taux_reussite = clotures_df["objectif_atteint"].mean() if not clotures_df.empty else None

        model, scaler, encoders = load_model_artifacts()

        # Une seule requete groupee pour toutes les pesees (au lieu d'une par client)
        all_weigh_ins = cached_weigh_ins_by_client()
        weigh_ins_by_client = {cid: all_weigh_ins.get(cid, EMPTY_WEIGH_INS) for cid in actifs_df["client_id"]}

        risques, portfolio_rows, activity_rows, stale = [], [], [], []
        for _, client in actifs_df.iterrows():
            nom_complet = f"{client['prenom']} {client['nom']}"
            pred = predict_for_client(client, model, scaler, encoders)
            if pred["proba"] < 0.70:
                risques.append({"client": nom_complet, "probabilite": pred["proba"], "statut": pred["interpretation"]})

            weigh_ins = weigh_ins_by_client[client["client_id"]]
            if not weigh_ins.empty:
                dates = pd.to_datetime(weigh_ins["date_saisie"])
                first_date = dates.min()
                last_date = dates.max()

                for (_, w), wdate in zip(weigh_ins.iterrows(), dates):
                    semaine = (wdate - first_date).days // 7
                    portfolio_rows.append({
                        "client": nom_complet, "semaine": semaine,
                        "progression_pct": progress_pct(client, w["poids"]),
                    })

                last_row = weigh_ins.iloc[-1]
                note_suffix = f" — {last_row['note']}" if last_row.get("note") else ""
                activity_rows.append({
                    "text": f"{nom_complet} — pesee a {last_row['poids']:.1f} kg{note_suffix}",
                    "date": last_date.date().isoformat(), "sort_key": last_date,
                })

                days_since = (pd.Timestamp.today().normalize() - last_date).days
                if days_since >= 10:
                    stale.append({"client": nom_complet, "derniere_pesee": last_date.date().isoformat(), "jours_ecoules": days_since})

        render_kpi_cards([
            {"icon": "👥", "value": len(actifs_df), "label": "Clients actifs"},
            {"icon": "🎯", "value": f"{taux_reussite:.0%}" if taux_reussite is not None else "N/A", "label": "Taux de reussite (clotures)"},
            {"icon": "⚠️", "value": len(risques), "label": "Clients a surveiller"},
            {"icon": "📋", "value": len(clotures_df), "label": "Suivis clotures"},
        ])

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Progression du portefeuille")
            if portfolio_rows:
                portfolio_df = pd.DataFrame(portfolio_rows)
                fig_portfolio = px.line(
                    portfolio_df, x="semaine", y="progression_pct", color="client", markers=True,
                    labels={"semaine": "Semaines depuis la 1ere pesee", "progression_pct": "Progression vers l'objectif (%)"},
                )
                fig_portfolio.add_hline(y=100, line_dash="dot", opacity=0.3, annotation_text="Objectif")
                fig_portfolio.add_hline(y=0, line_dash="dot", opacity=0.15)
                fig_portfolio.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_portfolio, use_container_width=True)
                st.caption("100% = objectif atteint, 0% = poids de depart, valeurs negatives = eloignement de l'objectif.")
            else:
                st.caption("Aucune pesee enregistree pour le moment.")

        with col_right:
            st.subheader("Activite recente")
            activity_rows.sort(key=lambda r: r["sort_key"], reverse=True)
            render_activity_feed(activity_rows[:8])

        st.subheader("Clients a surveiller")
        if risques:
            risques_df = pd.DataFrame(risques).sort_values("probabilite")
            risques_df["probabilite"] = (risques_df["probabilite"] * 100).round(1).astype(str) + " %"
            st.dataframe(risques_df[["client", "probabilite", "statut"]], use_container_width=True, hide_index=True)
            st.caption("Direction \"Mes clients\" pour ajuster leur programme.")
        else:
            st.success("Aucun client a risque actuellement.")

        st.subheader("Suivis a jour")
        if stale:
            st.warning(f"{len(stale)} client(s) sans pesee depuis plus de 10 jours :")
            st.dataframe(pd.DataFrame(stale), use_container_width=True, hide_index=True)
        else:
            st.caption("Tous les clients actifs ont une pesee recente.")

# ----------------------------------------------------------------------------
# MES CLIENTS
# ----------------------------------------------------------------------------
elif page == "Mes clients":
    if "_toast_message" in st.session_state:
        st.toast(st.session_state.pop("_toast_message"))
    render_header("Mes clients", "Gestion des clients reels, suivi hebdomadaire et export", tag="GESTION CLIENTS")
    if DEMO_MODE:
        st.caption(
            "🎭 Mode demo : les clients ci-dessous sont des profils fictifs generes pour "
            "illustrer le fonctionnement de l'application, pas de vrais clients."
        )

    tab_liste, tab_ajout, tab_suivi, tab_export = st.tabs(
        ["Mes clients", "Ajouter un client", "Suivi hebdomadaire", "Sauvegarde"]
    )

    with tab_liste:
        clients_df = cached_clients()
        if clients_df.empty:
            st.write("Aucun client enregistre pour le moment. Utilisez l'onglet \"Ajouter un client\".")
        else:
            all_weigh_ins = cached_weigh_ins_by_client()
            card_rows = []
            for _, c in clients_df.iterrows():
                w = all_weigh_ins.get(c["client_id"], EMPTY_WEIGH_INS)
                poids_actuel = w.iloc[-1]["poids"] if not w.empty else c["poids_initial_kg"]
                statut = compute_progress_status(c, poids_actuel)
                card_rows.append({
                    "prenom": c["prenom"], "nom": c["nom"], "objectif": c["objectif"], "niveau": c["niveau"],
                    "poids_actuel": poids_actuel, "poids_cible": c["poids_cible_kg"],
                    "progress_pct": progress_pct(c, poids_actuel),
                    **statut,
                })
            render_client_cards(card_rows)
            st.caption("🟢 se rapproche du poids cible · 🔴 s'en eloigne · ⚪ stable (variation < 150g). La barre indique la progression vers l'objectif.")

            with st.expander("Voir le tableau complet (dates, statut de cloture...)"):
                st.dataframe(
                    clients_df[[
                        "client_id", "prenom", "nom", "objectif", "niveau", "poids_initial_kg",
                        "poids_cible_kg", "date_creation", "objectif_atteint", "actif",
                    ]],
                    use_container_width=True, hide_index=True,
                )

            st.subheader("Fiche client")
            selected_id = st.selectbox(
                "Choisir un client", clients_df["client_id"],
                format_func=lambda cid: f"{cid} - " + clients_df.set_index('client_id').loc[cid, 'prenom'] + " " + clients_df.set_index('client_id').loc[cid, 'nom'],
            )
            client = clients_df.set_index("client_id").loc[selected_id]
            weigh_ins = all_weigh_ins.get(selected_id, EMPTY_WEIGH_INS)
            poids_actuel = weigh_ins.iloc[-1]["poids"] if not weigh_ins.empty else client["poids_initial_kg"]
            statut = compute_progress_status(client, poids_actuel)

            col1, col2 = st.columns([2, 1])
            with col1:
                if not weigh_ins.empty:
                    fig = px.line(weigh_ins, x="date_saisie", y="poids", markers=True,
                                  title=f"Evolution du poids - {client['prenom']} {client['nom']}")
                    fig.add_hline(y=client["poids_cible_kg"], line_dash="dot",
                                  annotation_text="Objectif")
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"### {statut['icone']} {statut['libelle']}")
                st.metric("Poids initial", f"{client['poids_initial_kg']} kg")
                st.metric(
                    "Poids actuel", f"{poids_actuel} kg" if not weigh_ins.empty else "N/A",
                    delta=f"{poids_actuel - client['poids_initial_kg']:+.1f} kg" if not weigh_ins.empty else None,
                    delta_color="off",
                )
                st.metric("Poids cible", f"{client['poids_cible_kg']} kg")

            st.subheader("Modifier les informations")
            with st.expander("Corriger une erreur de saisie"):
                with st.form(f"edit_form_{selected_id}"):
                    ecol1, ecol2, ecol3 = st.columns(3)
                    with ecol1:
                        e_prenom = st.text_input("Prenom", value=client["prenom"])
                        e_nom = st.text_input("Nom", value=client["nom"])
                        e_age = st.slider("Age", 18, 80, int(client["age"]))
                        e_sexe = st.selectbox("Sexe", ["H", "F"], index=["H", "F"].index(client["sexe"]))
                    with ecol2:
                        e_taille = st.slider("Taille (cm)", 140, 220, int(client["taille_cm"]))
                        e_poids_initial = st.number_input("Poids initial (kg)", 40.0, 200.0, float(client["poids_initial_kg"]))
                        e_poids_cible = st.number_input("Poids cible (kg)", 40.0, 200.0, float(client["poids_cible_kg"]))
                        e_objectif = st.selectbox(
                            "Objectif", ["seche", "prise_masse", "recomposition"],
                            index=["seche", "prise_masse", "recomposition"].index(client["objectif"]),
                        )
                        e_niveau = st.selectbox(
                            "Niveau", ["debutant", "intermediaire", "avance"],
                            index=["debutant", "intermediaire", "avance"].index(client["niveau"]),
                        )
                    with ecol3:
                        e_freq = st.slider("Frequence d'entrainement/semaine", 1, 7, int(client["frequence_entrainement_semaine"]))
                        e_calories = st.number_input("Calories quotidiennes", 1000, 6000, int(client["calories_quotidiennes"]))
                        e_proteines = st.number_input("Proteines (g/jour)", 40, 400, int(client["proteines_g_par_jour"]))
                        e_sommeil = st.slider("Heures de sommeil", 3.0, 12.0, float(client["heures_sommeil"]), step=0.5)
                        e_semaines = st.slider("Semaines de suivi prevues", 1, 52, int(client["semaines_suivi_prevues"]))
                        e_adherence = st.slider("Adherence estimee (%)", 0, 100, int(client["adherence_programme_pct"]))

                    if st.form_submit_button("Enregistrer les modifications"):
                        db.update_client(selected_id, {
                            "prenom": e_prenom, "nom": e_nom, "age": e_age, "sexe": e_sexe,
                            "taille_cm": e_taille, "poids_initial_kg": e_poids_initial,
                            "poids_cible_kg": e_poids_cible, "objectif": e_objectif, "niveau": e_niveau,
                            "frequence_entrainement_semaine": e_freq, "calories_quotidiennes": e_calories,
                            "proteines_g_par_jour": e_proteines, "heures_sommeil": e_sommeil,
                            "semaines_suivi_prevues": e_semaines, "adherence_programme_pct": e_adherence,
                        })
                        refresh_after_write("Informations mises a jour.")

            st.subheader("Cloturer le suivi")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Marquer : objectif atteint"):
                    db.update_client_status(selected_id, actif=False, objectif_atteint=1)
                    refresh_after_write("Client marque comme objectif atteint.")
            with col_b:
                if st.button("Marquer : objectif non atteint"):
                    db.update_client_status(selected_id, actif=False, objectif_atteint=0)
                    refresh_after_write("Client marque comme objectif non atteint.")
            with col_c:
                if st.button("Supprimer ce client", type="secondary"):
                    db.delete_client(selected_id)
                    refresh_after_write("Client supprime.")

            st.subheader("Export PDF")
            model, scaler, encoders = load_model_artifacts()
            prediction = predict_for_client(client, model, scaler, encoders)
            pdf_bytes = build_client_pdf(client, weigh_ins, prediction)
            st.download_button(
                "Telecharger le resume PDF", data=pdf_bytes,
                file_name=f"suivi_{client['prenom']}_{client['nom']}.pdf", mime="application/pdf",
            )

    with tab_ajout:
        st.write("Ajoutez un vrai client pour commencer son suivi.")
        with st.form("ajout_client_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                prenom = st.text_input("Prenom")
                nom = st.text_input("Nom")
                age = st.slider("Age", 18, 80, 30)
                sexe = st.selectbox("Sexe", ["H", "F"])
            with col2:
                taille_cm = st.slider("Taille (cm)", 140, 220, 175)
                poids_initial_kg = st.number_input("Poids initial (kg)", 40.0, 200.0, 80.0)
                poids_cible_kg = st.number_input("Poids cible (kg)", 40.0, 200.0, 75.0)
                objectif = st.selectbox("Objectif", ["seche", "prise_masse", "recomposition"])
                niveau = st.selectbox("Niveau", ["debutant", "intermediaire", "avance"])
            with col3:
                frequence_entrainement_semaine = st.slider("Frequence d'entrainement/semaine", 1, 7, 4)
                calories_quotidiennes = st.number_input("Calories quotidiennes", 1000, 6000, 2200)
                proteines_g_par_jour = st.number_input("Proteines (g/jour)", 40, 400, 160)
                heures_sommeil = st.slider("Heures de sommeil", 3.0, 12.0, 7.0, step=0.5)
                semaines_suivi_prevues = st.slider("Semaines de suivi prevues", 1, 52, 12)
                adherence_programme_pct = st.slider("Adherence estimee (%)", 0, 100, 75)

            consentement = st.checkbox(
                "Le client a ete informe de la collecte de ses donnees et y consent",
                help="A cocher lorsque le client a ete informe (finalite, duree de conservation, "
                     "droits RGPD) - voir docs/RGPD_AI_ACT.md. Horodate automatiquement.",
            )

            submitted_ajout = st.form_submit_button("Ajouter ce client")

        if submitted_ajout:
            if not prenom or not nom:
                st.error("Prenom et nom sont obligatoires.")
            elif not consentement:
                st.error("Le consentement du client doit etre recueilli avant d'enregistrer sa fiche.")
            else:
                new_id = db.add_client({
                    "prenom": prenom, "nom": nom, "age": age, "sexe": sexe, "taille_cm": taille_cm,
                    "poids_initial_kg": poids_initial_kg, "poids_cible_kg": poids_cible_kg,
                    "objectif": objectif, "niveau": niveau,
                    "frequence_entrainement_semaine": frequence_entrainement_semaine,
                    "calories_quotidiennes": calories_quotidiennes,
                    "proteines_g_par_jour": proteines_g_par_jour, "heures_sommeil": heures_sommeil,
                    "semaines_suivi_prevues": semaines_suivi_prevues,
                    "adherence_programme_pct": adherence_programme_pct,
                }, consentement_recueilli=True)
                refresh_after_write(f"Client ajoute : {new_id}.")

    with tab_suivi:
        clients_df = cached_clients()
        actifs_df = clients_df[clients_df["actif"] == 1] if not clients_df.empty else clients_df
        if actifs_df.empty:
            st.write("Aucun client actif. Ajoutez-en un dans l'onglet \"Ajouter un client\".")
        else:
            selected_id_suivi = st.selectbox(
                "Client", actifs_df["client_id"],
                format_func=lambda cid: f"{cid} - " + actifs_df.set_index('client_id').loc[cid, 'prenom'] + " " + actifs_df.set_index('client_id').loc[cid, 'nom'],
                key="select_suivi",
            )
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                poids_semaine = st.number_input("Poids releve cette semaine (kg)", 30.0, 250.0, 75.0)
            with col_b:
                tour_taille_semaine = st.number_input(
                    "Tour de taille (cm, optionnel)", 0.0, 200.0, 0.0,
                    help="Laisser a 0 si non mesure cette semaine.",
                )
            with col_c:
                energie_semaine = st.select_slider(
                    "Ressenti / energie", options=[1, 2, 3, 4, 5], value=3,
                    help="1 = epuise, 5 = en pleine forme",
                )
            note_semaine = st.text_input("Note (optionnel)", "")
            if st.button("Enregistrer la pesee"):
                db.add_weigh_in(
                    selected_id_suivi, poids_semaine, note_semaine,
                    energie=energie_semaine,
                    tour_taille_cm=tour_taille_semaine if tour_taille_semaine > 0 else None,
                )
                refresh_after_write("Pesee enregistree.")

            st.subheader("Historique")
            st.dataframe(db.get_weigh_ins(selected_id_suivi), use_container_width=True)

    with tab_export:
        st.write(
            "Vos vraies donnees vivent uniquement dans la base hebergee (Supabase). "
            "Exportez-les regulierement comme filet de securite independant de l'hebergeur."
        )
        export_data = db.export_all_data()
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "Telecharger mes clients (CSV)",
                data=export_data["clients"].to_csv(index=False).encode("utf-8"),
                file_name=f"clients_{date.today().isoformat()}.csv", mime="text/csv",
            )
        with col_exp2:
            st.download_button(
                "Telecharger l'historique des suivis (CSV)",
                data=export_data["suivis_hebdo"].to_csv(index=False).encode("utf-8"),
                file_name=f"suivis_hebdo_{date.today().isoformat()}.csv", mime="text/csv",
            )
        st.caption(f"{len(export_data['clients'])} client(s), {len(export_data['suivis_hebdo'])} pesee(s) au total.")

# ----------------------------------------------------------------------------
# PREDICTION EN TEMPS REEL
# ----------------------------------------------------------------------------
elif page == "Prediction en temps reel":
    render_header("Prediction en temps reel", "Simuler un profil sans creer de client", tag="MODELE IA")
    st.info(
        "**Transparence (AI Act, art. 50)** : ce resultat est une estimation statistique "
        "produite par un systeme d'IA, pas une decision automatique. Vous restez seul "
        "decisionnaire de l'accompagnement propose (RGPD, art. 22 - pas de decision "
        "entierement automatisee)."
    )

    model, scaler, encoders = load_model_artifacts()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age du client", 18, 65, 30, help="Age en annees")
            sexe = st.selectbox("Sexe", ["H", "F"], help="Sexe biologique du client")
            taille_cm = st.slider("Taille (cm)", 148, 202, 175)
            poids_initial_kg = st.slider("Poids initial (kg)", 45.0, 160.0, 80.0)
        with col2:
            poids_cible_kg = st.slider("Poids cible (kg)", 40.0, 170.0, 75.0)
            objectif = st.selectbox("Objectif", ["seche", "prise_masse", "recomposition"], help="Objectif principal du client")
            niveau = st.selectbox("Niveau", ["debutant", "intermediaire", "avance"], help="Niveau d'experience en musculation")
            frequence_entrainement_semaine = st.slider("Frequence d'entrainement (seances/semaine)", 1, 6, 4)
        with col3:
            calories_quotidiennes = st.slider("Calories quotidiennes", 1200, 4500, 2200, step=50)
            proteines_g_par_jour = st.slider("Proteines (g/jour)", 60, 320, 160)
            heures_sommeil = st.slider("Heures de sommeil / nuit", 3.5, 10.0, 7.0, step=0.5)
            semaines_suivi_prevues = st.slider("Semaines de suivi prevues", 4, 24, 12)
            adherence_programme_pct = st.slider("Adherence au programme (%)", 10, 100, 75, help="Pourcentage estime de respect du programme")

        submitted = st.form_submit_button("Predire")

    if submitted:
        profile = {
            "age": age, "sexe": sexe, "taille_cm": taille_cm,
            "poids_initial_kg": poids_initial_kg, "poids_cible_kg": poids_cible_kg,
            "objectif": objectif, "niveau": niveau,
            "frequence_entrainement_semaine": frequence_entrainement_semaine,
            "calories_quotidiennes": calories_quotidiennes,
            "proteines_g_par_jour": proteines_g_par_jour,
            "heures_sommeil": heures_sommeil,
            "semaines_suivi_prevues": semaines_suivi_prevues,
            "adherence_programme_pct": adherence_programme_pct,
        }
        X_row = build_feature_row(profile, encoders, scaler)
        proba = float(model.predict_proba(X_row)[0, 1])

        col_a, col_b = st.columns([1, 1])
        with col_a:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Probabilite d'atteinte de l'objectif (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#6E11F4"},
                    "steps": [
                        {"range": [0, 40], "color": "#f8b4b4"},
                        {"range": [40, 70], "color": "#ffe08a"},
                        {"range": [70, 100], "color": "#b7f0c1"},
                    ],
                },
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_b:
            if proba > 0.70:
                st.success(f"**Profil favorable** — probabilite estimee : {proba:.0%}")
            elif proba >= 0.40:
                st.warning(f"**Profil a risque, ajuster le programme** — probabilite estimee : {proba:.0%}")
            else:
                st.error(f"**Profil critique, revoir les bases** — probabilite estimee : {proba:.0%}")

            st.subheader("Features les plus influentes pour cette prediction")
            importance = local_feature_importance(model, X_row)
            top3 = importance.head(3)
            for feat, val in top3.items():
                sens = "favorise l'atteinte de l'objectif" if val > 0 else "penalise l'atteinte de l'objectif"
                st.write(f"- `{feat}` : contribution {val:+.3f} ({sens})")

# ----------------------------------------------------------------------------
# PRESENTATION DU PROJET
# ----------------------------------------------------------------------------
elif page == "Presentation du projet":
    st.title("Prediction de progression - Coaching fitness @builtbyarthur")

    st.header("Contexte")
    st.write(
        "Cet outil a ete developpe pour le coaching fitness en ligne @builtbyarthur. "
        "Il permet d'estimer, a partir du profil et des habitudes d'un client, la "
        "probabilite qu'il atteigne son objectif physique (seche, prise de masse ou "
        "recomposition corporelle)."
    )

    st.header("Problematique")
    st.write(
        "Comment predire si un client atteindra son objectif afin d'adapter "
        "l'accompagnement propose a distance, avant que le client ne decroche ?"
    )

    st.header("Architecture du projet")
    if ARCHITECTURE_PNG.exists():
        st.image(str(ARCHITECTURE_PNG), caption="Flux complet du pipeline IA, de la donnee brute au coach.", width=500)
    else:
        st.info("Schema d'architecture non trouve. Executer src/architecture_diagram.py")

    st.header("Stack technique")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Donnees & ML**")
        st.markdown("- pandas / numpy\n- scikit-learn\n- XGBoost\n- joblib")
    with col2:
        st.markdown("**Visualisation**")
        st.markdown("- Plotly\n- Matplotlib")
    with col3:
        st.markdown("**Application**")
        st.markdown("- Streamlit\n- Deploiement Streamlit Cloud")

    st.header("Conformite RGPD & AI Act")
    st.write(
        "Les vrais clients (espace \"Mes clients\") sont stockes dans une base de "
        "donnees hebergee dediee, dont la connexion est protegee par un secret "
        "(jamais commite) et dont l'acces a l'application est restreint - separee du "
        "dataset synthetique public utilise pour l'entrainement initial. Le systeme est "
        "classe a risque minimal au sens de l'AI Act (pas de decision automatisee, "
        "supervision humaine du coach maintenue, y compris lors du reentrainement)."
    )
    if RGPD_AI_ACT_DOC.exists():
        with st.expander("Voir l'analyse complete RGPD & AI Act"):
            st.markdown(RGPD_AI_ACT_DOC.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------
# PIPELINE ETL
# ----------------------------------------------------------------------------
elif page == "Pipeline ETL":
    st.title("Pipeline ETL")

    raw_df = load_raw()
    processed_df = load_processed()

    st.header("Donnees brutes vs donnees traitees")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Donnees brutes")
        st.dataframe(raw_df.head(15), use_container_width=True)
    with col2:
        st.subheader("Donnees traitees (features engineered)")
        st.dataframe(processed_df.head(15), use_container_width=True)

    st.header("Statistiques descriptives")
    st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)

    st.header("Distribution des features")
    numeric_choice = st.selectbox("Choisir une variable numerique", NUM_COLS[:10], index=0)
    fig_hist = px.histogram(
        processed_df, x=numeric_choice, color=TARGET_COL,
        barmode="overlay", title=f"Distribution de {numeric_choice} selon l'atteinte d'objectif",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"Histogramme de la variable {numeric_choice}, colore selon la variable cible (objectif_atteint).")

    st.header("Heatmap de correlation")
    corr_cols = [c for c in NUM_COLS if c in processed_df.columns] + [TARGET_COL]
    corr = processed_df[corr_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Matrice de correlation des variables numeriques",
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Correlation de Pearson entre chaque variable numerique et la variable cible.")

    st.header("Equilibre des classes (variable cible)")
    target_counts = raw_df[TARGET_COL].value_counts(normalize=True).rename({0: "Non atteint", 1: "Atteint"})
    fig_target = px.pie(
        values=target_counts.values, names=target_counts.index,
        title="Repartition de la variable cible objectif_atteint",
    )
    st.plotly_chart(fig_target, use_container_width=True)
    st.caption(f"{target_counts.get('Atteint', 0):.0%} des clients atteignent leur objectif dans ce dataset.")

    st.header("Transformations appliquees")
    st.markdown(
        """
- **Extraction** : generation de 600 profils clients + inventaire (types, valeurs manquantes, doublons) - aucune valeur manquante ni doublon detecte.
- **Feature engineering** :
    - `imc` = poids / taille² (indicateur de corpulence)
    - `ratio_proteines_poids` = apport proteique rapporte au poids (g/kg)
    - `besoin_calorique_estime` = formule de Mifflin-St Jeor x facteur d'activite
    - `deficit_calorique` = besoin calorique estime - calories quotidiennes
    - `score_mode_de_vie` = combinaison ponderee du sommeil et de l'adherence au programme
- **Encodage** : `LabelEncoder` sur les variables categorielles (sexe, objectif, niveau)
- **Normalisation** : `StandardScaler` sur l'ensemble des variables numeriques (moyenne 0, ecart-type 1), sauvegarde dans `scaler.pkl` pour etre reutilise a la prediction.
        """
    )

# ----------------------------------------------------------------------------
# COMPARAISON DES MODELES
# ----------------------------------------------------------------------------
elif page == "Comparaison des modeles":
    st.title("Comparaison des modeles")

    results = load_results()
    ranking = pd.DataFrame(results["ranking"])
    full_results = results["full_results"]
    winner_key = results["winner"]

    st.header("Tableau comparatif")
    display_cols = ["label", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted",
                     "auc_roc", "cv_mean", "cv_std", "training_time_sec", "composite_score"]
    styled = ranking[display_cols].style.highlight_max(
        subset=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "auc_roc", "composite_score"],
        color="#F3EBFF",
    ).highlight_min(subset=["training_time_sec"], color="#F3EBFF").format(precision=3)
    st.dataframe(styled, use_container_width=True)
    st.caption("Surbrillance = meilleure valeur sur chaque metrique (temps d'entrainement : le plus bas est le meilleur).")

    st.header("Matrices de confusion")
    cols = st.columns(4)
    for col, (key, res) in zip(cols, full_results.items()):
        with col:
            cm = np.array(res["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predit", y="Reel"), x=["Non atteint", "Atteint"], y=["Non atteint", "Atteint"],
                title=res["label"],
            )
            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.header("Courbes ROC superposees")
    fig_roc = go.Figure()
    for key, res in full_results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{res['label']} (AUC={res['auc_roc']:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard", line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                           title="Courbes ROC des 4 modeles")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("Plus une courbe est proche du coin superieur gauche, meilleure est la capacite de discrimination du modele.")

    st.header(f"Learning curves - {full_results[winner_key]['label']} (modele retenu)")
    lc = full_results[winner_key]["learning_curve"]
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(x=lc["train_sizes"], y=lc["train_scores_mean"], mode="lines+markers", name="Score entrainement"))
    fig_lc.add_trace(go.Scatter(x=lc["train_sizes"], y=lc["val_scores_mean"], mode="lines+markers", name="Score validation"))
    fig_lc.update_layout(xaxis_title="Taille du jeu d'entrainement", yaxis_title="AUC-ROC (CV 5-fold)")
    st.plotly_chart(fig_lc, use_container_width=True)
    st.caption("Convergence des scores entrainement/validation : un ecart faible indique une bonne generalisation (pas de surapprentissage majeur).")

    st.header("Justification de la selection automatique")
    winner_row = ranking[ranking["key"] == winner_key].iloc[0]
    st.success(
        f"Modele retenu : **{winner_row['label']}** — score composite = **{winner_row['composite_score']:.4f}**\n\n"
        f"score = 0.4 x AUC({winner_row['auc_roc']:.3f}) + 0.3 x F1({winner_row['f1_weighted']:.3f}) "
        f"+ 0.2 x Accuracy({winner_row['accuracy']:.3f}) + 0.1 x score_vitesse"
    )
    if SELECTION_REPORT.exists():
        with st.expander("Voir le rapport de selection complet"):
            st.markdown(SELECTION_REPORT.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------
# DEMO DASHBOARD (DONNEES SIMULEES)
# ----------------------------------------------------------------------------
elif page == "Demo dashboard (donnees simulees)":
    st.title("Demo dashboard (donnees simulees)")
    st.caption(
        "Simulation a des fins de demonstration/certification, basee sur le dataset "
        "synthetique. Pour vos vrais clients, voir l'espace coach > \"Accueil\" / \"Mes clients\"."
    )
    st.write("Simulation du suivi de 10 clients sur 12 semaines (poids reel vs objectif).")

    raw_df = load_raw()

    @st.cache_data
    def simulate_tracking(seed: int = 7, n_clients: int = 10, n_weeks: int = 12):
        rng = np.random.default_rng(seed)
        sample = raw_df.sample(n=n_clients, random_state=seed).reset_index(drop=True)
        rows = []
        for _, client in sample.iterrows():
            poids_debut = client["poids_initial_kg"]
            poids_cible = client["poids_cible_kg"]
            adherence = client["adherence_programme_pct"] / 100
            total_delta = poids_cible - poids_debut
            poids_actuel = poids_debut
            for week in range(n_weeks + 1):
                progress_ratio = (week / n_weeks) * adherence
                bruit = rng.normal(0, 0.35)
                poids_actuel = poids_debut + total_delta * progress_ratio + bruit
                rows.append({
                    "client_id": client["client_id"], "semaine": week,
                    "poids": round(poids_actuel, 1), "poids_cible": poids_cible,
                    "poids_initial": poids_debut, "objectif": client["objectif"],
                })
        return pd.DataFrame(rows)

    tracking_df = simulate_tracking()

    st.header("Progression par client")
    fig_track = px.line(
        tracking_df, x="semaine", y="poids", color="client_id", markers=True,
        title="Evolution du poids sur 12 semaines (simulation)",
    )
    for client_id in tracking_df["client_id"].unique():
        cible = tracking_df[tracking_df["client_id"] == client_id]["poids_cible"].iloc[0]
        fig_track.add_hline(y=cible, line_dash="dot", opacity=0.15)
    st.plotly_chart(fig_track, use_container_width=True)
    st.caption("Chaque courbe represente la trajectoire de poids simulee d'un client vers son objectif.")

    st.header("KPIs globaux (simulation)")
    summary_rows = []
    for client_id, grp in tracking_df.groupby("client_id"):
        debut = grp["poids_initial"].iloc[0]
        cible = grp["poids_cible"].iloc[0]
        actuel = grp[grp["semaine"] == grp["semaine"].max()]["poids"].iloc[0]
        denom = (cible - debut) if abs(cible - debut) > 1e-6 else 1e-6
        progress_pct = np.clip((actuel - debut) / denom, -0.5, 1.5)
        if progress_pct >= 0.95:
            statut = "Objectif atteint"
        elif progress_pct >= 0.5:
            statut = "En bonne voie"
        else:
            statut = "A risque"
        summary_rows.append({
            "client_id": client_id, "poids_initial": debut, "poids_actuel": round(actuel, 1),
            "poids_cible": cible, "progression": f"{progress_pct:.0%}", "statut": statut,
        })
    summary_df = pd.DataFrame(summary_rows)

    taux_reussite = (summary_df["statut"] == "Objectif atteint").mean()
    plus_avance = summary_df.loc[summary_df["progression"].str.rstrip("%").astype(float).idxmax(), "client_id"]
    plus_en_retard = summary_df.loc[summary_df["progression"].str.rstrip("%").astype(float).idxmin(), "client_id"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Taux de reussite global", f"{taux_reussite:.0%}")
    col2.metric("Client le plus en avance", plus_avance)
    col3.metric("Client le plus en retard", plus_en_retard)

    st.header("Recapitulatif par client")
    st.dataframe(summary_df, use_container_width=True)

# ----------------------------------------------------------------------------
# REENTRAINEMENT (AVANCE)
# ----------------------------------------------------------------------------
elif page == "Reentrainement (avance)":
    st.title("Reentrainement du modele")
    st.caption(
        "Declenchement manuel volontaire (pas de tache planifiee automatique) : au rythme "
        "actuel d'arrivee de clients reels labellises, une automatisation ajouterait de la "
        "complexite (planification, gestion d'erreurs) sans benefice reel. A activer quand "
        "le volume de clients rendra le clic manuel penible - voir Gestion de projet."
    )
    st.write(
        f"Le modele peut etre reentraine en integrant les clients reels dont l'issue "
        f"est connue (objectif atteint ou non). Un minimum de {MIN_REAL_CLIENTS} clients "
        f"labellises est requis pour eviter un reentrainement instable."
    )
    labelled = db.get_labelled_clients()
    st.metric("Clients reels labellises disponibles", len(labelled))

    if st.button("Lancer le reentrainement manuellement"):
        with st.spinner("Reentrainement en cours (GridSearchCV sur 4 modeles)..."):
            result = run_retrain()

        if result["status"] == "skipped":
            st.warning(result["reason"])
        elif result["status"] == "promoted":
            st.success(
                f"Nouveau modele promu en production : **{result['new_winner_label']}**. "
                f"Score composite : {result['old_score']:.4f} -> {result['new_score']:.4f} "
                f"({result['n_real_clients']} clients reels integres)."
            )
        else:
            st.error(
                f"Reentrainement rejete : le nouveau score ({result['new_score']:.4f}) est "
                f"inferieur a l'actuel ({result['old_score']:.4f}). Le modele en production "
                "n'a pas ete modifie. Continuez a collecter des donnees reelles."
            )

    if SELECTION_REPORT.exists():
        with st.expander("Voir le dernier rapport de selection"):
            st.markdown(SELECTION_REPORT.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------
# GESTION DE PROJET
# ----------------------------------------------------------------------------
elif page == "Gestion de projet":
    st.title("Gestion de projet")

    st.header("Suivi des taches")
    tasks = pd.DataFrame([
        ["Definition du CDC et des objectifs mesurables", "1", "Termine", 0.5, 0.5],
        ["Generation du dataset synthetique (600 clients)", "1", "Termine", 1.0, 1.2],
        ["Feature engineering", "1", "Termine", 1.0, 0.8],
        ["Encodage + normalisation + export", "1", "Termine", 0.5, 0.5],
        ["Entrainement Regression logistique + GridSearchCV", "2", "Termine", 0.5, 0.4],
        ["Entrainement Foret aleatoire + GridSearchCV", "2", "Termine", 0.75, 0.6],
        ["Entrainement XGBoost + GridSearchCV", "2", "Termine", 0.75, 0.5],
        ["Entrainement MLP + GridSearchCV", "2", "Termine", 0.75, 0.9],
        ["Selection automatique du meilleur modele", "2", "Termine", 0.5, 0.4],
        ["Schema d'architecture (matplotlib)", "2", "Termine", 0.5, 0.3],
        ["Pages Streamlit (7 pages)", "3", "Termine", 5.25, 5.2],
        ["Documentation (CDC, ARCHITECTURE, README)", "3", "Termine", 1.0, 1.0],
        ["Page 'Mes clients' (CRUD reel, SQLite, PDF, reentrainement)", "4", "Termine", 3.0, 3.2],
        ["Passage a usage pro : Accueil KPIs, reorganisation, theme, CI reentrainement", "5", "Termine", 2.5, 2.4],
    ], columns=["Tache", "Sprint", "Statut", "Temps estime (h)", "Temps reel (h)"])
    tasks["Ecart"] = (tasks["Temps reel (h)"] - tasks["Temps estime (h)"]).round(2)
    st.dataframe(tasks, use_container_width=True)

    st.header("Pilotage des prestataires techniques")

    st.subheader("Modeles : scikit-learn vs TensorFlow/Keras")
    st.dataframe(pd.DataFrame({
        "Critere": ["Facilite d'integration", "Documentation", "Performance", "Temps de developpement", "Courbe d'apprentissage", "Total /25"],
        "scikit-learn": [5, 5, 5, 5, 5, 25],
        "TensorFlow/Keras": [3, 4, 3, 2, 3, 15],
    }), use_container_width=True, hide_index=True)
    st.caption("Decision : scikit-learn - suffisant et plus rapide a developper pour un dataset tabulaire de 600 lignes.")

    st.subheader("Dashboard : Streamlit vs Dash vs Flask")
    st.dataframe(pd.DataFrame({
        "Critere": ["Rapidite de developpement", "Interactivite", "Deploiement", "Maintenance", "Total /20"],
        "Streamlit": [5, 4, 5, 5, 19],
        "Dash": [3, 5, 3, 3, 14],
        "Flask": [2, 2, 2, 2, 8],
    }), use_container_width=True, hide_index=True)
    st.caption("Decision : Streamlit - repond a la contrainte budget zero / deploiement sans DevOps.")

    st.subheader("Boosting : XGBoost vs LightGBM vs CatBoost")
    st.dataframe(pd.DataFrame({
        "Critere": ["Performance", "Vitesse", "Gestion categorielles", "Maturite/doc", "Total /20"],
        "XGBoost": [5, 4, 3, 5, 17],
        "LightGBM": [4, 5, 3, 4, 16],
        "CatBoost": [5, 3, 5, 3, 16],
    }), use_container_width=True, hide_index=True)
    st.caption("Decision : XGBoost - meilleure documentation et integration eprouvee avec scikit-learn.")

    st.subheader("Automatisation du reentrainement : GitHub Actions vs tache planifiee locale vs manuel")
    st.dataframe(pd.DataFrame({
        "Critere": ["Cout", "Techniquement possible (base hebergee accessible)", "Simplicite de mise en place", "Adapte au volume de clients actuel", "Total /20"],
        "GitHub Actions (cloud)": [5, 5, 4, 2, 16],
        "Tache planifiee locale (Task Scheduler)": [5, 5, 2, 2, 14],
        "Declenchement manuel (retenu)": [5, 5, 5, 5, 20],
    }), use_container_width=True, hide_index=True)
    st.caption(
        "Decision : declenchement manuel, malgre le fait que la migration vers une base "
        "hebergee (Supabase) rende desormais GitHub Actions techniquement viable (contrairement "
        "a la version precedente ou coaching.db etait un fichier local inaccessible depuis le "
        "cloud). Au volume actuel de clients reels labellises, automatiser ajouterait une "
        "complexite (planification, gestion d'erreurs, secrets CI) sans benefice reel. "
        "Cette decision est reevaluee a chaque changement de contrainte plutot que figee : "
        "l'automatisation GitHub Actions redevient l'option recommandee des que le volume de "
        "clients rendra le clic manuel penible."
    )

    st.header("Retrospective")
    st.markdown(
        """
**Ce qui a fonctionne :** l'automatisation complete de la selection du modele
a supprime toute subjectivite dans le choix final ; la reutilisation des memes
fonctions de feature engineering entre l'ETL et l'API de prediction a evite
toute divergence entre donnees d'entrainement et donnees de prediction.

**Ce qui aurait pu etre ameliore :** le dataset synthetique reste un proxy ;
le calibrage de la variable cible a necessite plusieurs iterations pour
atteindre l'objectif d'AUC-ROC > 0.75.

**Decisions techniques qui ont evolue :** le score composite de selection a
ete prefere a une simple comparaison d'AUC-ROC, pour eviter de choisir un
modele lent ou instable en validation croisee. Apres un premier usage reel,
l'application a ete reorganisee en deux espaces (coach / documentation
technique) pour correspondre a un usage professionnel quotidien plutot qu'a
une simple demonstration, avec une page d'accueil orientee KPIs actionnables
plutot que des metriques ML brutes.

**Prochaines evolutions envisagees :** integration d'une API MyFitnessPal,
notification automatique si client a risque, version mobile.
        """
    )
