"""Dashboard Streamlit - Prediction de progression coaching fitness (@builtbyarthur)."""

import json
import sys
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

from etl import CAT_COLS, FEATURE_COLUMNS, NUM_COLS, TARGET_COL  # noqa: E402
from predict import build_feature_row, load_artifacts  # noqa: E402

st.set_page_config(page_title="Coaching ML - builtbyarthur", page_icon="💪", layout="wide")

RAW_PATH = ROOT_DIR / "data" / "raw" / "clients_raw.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "dataset_final.csv"
RESULTS_PATH = ROOT_DIR / "models" / "results.json"
ARCHITECTURE_PNG = ROOT_DIR / "docs" / "architecture_diagram.png"
SELECTION_REPORT = ROOT_DIR / "docs" / "MODEL_SELECTION_REPORT.md"


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


def local_feature_importance(model, X_row: pd.DataFrame) -> pd.Series:
    """Contribution de chaque feature a la prediction (approche generique multi-modeles)."""
    if hasattr(model, "coef_"):
        contrib = model.coef_[0] * X_row.iloc[0].to_numpy()
        return pd.Series(contrib, index=X_row.columns).sort_values(key=abs, ascending=False)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=X_row.columns).sort_values(ascending=False)
    return pd.Series(np.zeros(len(X_row.columns)), index=X_row.columns)


PAGES = [
    "1. Presentation du projet",
    "2. Pipeline ETL",
    "3. Comparaison des modeles",
    "4. Prediction en temps reel",
    "5. Dashboard de suivi clients",
    "6. Gestion de projet",
]

page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Coaching ML - @builtbyarthur")

# ----------------------------------------------------------------------------
# PAGE 1 - PRESENTATION
# ----------------------------------------------------------------------------
if page == PAGES[0]:
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

# ----------------------------------------------------------------------------
# PAGE 2 - PIPELINE ETL
# ----------------------------------------------------------------------------
elif page == PAGES[1]:
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
# PAGE 3 - COMPARAISON DES MODELES
# ----------------------------------------------------------------------------
elif page == PAGES[2]:
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
        color="#b7f0c1",
    ).highlight_min(subset=["training_time_sec"], color="#b7f0c1").format(precision=3)
    st.dataframe(styled, use_container_width=True)
    st.caption("Vert = meilleure valeur sur chaque metrique (temps d'entrainement : le plus bas est le meilleur).")

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
# PAGE 4 - PREDICTION EN TEMPS REEL
# ----------------------------------------------------------------------------
elif page == PAGES[3]:
    st.title("Prediction en temps reel")
    st.write("Renseignez le profil d'un client pour estimer sa probabilite d'atteindre son objectif.")

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
                    "bar": {"color": "#333333"},
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
# PAGE 5 - DASHBOARD DE SUIVI CLIENTS
# ----------------------------------------------------------------------------
elif page == PAGES[4]:
    st.title("Dashboard de suivi clients")
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

    st.header("KPIs globaux")
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
# PAGE 6 - GESTION DE PROJET
# ----------------------------------------------------------------------------
elif page == PAGES[5]:
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
        ["Pages Streamlit (6 pages)", "3", "Termine", 5.25, 5.2],
        ["Documentation (CDC, ARCHITECTURE, README)", "3", "Termine", 1.0, 1.0],
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

    st.header("Retrospective")
    st.markdown(
        """
**Ce qui a fonctionne :** l'automatisation complete de la selection du modele
a supprime toute subjectivite dans le choix final ; la reutilisation des memes
fonctions de feature engineering entre l'ETL et l'API de prediction a evite
toute divergence entre donnees d'entrainement et donnees de prediction.

**Ce qui aurait pu etre ameliore :** le dataset synthetique reste un proxy ;
un futur cycle devra integrer des donnees reelles anonymisees issues du suivi
client. Le calibrage de la variable cible a necessite plusieurs iterations
pour atteindre l'objectif d'AUC-ROC > 0.75.

**Decisions techniques qui ont evolue :** le score composite de selection a
ete prefere a une simple comparaison d'AUC-ROC, pour eviter de choisir un
modele lent ou instable en validation croisee.

**Prochaines evolutions envisagees :** integration d'une API MyFitnessPal,
notification automatique si client a risque, version mobile.
        """
    )
