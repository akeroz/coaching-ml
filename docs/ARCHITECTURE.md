# Architecture du pipeline IA

## Vue d'ensemble

```
Donnees brutes (data/raw/clients_raw.csv)
        |
        v
ETL (src/etl.py)
  - Extraction : generation + inventaire (types, NA, doublons)
  - Transformation : feature engineering, encodage, normalisation
  - Chargement : dataset_final.csv, scaler.pkl, label_encoders.pkl
        |
        v
Features engineered (data/processed/dataset_final.csv)
        |
        v
Split train/test stratifie 80/20 (src/train.py)
        |
        v
Entrainement de 4 modeles (GridSearchCV, CV 5-fold)
  - Regression logistique
  - Foret aleatoire
  - XGBoost
  - Reseau de neurones (MLP)
        |
        v
Evaluation automatique (accuracy, F1, precision, rappel, AUC-ROC,
matrice de confusion, temps d'entrainement, learning curves)
        |
        v
Selection du meilleur modele (src/select_model.py)
  - Score composite = 0.4*AUC + 0.3*F1 + 0.2*Accuracy + 0.1*(1/temps normalise)
        |
        v
Sauvegarde (models/best_model.pkl, models/results.json,
docs/MODEL_SELECTION_REPORT.md)
        |
        v
API de prediction (src/predict.py)
  - Recharge best_model.pkl + scaler.pkl + label_encoders.pkl
  - Applique la meme feature engineering a un nouveau profil client
        |
        v
Dashboard Streamlit (app.py)
  - 6 pages : presentation, ETL, comparaison des modeles,
    prediction temps reel, suivi clients, gestion de projet
        |
        v
Utilisateur final (coach @builtbyarthur)
```

## Schema visuel

Voir `docs/architecture_diagram.png` (genere par `src/architecture_diagram.py`).

## Maquette de l'infrastructure de deploiement

Cette maquette decrit l'infrastructure d'hebergement (distincte du pipeline
ML ci-dessus), pour permettre a un tiers de comprendre et reproduire le
deploiement :

```
GitHub (depot public)
  |-- code source (app.py, src/)
  |-- .github/workflows/tests.yml   -> CI : pytest sur chaque push
  |-- .github/workflows/keepalive.yml -> ping Supabase toutes les 3 jours
  |
  v
Streamlit Cloud (hebergement de l'application, gratuit)
  |-- Deploiement automatique a chaque push sur main
  |-- Secrets (DATABASE_URL) injectes via l'interface Streamlit Cloud,
  |   jamais dans le depot
  |-- Acces restreint : "Only specific people can view this app"
  |
  v
Supabase (Postgres hebergee, gratuit)
  |-- Tables clients / suivis_hebdo (donnees reelles)
  |-- Connexion via pooler (aws-*.pooler.supabase.com:6543),
  |   compatible IPv4, chiffree en transit (TLS)
```

Il n'existe pas d'environnement de staging distinct de la production : ce
choix est assume (contrainte budget zero du CDC) plutot que subi - Streamlit
Cloud sert a la fois de bac a sable de demonstration (`DEMO_MODE`) et
d'environnement reel, la separation etant faite au niveau des donnees
(dataset synthetique public vs base clients privee) plutot qu'au niveau de
l'infrastructure.

## Choix techniques

| Composant | Choix | Justification |
|---|---|---|
| Langage | Python 3.10 | Ecosysteme ML mature (scikit-learn, xgboost) |
| Donnees | CSV + pickle (joblib) | Pas de base de donnees necessaire, budget zero, deploiement simple |
| Modeles | scikit-learn + XGBoost | Couvre modele lineaire, ensemble bagging, boosting et reseau de neurones |
| Selection | Score composite automatique | Choix objectif et reproductible, pas d'intervention manuelle |
| Interface | Streamlit | Developpement rapide, pas de frontend a coder, deploiement gratuit sur Streamlit Cloud |
| Visualisation | Plotly + Matplotlib | Plotly pour l'interactivite du dashboard, Matplotlib pour le schema d'architecture statique |

## Flux de donnees pour une prediction en temps reel

1. Le coach saisit le profil du client dans le formulaire (page 4 de l'app).
2. `predict.py` applique `feature_engineering()` (memes formules que l'ETL) puis
   `transform_features()` avec les encodeurs/scaler sauvegardes (pas de re-entrainement).
3. Le modele `best_model.pkl` retourne une probabilite d'atteinte d'objectif.
4. L'application traduit cette probabilite en recommandation
   (profil favorable / a risque / critique).
