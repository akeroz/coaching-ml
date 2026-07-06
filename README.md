# Coaching ML - Prediction de progression fitness

Outil de machine learning pour le coaching fitness en ligne **@builtbyarthur**,
permettant d'estimer la probabilite qu'un client atteigne son objectif
(seche, prise de masse, recomposition) a partir de son profil et de ses
habitudes, afin d'adapter l'accompagnement propose a distance.

**Demo en ligne (dataset synthetique)** : https://coaching-ml-dahro8p5dpihporqh7uhjm.streamlit.app/

## Architecture technique

```
Donnees brutes -> ETL -> Features engineered -> Split train/test
   -> Entrainement 4 modeles -> Evaluation automatique
   -> Selection du meilleur modele -> Sauvegarde
   -> API de prediction -> Dashboard Streamlit -> Coach
```

Voir `docs/ARCHITECTURE.md` et `docs/architecture_diagram.png` pour le detail.

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Lancement

```bash
# 1. Generer les donnees et le dataset final
python src/etl.py

# 2. Entrainer les 4 modeles
python src/train.py

# 3. Selectionner automatiquement le meilleur modele
python src/select_model.py

# 4. (optionnel) Regenerer le schema d'architecture
python src/architecture_diagram.py

# 5. Lancer le dashboard
streamlit run app.py
```

Une fois l'application lancee, la page **"Mes clients (usage reel)"** permet
d'ajouter de vrais clients (stockes localement dans `data/coaching.db`,
jamais commite - voir `docs/RGPD_AI_ACT.md`), de suivre leur poids semaine
apres semaine, d'exporter un resume PDF, et de relancer un reentrainement du
modele integrant ces clients reels des qu'il y en a suffisamment (garde
anti-regression automatique, voir `src/retrain_with_real_data.py`).

## Structure du projet

```
coaching-ml/
├── data/
│   ├── raw/                  # Donnees brutes generees (clients_raw.csv)
│   └── processed/             # Dataset final + scaler + encoders
├── src/
│   ├── etl.py                    # Extraction / Transformation / Chargement
│   ├── train.py                   # Entrainement des 4 modeles + GridSearchCV
│   ├── evaluate.py                 # (metriques integrees dans train.py)
│   ├── select_model.py             # Selection automatique du meilleur modele
│   ├── predict.py                  # API de prediction (recharge best_model.pkl)
│   ├── architecture_diagram.py     # Schema d'architecture (matplotlib)
│   ├── db.py                       # Persistance SQLite des vrais clients (prive)
│   ├── report.py                   # Export PDF par client
│   └── retrain_with_real_data.py   # Reentrainement + garde anti-regression
├── models/                      # Modeles entraines, best_model.pkl, results.json
├── data/coaching.db              # Vrais clients (SQLite, jamais commite)
├── docs/
│   ├── CDC.md
│   ├── ARCHITECTURE.md
│   ├── PROJECT_MANAGEMENT.md
│   ├── MODEL_SELECTION_REPORT.md
│   ├── RGPD_AI_ACT.md
│   └── architecture_diagram.png
├── app.py                       # Dashboard Streamlit (7 pages)
├── requirements.txt
└── README.md
```

## Resultats obtenus

4 modeles ont ete compares automatiquement (score composite = 0.4×AUC + 0.3×F1
+ 0.2×Accuracy + 0.1×score de vitesse) :

| Modele | Accuracy | F1 (weighted) | AUC-ROC | Temps (s) | Score composite |
|---|---|---|---|---|---|
| **Regression logistique** (retenu) | 0.767 | 0.755 | 0.775 | 4.7 | 0.789 |
| XGBoost | 0.742 | 0.728 | 0.740 | 4.4 | 0.763 |
| Foret aleatoire | 0.733 | 0.711 | 0.753 | 21.0 | 0.707 |
| Reseau de neurones (MLP) | 0.700 | 0.700 | 0.705 | 35.2 | 0.632 |

Le modele retenu (Regression logistique) depasse l'objectif du cahier des
charges (AUC-ROC > 0.75). Detail complet dans `docs/MODEL_SELECTION_REPORT.md`.

## Pistes d'evolution

- Integration d'une API MyFitnessPal pour recuperer les calories/macros reelles.
- Notification automatique si un client bascule en "profil a risque".
- Remplacement progressif du dataset synthetique par des donnees reelles
  anonymisees issues du suivi client.
- Version mobile / PWA.

## Licence

MIT
