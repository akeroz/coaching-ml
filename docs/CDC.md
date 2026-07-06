# Cahier des charges

## Contexte et besoin

En tant que coach fitness independant (@builtbyarthur), j'accompagne a distance des
clients ayant des objectifs physiques varies (seche, prise de masse, recomposition
corporelle). Le suivi se fait aujourd'hui manuellement (echanges Instagram, feuilles
de suivi). Je dois pouvoir estimer, a partir du profil et des habitudes d'un client,
la probabilite qu'il atteigne son objectif, afin d'ajuster proactivement
l'accompagnement (frequence des check-ins, ajustement du programme nutritionnel
ou d'entrainement) avant que le client ne decroche.

## Objectifs mesurables

- Modele de prediction avec un **AUC-ROC > 0.75** sur le jeu de test.
- Temps de reponse de l'application **< 5 secondes** entre la saisie du profil et
  l'affichage de la prediction.
- Dashboard **lisible sur mobile** (le coach pilote son activite depuis son telephone).
- Selection du modele **automatisee et reproductible** (score composite documente).

## Perimetre fonctionnel

Inclus :
- Generation/import et traitement des donnees clients (ETL).
- Entrainement et comparaison de 4 modeles de classification.
- Selection automatique du meilleur modele.
- Application de prediction en temps reel pour un nouveau profil client.
- Dashboard de suivi de la progression de clients existants.
- Documentation projet (cadrage, architecture, gestion de projet).

Hors perimetre (v1) :
- Connexion a une base de donnees externe ou API tierce (MyFitnessPal, Instagram API).
- Authentification multi-utilisateurs.
- Envoi de notifications automatiques.

## Perimetre technique

- Python (pandas, scikit-learn, XGBoost, joblib) pour le pipeline ML.
- Streamlit pour l'interface utilisateur.
- Plotly / Matplotlib pour la data visualisation.
- Stockage local (CSV, pickle) - pas de base de donnees.

## Contraintes

- **Budget zero** : uniquement des librairies open source.
- **Deployable sur Streamlit Cloud** (ou equivalent gratuit), sans infrastructure dediee.
- **Maintenable sans competences DevOps** : pas de conteneurisation, pas de CI/CD complexe,
  un seul point d'entree (`streamlit run app.py`).
- Respect des principes RGPD : aucune donnee directement identifiante (voir `src/etl.py`
  pour la note de finalite et de pseudonymisation).
- Accessibilite WCAG AA (voir `PROJECT_MANAGEMENT.md`).

## Livrables et criteres d'acceptation

| Livrable | Critere d'acceptation |
|---|---|
| `data/processed/dataset_final.csv` | Dataset de 600 clients, sans valeur manquante ni doublon, features engineered presentes |
| `models/best_model.pkl` | Modele selectionne automatiquement, AUC-ROC > 0.75 sur le test set |
| `docs/MODEL_SELECTION_REPORT.md` | Comparatif des 4 modeles et justification du choix |
| `app.py` | Application Streamlit fonctionnelle avec 6 pages, temps de reponse < 5s |
| Documentation (`CDC.md`, `ARCHITECTURE.md`, `PROJECT_MANAGEMENT.md`, `README.md`) | Complete et a jour avec le code livre |

## Conditions go/no-go

- **Go** si : AUC-ROC du meilleur modele > 0.75 ET l'application demarre sans erreur
  ET les 6 pages du dashboard sont fonctionnelles.
- **No-go** si : AUC-ROC <= 0.75 (retour au feature engineering / donnees) OU
  l'application plante au demarrage OU une page critique (prediction, comparaison
  des modeles) est non fonctionnelle.
