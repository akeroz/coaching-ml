# Gestion de projet

## CDC simplifie

Voir `docs/CDC.md` pour le detail. Resume : construire un pipeline IA complet
(ETL -> entrainement -> selection automatique -> API -> dashboard) permettant
a un coach fitness de predire si un client atteindra son objectif, avec un
AUC-ROC > 0.75, budget zero, deployable sur Streamlit Cloud.

## Organisation agile - 3 sprints

- **Sprint 1 (matin)** : ETL, generation dataset, feature engineering
  -> Livrable : `data/processed/dataset_final.csv`
- **Sprint 2 (apres-midi)** : entrainement des 4 modeles, evaluation, selection automatique
  -> Livrable : `models/best_model.pkl` + `docs/MODEL_SELECTION_REPORT.md`
- **Sprint 3 (soir)** : application Streamlit complete
  -> Livrable : `app.py` fonctionnelle (6 pages)

## Tableau de bord de suivi projet

| Tache | Sprint | Statut | Temps estime (h) | Temps reel (h) | Ecart | Commentaire |
|---|---|---|---|---|---|---|
| Definition du CDC et des objectifs mesurables | 1 | Termine | 0.5 | 0.5 | 0 | Objectif AUC>0.75 fixe des le depart |
| Generation du dataset synthetique (600 clients) | 1 | Termine | 1.0 | 1.2 | +0.2 | Calibrage de la logique de generation de la cible (equilibre des classes) |
| Feature engineering (IMC, ratio proteines, deficit calorique, score mode de vie) | 1 | Termine | 1.0 | 0.8 | -0.2 | RAS |
| Encodage + normalisation + export dataset_final.csv | 1 | Termine | 0.5 | 0.5 | 0 | LabelEncoder + StandardScaler sauvegardes pour reutilisation en prediction |
| Entrainement Regression logistique + GridSearchCV | 2 | Termine | 0.5 | 0.4 | -0.1 | RAS |
| Entrainement Foret aleatoire + GridSearchCV | 2 | Termine | 0.75 | 0.6 | -0.15 | RAS |
| Entrainement XGBoost + GridSearchCV | 2 | Termine | 0.75 | 0.5 | -0.25 | Entrainement rapide sur ce volume de donnees |
| Entrainement MLP + GridSearchCV | 2 | Termine | 0.75 | 0.9 | +0.15 | Convergence plus lente, augmentation de max_iter |
| Selection automatique du meilleur modele | 2 | Termine | 0.5 | 0.4 | -0.1 | Score composite documente dans le rapport |
| Schema d'architecture (matplotlib) | 2 | Termine | 0.5 | 0.3 | -0.2 | RAS |
| Page 1 - Presentation | 3 | Termine | 0.5 | 0.4 | -0.1 | RAS |
| Page 2 - Pipeline ETL | 3 | Termine | 1.0 | 1.0 | 0 | Heatmap + histogrammes Plotly |
| Page 3 - Comparaison des modeles | 3 | Termine | 1.25 | 1.3 | +0.05 | Courbes ROC superposees, mise en forme conditionnelle |
| Page 4 - Prediction temps reel | 3 | Termine | 1.0 | 1.1 | +0.1 | Jauge de probabilite + feature importance |
| Page 5 - Dashboard de suivi clients | 3 | Termine | 1.0 | 1.0 | 0 | Simulation de 10 clients sur 12 semaines |
| Page 6 - Gestion de projet | 3 | Termine | 0.5 | 0.4 | -0.1 | RAS |
| Documentation (CDC, ARCHITECTURE, README) | 3 | Termine | 1.0 | 1.0 | 0 | RAS |
| **Total** | | | **12.5** | **12.3** | **-0.2** | Projet livre dans les temps |

## Pilotage des prestataires techniques

### Modeles : scikit-learn vs TensorFlow/Keras

| Critere | scikit-learn | TensorFlow/Keras |
|---|---|---|
| Facilite d'integration | 5/5 | 3/5 |
| Documentation | 5/5 | 4/5 |
| Performance (sur ce volume de donnees, ~600 lignes) | 5/5 | 3/5 |
| Temps de developpement | 5/5 | 2/5 |
| Courbe d'apprentissage | 5/5 | 3/5 |
| **Total /25** | **25** | **15** |

**Decision : scikit-learn.** Sur un dataset tabulaire de 600 lignes, un reseau de
neurones profond (Keras) n'apporte aucun gain et complexifie inutilement le
deploiement (poids du framework, dependance GPU optionnelle). Le `MLPClassifier`
de scikit-learn suffit a couvrir la brique "reseau de neurones" du cahier des charges.

### Dashboard : Streamlit vs Dash vs Flask

| Critere | Streamlit | Dash | Flask |
|---|---|---|---|
| Rapidite de developpement | 5/5 | 3/5 | 2/5 |
| Interactivite | 4/5 | 5/5 | 2/5 (necessite JS) |
| Deploiement (Streamlit Cloud gratuit) | 5/5 | 3/5 | 2/5 |
| Maintenance (budget zero, sans DevOps) | 5/5 | 3/5 | 2/5 |
| **Total /20** | **19** | **14** | **8** |

**Decision : Streamlit.** Repond directement a la contrainte "budget zero,
deployable sans DevOps" tout en couvrant les besoins d'interactivite
(formulaires, graphiques Plotly, filtres).

### Boosting : XGBoost vs LightGBM vs CatBoost

| Critere | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Performance (petits jeux de donnees tabulaires) | 5/5 | 4/5 | 5/5 |
| Vitesse d'entrainement | 4/5 | 5/5 | 3/5 |
| Gestion des variables categorielles | 3/5 (encodage manuel requis) | 3/5 | 5/5 (natif) |
| Maturite / documentation / communaute | 5/5 | 4/5 | 3/5 |
| **Total /20** | **17** | **16** | **16** |

**Decision : XGBoost.** Meilleure documentation et communaute la plus large,
integration eprouvee avec scikit-learn (API `XGBClassifier` compatible
`GridSearchCV`). Le gain marginal de CatBoost sur les categorielles n'est pas
determinant ici car le pipeline encode deja les categorielles via LabelEncoder
pour l'ensemble des modeles (coherence du pipeline de features).

## Strategie d'accessibilite (WCAG AA)

- **Contrastes** : palette du dashboard verifiee pour un ratio de contraste
  texte/fond superieur a 4.5:1 (couleurs Streamlit par defaut + verification
  manuelle des couleurs personnalisees utilisees dans les graphiques).
- **Navigation clavier** : l'ensemble des composants Streamlit (sliders,
  selectbox, boutons, navigation sidebar) sont nativement accessibles au clavier
  (Tab / fleches / Entree), sans composant custom en JavaScript brut.
- **Labels explicites** : chaque champ de formulaire (page 4) possede un label
  textuel explicite (pas de placeholder seul en guise de label).
- **Pas de contenu exclusivement visuel** : chaque graphique (courbes ROC,
  matrices de confusion, jauge de probabilite) est accompagne d'un texte
  d'interpretation (ex. "AUC de 0.82" en plus de la courbe, interpretation
  textuelle de la jauge).
- **Police lisible** : taille de police minimale de 14px, respect de la
  typographie par defaut de Streamlit (deja optimisee pour la lisibilite).
- **Pas de CAPTCHA** : l'application est un outil interne au coach, sans
  formulaire d'inscription public.

## Communication et coordination a distance

Le projet sert directement les clients de coaching @builtbyarthur, geres a
100% a distance. L'equipe "distribuee" est ici constituee des clients
eux-memes, geographiquement disperses, animes via des outils numeriques :

- **Check-ins hebdomadaires** via DM Instagram : recueil du poids, des photos
  de progression, du ressenti (sommeil, adherence, difficultes).
- **Google Sheets de suivi partage** : chaque client renseigne ses donnees
  hebdomadaires dans une feuille partagee, servant de source pour alimenter le
  modele de prediction.
- **ManyChat** : automatisation des rappels de check-in et des relances en cas
  de non-reponse, pour fiabiliser la collecte de donnees.
- **Canal dedie par client** (DM Instagram individuel) : partage des
  predictions et recommandations personnalisees issues du modele, en langage
  simple et actionnable (ex. "ton adherence a baisse, on ajuste le programme").
- **Recueil des retours terrain** : les clients signalent les ecarts entre la
  prediction et leur ressenti reel ; ces retours alimentent les iterations
  futures du modele (ajustement des seuils, des features).

## Strategie d'accueil handicap

L'application a ete concue pour rester utilisable par des personnes en
situation de handicap visuel ou moteur :

- **Interface navigable au clavier** : aucune interaction n'est exclusivement
  a la souris (Streamlit gere nativement le focus clavier).
- **Contrastes respectes** : voir section WCAG AA ci-dessus.
- **Police lisible** : taille minimale de 14px sur l'ensemble des pages.
- **Pas de CAPTCHA** : aucune barriere d'authentification complexe.
- **Compatibilite lecteur d'ecran** : les elements Streamlit (titres, labels,
  tableaux `st.dataframe`) exposent une structure HTML semantique lisible par
  les lecteurs d'ecran standards (NVDA, VoiceOver).

## Maintenabilite et procedure de rollback du modele

Le reentrainement (`src/retrain_with_real_data.py`) integre deja une garde
anti-regression au moment T : un nouveau modele n'est promu en production
que si son score composite n'est pas inferieur a l'actuel. Cette garde ne
couvre toutefois pas une degradation qui n'apparaitrait qu'a l'usage (ex.
biais introduit par un petit nombre de clients reels non representatifs).
Procedure de retour en arriere en cas de probleme constate apres coup :

1. **Detecter** : chaque promotion de modele met a jour
   `docs/MODEL_SELECTION_REPORT.md` avec la mention "Reentrainement avec
   donnees reelles" et les scores avant/apres — comparer ce rapport a sa
   version precedente (historique Git) permet de reperer un changement
   suspect.
2. **Isoler** : `models/candidate/` conserve les modeles candidats du dernier
   reentrainement (avant promotion), ce qui permet de comparer directement
   l'ancien et le nouveau modele sans avoir a relancer un entrainement.
3. **Revenir en arriere** : restaurer `models/best_model.pkl`,
   `data/processed/scaler.pkl`, `data/processed/label_encoders.pkl` et
   `data/processed/dataset_final.csv` a partir du commit Git precedent la
   promotion (`git checkout <commit_precedent> -- models/ data/processed/`),
   puis committer ce retour en arriere avec un message explicite.
4. **Documenter** : ajouter une entree dans la retrospective ci-dessous
   expliquant la cause du rollback, pour eviter de reproduire l'erreur lors
   d'un futur reentrainement (ex. exiger un nombre minimal de clients plus
   eleve, ou verifier la representativite du profil des nouveaux clients).

## Retrospective

**Ce qui a fonctionne :**
- L'automatisation complete de la selection du modele (score composite) a
  supprime toute subjectivite dans le choix final.
- La reutilisation des memes fonctions de feature engineering entre l'ETL et
  l'API de prediction (`etl.py` importe par `predict.py`) a evite toute
  divergence entre donnees d'entrainement et donnees de prediction.

**Ce qui aurait pu etre ameliore :**
- Le dataset synthetique, bien que calibre pour etre realiste, reste un proxy :
  un futur cycle devra integrer des donnees reelles anonymisees issues du
  Google Sheet de suivi.
- Le calibrage de la variable cible (equilibre des classes, force du signal)
  a necessite plusieurs iterations pour atteindre l'objectif d'AUC-ROC > 0.75.

**Decisions techniques qui ont evolue :**
- Le score composite de selection a ete prefere a une simple comparaison
  d'AUC-ROC, pour eviter de choisir un modele lent ou instable en validation
  croisee au seul benefice d'un score legerement superieur sur un split unique.

**Prochaines evolutions envisagees :**
- Integration d'une API MyFitnessPal pour recuperer automatiquement les
  calories et macros reelles plutot que declaratives.
- Notification automatique (email ou DM) si un client bascule en "profil a
  risque" entre deux check-ins.
- Version mobile native (ou PWA) pour un acces encore plus direct depuis le
  telephone du coach.
