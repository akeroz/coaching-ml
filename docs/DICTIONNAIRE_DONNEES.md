# Dictionnaire des donnees

Nomenclature complete de chaque donnee manipulee par l'application : d'ou elle
vient, ce qu'elle signifie, comment elle est utilisee. Objectif : pouvoir
repondre a "c'est quoi cette colonne et pourquoi elle existe" pour n'importe
quel champ, coté donnees synthetiques (entrainement du modele) comme coté
donnees reelles (base clients hebergee).

---

## 1. Deux jeux de donnees distincts, a ne jamais confondre

| | Donnees synthetiques | Donnees reelles (production) |
|---|---|---|
| **Ou** | `data/raw/clients_raw.csv`, `data/processed/dataset_final.csv` | Base Postgres hebergee (Supabase), tables `clients` / `suivis_hebdo` |
| **Origine** | Generees artificiellement (`etl.py::generate_raw_data`), a partir de lois statistiques realistes (age, poids, sexe...) et d'une regle de generation de la cible | Saisies par le coach dans l'app, un vrai client a la fois |
| **A quoi ca sert** | Entrainer et evaluer les 4 modeles (volume suffisant : 600 lignes) | Usage quotidien du coach (suivi reel) + reentrainement periodique une fois assez de clients reels labellises (`MIN_REAL_CLIENTS = 5`, voir `docs/JUSTIFICATIONS_METHODOLOGIQUES.md`) |
| **Identite** | Aucune - `CLIENT_XXX` genere sequentiellement | Pseudonymise - `REAL_XXX` genere sequentiellement, prenom/nom stockes separement (voir `docs/RGPD_AI_ACT.md`) |

Le modele deploye (`models/best_model.pkl`) est entraine sur les donnees
synthetiques, puis periodiquement reentraine en combinant synthetique + reel
des qu'il y a assez de clients reels labellises - jamais sur les donnees
reelles seules (volume insuffisant pour un entrainement stable au debut de
l'activite).

---

## 2. Table `clients` (un client reel = une ligne)

| Colonne | Type | Origine | Signification |
|---|---|---|---|
| `client_id` | texte | genere (`REAL_001`, `REAL_002`...) | Identifiant pseudonyme, jamais l'identite reelle |
| `prenom`, `nom` | texte | saisi par le coach | Identite du client - jamais transmis au modele (voir `predict.py` / `coaching_logic.py::build_profile_from_client_row`, qui ne selectionne que les champs numeriques/categoriels utiles a la prediction) |
| `age` | entier | saisi par le coach | Age du client en annees |
| `sexe` | `H` / `F` | saisi par le coach | Utilise pour le calcul du metabolisme de base (formule Mifflin-St Jeor, terme +5/-161) |
| `taille_cm` | reel | saisi par le coach | Taille en cm |
| `poids_initial_kg` | reel | saisi par le coach, a la creation de la fiche | Poids constate au premier jour du suivi |
| `poids_cible_kg` | reel | defini avec le client | Objectif de poids fixe avec le client |
| `objectif` | `seche` / `prise_masse` / `recomposition` | choisi par le coach avec le client | Type d'objectif physique poursuivi |
| `niveau` | `debutant` / `intermediaire` / `avance` | evalue par le coach | Niveau d'experience sportive du client |
| `frequence_entrainement_semaine` | entier | convenu avec le client | Nombre de seances par semaine prevues au programme |
| `calories_quotidiennes` | reel | defini par le coach (plan nutritionnel) | Apport calorique quotidien cible |
| `proteines_g_par_jour` | reel | defini par le coach (plan nutritionnel) | Apport proteique quotidien cible |
| `heures_sommeil` | reel | declare par le client | Moyenne d'heures de sommeil par nuit |
| `semaines_suivi_prevues` | entier | convenu avec le client | Duree prevue du programme de coaching |
| `adherence_programme_pct` | reel (0-100) | estime/observe par le coach | Taux de respect du programme (entrainements + nutrition) |
| `date_creation` | date | automatique | Date d'entree du client dans l'app |
| `objectif_atteint` | booleen (nullable) | renseigne par le coach a la cloture du suivi | **C'est la cible (label)** utilisee pour reentrainer le modele - null tant que le suivi n'est pas cloture |
| `actif` | booleen | automatique | Suivi en cours (1) ou archive/cloture (0) |
| `consentement_recueilli` | booleen | coche par le coach | Preuve que le client a ete informe de la collecte de ses donnees (RGPD, art. 15-20) |
| `date_consentement` | date (nullable) | automatique | Horodatage du consentement |

## 3. Table `suivis_hebdo` (un check-in = une ligne)

| Colonne | Type | Origine | Signification |
|---|---|---|---|
| `id` | entier | automatique | Identifiant technique de la ligne |
| `client_id` | texte | reference `clients.client_id` | A quel client appartient ce check-in |
| `date_saisie` | date | saisi (ou date du jour par defaut) | Date du check-in |
| `poids` | reel | saisi par le coach (declare par le client) | Poids constate a cette date |
| `note` | texte (optionnel) | saisi par le coach | Commentaire libre sur le ressenti/contexte |
| `energie` | entier 1-5 (optionnel) | declare par le client | Niveau d'energie ressenti (1 = epuise, 5 = pleine forme) |
| `tour_taille_cm` | reel (optionnel) | mesure par le client | Mensuration complementaire au poids seul |

---

## 4. Ce que le modele voit reellement (features d'entree)

Le modele **ne recoit jamais** `client_id`, `prenom`, `nom`, `date_creation`,
`actif`, `consentement_recueilli` ni l'historique des `suivis_hebdo` - unique-
ment les champs numeriques/categoriels de `clients` (ci-dessus), transformes
ainsi (`etl.py`) :

| Feature derivee | Calcul | D'ou ca vient |
|---|---|---|
| `imc` | poids / (taille en m)^2 | Formule standard (Indice de Masse Corporelle, OMS) |
| `ratio_proteines_poids` | proteines_g_par_jour / poids_initial_kg | Ratio nutritionnel classique (g de proteines par kg de poids corporel) |
| `besoin_calorique_estime` | Mifflin-St Jeor (BMR) x facteur d'activite | Formule scientifique publiee (Mifflin et al., 1990) - voir `docs/JUSTIFICATIONS_METHODOLOGIQUES.md` §9 |
| `deficit_calorique` | besoin_calorique_estime - calories_quotidiennes | Ecart entre besoin theorique et apport reel |
| `score_mode_de_vie` | combinaison sommeil + frequence d'entrainement | Indicateur synthetique d'hygiene de vie |

Toutes les variables numeriques sont ensuite normalisees (`StandardScaler`,
ajuste uniquement sur le train set - voir la note anti-fuite dans `train.py`)
et les variables categorielles (`sexe`, `objectif`, `niveau`) encodees
(`LabelEncoder`). Le detail complet de l'entrainement et de la selection du
modele est dans `docs/MODEL_SELECTION_REPORT.md` et
`docs/JUSTIFICATIONS_METHODOLOGIQUES.md`.

## 5. Ou repartir pour aller plus loin

- **Comment le modele est entraine, sur quoi il se base pour predire, et
  pourquoi ces choix** -> `docs/JUSTIFICATIONS_METHODOLOGIQUES.md`
- **Resultats de la comparaison des 4 modeles** -> `docs/MODEL_SELECTION_REPORT.md`
- **Conformite RGPD/AI Act (base legale, retention, securite)** -> `docs/RGPD_AI_ACT.md`
