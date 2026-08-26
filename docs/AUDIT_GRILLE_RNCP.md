# Audit de couverture de la grille RNCP

Ce document reprend le referentiel de competences "chef.fe de projet
expert.e en intelligence artificielle" (EPSI, septembre 2022) fourni par
l'utilisateur, competence par competence, et indique honnetement ou en est
le projet : **Valide** (preuve concrete dans le repo), **Partiel**
(demontre mais avec une adaptation au contexte solo a expliquer a l'oral,
ou un point a renforcer), **A construire** (rien de concret actuellement).

L'objectif n'est pas de se donner raison partout - certaines competences du
Bloc 4 portent sur le management d'une equipe reelle, difficilement
demontrables dans un projet mene seul. Pour celles-la, la bonne reponse
n'est pas d'ajouter du code, mais de preparer un discours honnete a l'oral.

---

## Bloc 1 - Analyser un besoin ou une demande en IA

| Competence | Statut | Preuve / commentaire |
|---|---|---|
| Collecter le besoin d'une direction metier, analyser au regard du contexte et des enjeux | **Valide** | `docs/CDC.md` section "Contexte et besoin" - ici le coach est a la fois porteur du besoin et client final (auto-entreprise), ce qui remplace legitimement la "direction metier" d'un contexte salarie |
| Formaliser le besoin via une note de cadrage (contexte, besoin, objectifs, plus-value, risques) | **Valide** | `docs/CDC.md` - contexte/besoin/objectifs/perimetre + nouvelle section "Note de faisabilite / opportunite" (plus-value) et "Analyse de risques du projet" (risques), ajoutees pour completer la note de cadrage |
| Analyser les risques du projet IA (indicateur EY) selon strategie numerique et enjeux metiers | **Partiel** | Grille de risques ajoutee (`docs/CDC.md`, probabilite x impact x mitigation) selon une logique standard d'analyse de risque - je n'ai pas le detail exact de l'"indicateur EY" mentionne dans le referentiel (outil/grille specifique du cours). Si tu as le support de cours associe, dis-le moi pour que j'aligne la grille sur ce format precis |
| Mettre en place une veille technologique et reglementaire FR/EN | **Valide** | `docs/CDC.md` section "Veille technologique et reglementaire" (AI Act, CNIL, doc scikit-learn/XGBoost/Streamlit) |
| Rechercher et evaluer les solutions disponibles (reseaux de neurones, arbres de decision, forets aleatoires, boosting, clustering...) | **Valide** | `src/train.py` (4 familles : lineaire, bagging, boosting, reseau de neurones) + `docs/PROJECT_MANAGEMENT.md` "Pilotage des prestataires techniques" (comparatifs scikit-learn/TensorFlow, XGBoost/LightGBM/CatBoost). Le clustering n'est pas pertinent ici (probleme supervise, pas de segmentation) - a assumer si le jury demande pourquoi |
| Auditer les donnees de l'entreprise necessaires au projet, analyse macro, respect RGPD | **Valide** | `docs/DICTIONNAIRE_DONNEES.md` (nomenclature complete) + `docs/RGPD_AI_ACT.md` |
| Rediger une note de faisabilite/opportunite pour validation par les decideurs | **Valide** | `docs/CDC.md` nouvelle section "Note de faisabilite / opportunite" |

---

## Bloc 2 - Concevoir une solution IA

| Competence | Statut | Preuve / commentaire |
|---|---|---|
| Analyser le corpus de donnees, choisir les donnees appropriees, outils ETL (avec/sans DPO) | **Valide** | `src/etl.py`, `docs/DICTIONNAIRE_DONNEES.md`. Pas de DPO au sens strict (auto-entreprise) - role assume par le coach lui-meme, documente dans `docs/RGPD_AI_ACT.md` |
| Structurer les donnees dans le respect du RGPD pour produire un prototype | **Valide** | Schema `src/db.py` (tables `clients`/`suivis_hebdo`), separation stricte donnees reelles/synthetiques (`docs/RGPD_AI_ACT.md` §1.2) |
| Modeliser l'architecture de donnees (flux DATA, framework) | **Valide** | `docs/ARCHITECTURE.md` (pipeline complet) + `docs/architecture_diagram.png` |
| Evaluer l'adequation des modeles disponibles avec le projet | **Valide** | `docs/MODEL_SELECTION_REPORT.md` (comparatif 4 modeles) |
| Concevoir/adapter un modele a partir des specificites des donnees (analyses statistiques/mathematiques) | **Valide** | Feature engineering base sur des formules validees (Mifflin-St Jeor, IMC, ratio proteines...) + GridSearchCV par modele - voir `docs/JUSTIFICATIONS_METHODOLOGIQUES.md` |
| Analyser performances et capacite predictive du modele | **Valide** | Metriques completes (accuracy/F1/precision/rappel/AUC-ROC/matrice de confusion/CV 5-fold/learning curves) + analyse de sensibilite des poids du score composite (`src/select_model.py`) |
| Definir une procedure d'entrainement adequate, selectionner les donnees d'apprentissage | **Valide** | `src/train.py::load_data()` - split stratifie avant normalisation (anti-fuite), documente |
| Definir une phase de test et validation du modele choisi | **Valide** | Test set 20% + validation croisee 5-fold + learning curves |
| Maquetter l'infrastructure necessaire au deploiement | **Valide** | `docs/ARCHITECTURE.md` nouvelle section "Maquette de l'infrastructure de deploiement" (GitHub -> Streamlit Cloud -> Supabase) |

---

## Bloc 3 - Maintenabilite et deploiement

| Competence | Statut | Preuve / commentaire |
|---|---|---|
| Definir un process de maintenabilite (assurance qualite) | **Valide** | `docs/PROJECT_MANAGEMENT.md` "Maintenabilite et procedure de rollback" + `models/retrain_history.json` (suivi de derive) |
| Gerer la documentation d'information et technique | **Valide** | `docs/` complet : CDC, ARCHITECTURE, PROJECT_MANAGEMENT, RGPD_AI_ACT, MODEL_SELECTION_REPORT, JUSTIFICATIONS_METHODOLOGIQUES, DICTIONNAIRE_DONNEES |
| Realiser/superviser un test de deploiement par simulation a l'echelle reelle | **Partiel** | CI (`'.github/workflows/tests.yml`) execute les tests a chaque push, et l'app a ete testee en conditions reelles (navigateur) a chaque evolution majeure. Il n'existe pas d'environnement de **staging** distinct de la prod - choix assume et justifie dans `docs/ARCHITECTURE.md` (contrainte budget zero), a expliciter clairement a l'oral plutot que laisser croire a un oubli |
| Apporter une expertise technique pour garantir le bon deploiement / resoudre des problemes techniques | **Partiel** | Demontre en pratique (ex. debogage de la connexion Supabase IPv6/pooler, correction de la fuite de donnees) mais documente comme retrospective solo (`docs/PROJECT_MANAGEMENT.md`), pas comme support a une equipe - a assumer a l'oral : le role d'expert technique s'est exerce sur soi-meme faute d'equipe |
| Proposer une strategie d'accompagnement du changement, plan d'action | **Valide** | `docs/PROJECT_MANAGEMENT.md` nouvelle section "Strategie d'accompagnement du changement" |
| Assurer la bonne utilisation de l'IA en accompagnant les partenaires | **Valide** | Meme section - communication au client sur ce que la prediction signifie et ne signifie pas |

---

## Bloc 4 - Manager un projet avec agilite

| Competence | Statut | Preuve / commentaire |
|---|---|---|
| Identifier les etapes, organiser en taches/livrables selon les ressources | **Valide** | `docs/PROJECT_MANAGEMENT.md` tableau des sprints |
| Concevoir les cahiers des charges technique et fonctionnel | **Valide** | `docs/CDC.md` (perimetre fonctionnel + technique) |
| Gerer un projet agile (methodes/outils, iteration) | **Partiel** | 3 sprints documentes avec retrospective et decisions qui ont evolue (`docs/PROJECT_MANAGEMENT.md`) - le format est compresse (une journee) par rapport a un vrai cycle agile sur plusieurs semaines : a assumer comme adaptation d'echelle, les principes (iteration, retro, ajustement) restent presents |
| Etablir des tableaux de bord de suivi de performance | **Valide** | `docs/PROJECT_MANAGEMENT.md` tableau temps estime/reel/ecart + page "Gestion de projet" dans l'app |
| Piloter des prestataires exterieurs (cartographie SI) | **Valide** (adapte) | `docs/PROJECT_MANAGEMENT.md` "Pilotage des prestataires techniques" - les bibliotheques/frameworks open source (scikit-learn, XGBoost, Streamlit) sont traites comme des choix de prestataires a arbitrer, faute de prestataires humains reels dans un projet solo |
| Conduire une equipe projet en diffusant les fondamentaux de l'agilite | **A construire** (narratif a preparer) | Pas d'equipe reelle a "conduire". A l'oral : expliquer que les principes (adaptation, flexibilite, amelioration continue) ont ete appliques a l'auto-organisation du projet, et decrire comment ils s'appliqueraient si l'activite grossissait (ex. recrutement d'un second coach) |
| Adopter une strategie d'accueil aux handicaps | **Partiel** | `docs/PROJECT_MANAGEMENT.md` "Strategie d'accueil handicap" - strategie ecrite et coherente (accessibilite WCAG, navigation clavier) mais construite pour une equipe hypothetique, pas verifiee sur une personne reelle en situation de handicap. A presenter comme une strategie preparee en amont, pas une preuve d'usage |
| Communiquer avec l'equipe (culture/langue) pour garantir l'integration | **Partiel** | `docs/PROJECT_MANAGEMENT.md` "Communication et coordination a distance" reformule les clients comme "equipe distribuee" (DM Instagram, Sheets partage, ManyChat, canal dedie) - une analogie plausible (parties prenantes regulieres, distantes, geree via outils numeriques) mais qui n'est pas une equipe projet au sens strict. A assumer explicitement comme une transposition du principe, pas un contournement, sinon le jury peut la percevoir comme artificielle |
| Animer des reunions a distance pour maintenir la dynamique de groupe | **A construire** (narratif a preparer) | Pas de "reunions" dans une relation coach-client individuelle. A l'oral : mentionner comment ce principe s'appliquerait a une equipe (ex. si un second coach rejoignait l'activite, format de reunion hebdo a distance envisage) |
| Accompagner l'equipe dans l'appropriation du teletravail (motivation, resilience, equilibre vie pro/perso) | **A construire** (narratif a preparer) | Meme limite - pas d'equipe salariee. A preparer comme scenario hypothetique de croissance de l'activite plutot que comme preuve actuelle |

---

## Synthese et priorites avant l'oral

**Solide (a defendre sereinement)** : Bloc 1 (sauf l'indicateur EY specifique),
Bloc 2 en integralite, Bloc 3 en grande partie, Bloc 4 sur la partie
cadrage/pilotage/outils.

**A clarifier explicitement a l'oral (pas a coder)** : toutes les
competences du Bloc 4 liees au management d'une equipe reelle (conduite
d'equipe, reunions a distance, accompagnement teletravail). La meilleure
strategie n'est pas de simuler artificiellement une equipe dans le code,
mais d'assumer directement devant le jury : "le projet est mene seul, donc
ces competences sont demontrees soit par transposition (les clients comme
parties prenantes distantes a coordonner), soit par un scenario prepare de
passage a l'echelle (que ferais-je si je recrutais un second coach)".

**Seul point factuellement incomplet** : la correspondance exacte avec
l'"indicateur EY" cite dans le referentiel pour l'analyse de risques - si
un support de cours precise cet outil, il vaut mieux aligner
`docs/CDC.md` dessus plutot que sur une grille de risque generique.
