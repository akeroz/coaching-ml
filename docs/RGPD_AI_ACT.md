# Analyse RGPD & AI Act

## 1. RGPD

### 1.1 Finalite et base legale

Le traitement vise a estimer la probabilite qu'un client de coaching fitness
atteigne son objectif physique, afin d'adapter l'accompagnement propose par
le coach. Base legale mobilisable : **execution du contrat de coaching**
(art. 6.1.b RGPD) conclu avec chaque client, complete par l'**interet
legitime** du coach a ameliorer la qualite de son accompagnement (art. 6.1.f).

### 1.2 Separation stricte des donnees reelles et du dataset d'entrainement

Le projet distingue deux mondes de donnees totalement separes :

| Composant | Contenu | Statut |
|---|---|---|
| `data/raw/clients_raw.csv`, `data/processed/dataset_final.csv` | Dataset **synthetique** (600 profils generes, aucune personne reelle) | Public (depot GitHub, demo Streamlit Cloud) |
| `data/coaching.db` (SQLite) | Vrais clients du coach (prenom, nom, profil, suivi hebdomadaire de poids) | **Prive**, exclu du depot via `.gitignore`, jamais publie |

Cette separation n'est pas qu'une precaution theorique : `src/db.py` est le
seul module qui lit/ecrit `data/coaching.db`, et ce fichier ne quitte jamais
la machine du coach (il n'est ni commite, ni deploye sur Streamlit Cloud avec
le reste du code). Le pipeline ML (`etl.py`, `train.py`, `select_model.py`,
`predict.py`) ne touche que le dataset synthetique et les modeles entraines
dessus ; le module de reentrainement (`retrain_with_real_data.py`) est le
seul pont entre les deux mondes, et il n'exporte que des **agregats
statistiques anonymises** (features numeriques + issue objectif_atteint) vers
le nouveau dataset d'entrainement — jamais le prenom, le nom, ni aucun
detail identifiant.

### 1.3 Minimisation (art. 5.1.c)

Le modele entraine ne recoit jamais de champ d'identite. Meme lorsqu'un vrai
client est ajoute via la page "Mes clients", seules ses caracteristiques
physiologiques et comportementales (age, poids, calories, adherence...)
alimentent, le cas echeant, un reentrainement — jamais son prenom/nom.

### 1.4 Duree de conservation

Un client reel est conserve dans `data/coaching.db` pour la duree du suivi
coaching plus une periode raisonnable d'analyse de la progression (ex. 12
mois apres la fin du suivi), puis supprime (fonction `db.delete_client`,
deja implementee et utilisable depuis l'application) ou ses donnees
identifiantes anonymisees si le coach souhaite conserver l'historique
statistique pour ameliorer le modele.

### 1.5 Droits des personnes concernees

Chaque client dispose des droits d'acces, de rectification, d'effacement,
d'opposition et de portabilite (art. 15 a 20 RGPD). Concretement dans
l'outil : la page "Mes clients" permet deja de supprimer un client (droit a
l'effacement) et de visualiser/exporter ses donnees (droit d'acces et de
portabilite, via l'export PDF).

### 1.6 Absence de decision entierement automatisee (art. 22)

Le systeme ne prend **aucune decision automatique** ayant un effet sur le
client : il produit une probabilite et une recommandation textuelle
("profil favorable / a risque / critique"), affichee et exportee (PDF)
uniquement a destination du coach. C'est le coach qui interprete ce resultat
et decide de l'accompagnement propose. Ce principe est rappele explicitement
dans l'application (page "Prediction en temps reel") et dans l'export PDF
genere pour chaque client (`src/report.py`).

### 1.7 Securite

`data/coaching.db` est le seul point de stockage de donnees personnelles
reelles ; il reste local, exclu du depot Git, et n'est jamais transmis a un
tiers. Le reentrainement (`retrain_with_real_data.py`) lit ce fichier en
lecture seule et n'ecrit que des artefacts anonymises (modeles, dataset
agrege) dans `models/` et `data/processed/`.

## 2. AI Act (reglement europeen sur l'intelligence artificielle)

### 2.1 Classification du niveau de risque

Le systeme decrit ici :
- n'entre dans **aucune categorie a risque eleve** de l'Annexe III (pas de
  systeme RH/emploi, pas de notation de credit, pas de biometrie, pas de
  dispositif medical au sens reglementaire — il estime une probabilite
  d'atteinte d'un objectif sportif, pas un diagnostic de sante) ;
- ne constitue pas une pratique interdite (pas de manipulation, pas de
  notation sociale, pas de surveillance biometrique de masse) ;
- s'apparente a un systeme a **risque limite / minimal**, utilise comme
  outil d'aide a la decision pour un professionnel (le coach), avec
  supervision humaine systematique.

### 2.2 Obligations de transparence (art. 50)

- Bandeau explicite sur la page de prediction ("ce resultat est une
  estimation statistique produite par un systeme d'IA, pas une decision
  automatique") ;
- Interpretation textuelle systematique associee a chaque probabilite
  affichee, y compris dans l'export PDF envoye au client ;
- Documentation du modele retenu et de ses limites accessible depuis
  l'application (rapport de selection, page "Comparaison des modeles").

### 2.3 Supervision humaine

Le systeme reste un outil d'aide a la decision : la prediction n'est jamais
transmise directement au client sans validation du coach, qui reste seul
decisionnaire de l'accompagnement. Le mecanisme de reentrainement lui-meme
integre une supervision humaine implicite forte : le coach doit explicitement
cliquer sur "Lancer le reentrainement" et le systeme refuse automatiquement
toute promotion d'un modele moins performant (garde anti-regression dans
`retrain_with_real_data.py`), evitant qu'un modele degrade silencieusement
la qualite du service sans intervention humaine.

### 2.4 Gestion des biais et documentation technique

- Le dataset d'entrainement initial est **synthetique**, calibre pour etre
  realiste (formule de Mifflin-St Jeor, plages de valeurs coherentes), mais
  ne reflete pas la distribution exacte d'une patientele reelle. Le
  mecanisme de reentrainement (`retrain_with_real_data.py`) permet de
  reduire progressivement cette limite en integrant des clients reels
  labellises, avec un seuil minimal (5 clients) pour eviter d'introduire du
  bruit statistique instable ;
- L'equilibre des classes de la variable cible est verifie (62% / 38%,
  page "Pipeline ETL") pour eviter un biais de classe majoritaire ;
- Les metriques completes (accuracy, F1, precision, rappel, AUC-ROC,
  matrice de confusion, validation croisee) sont conservees et publiees
  (`models/results.json`, `docs/MODEL_SELECTION_REPORT.md`), permettant une
  tracabilite et un audit du modele a tout moment, y compris apres un
  reentrainement (voir procedure de rollback, `PROJECT_MANAGEMENT.md`).

### 2.5 Synthese

| Exigence AI Act | Statut dans le projet |
|---|---|
| Classification du risque | Risque minimal/limite, justifie ci-dessus |
| Transparence utilisateur | Bandeau + interpretation textuelle sur chaque prediction et export PDF |
| Supervision humaine | Le coach reste seul decisionnaire ; reentrainement declenche manuellement avec garde anti-regression |
| Documentation technique | Rapport de selection, metriques, learning curves conserves et mis a jour a chaque reentrainement |
| Gestion des biais | Equilibre des classes verifie, integration progressive de donnees reelles labellisees |
