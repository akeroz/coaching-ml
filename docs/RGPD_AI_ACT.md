# Analyse RGPD & AI Act

## 1. RGPD

### 1.1 Finalite et base legale

Le traitement vise a estimer la probabilite qu'un client de coaching fitness
atteigne son objectif physique, afin d'adapter l'accompagnement propose par
le coach. Base legale mobilisable : **execution du contrat de coaching**
(art. 6.1.b RGPD) conclu avec chaque client, complete par l'**interet
legitime** du coach a ameliorer la qualite de son accompagnement (art. 6.1.f).

### 1.2 Minimisation et pseudonymisation (art. 5.1.c, art. 4(5))

Le projet applique une **separation stricte** entre deux tables :

| Table | Contenu | Usage |
|---|---|---|
| `data/raw/annuaire_clients.csv` | `client_id`, `prenom`, `nom` | Affichage cote coach uniquement (lisibilite du dashboard) |
| `data/processed/dataset_final.csv` | `client_id` + features numeriques/encodees + cible | Entrainement et prediction du modele |

Le pipeline ML (`etl.py`, `train.py`, `select_model.py`, `predict.py`) ne
charge et ne manipule **jamais** la table d'identite : seul `app.py` la
fusionne, et uniquement pour l'affichage (page "Pipeline ETL" et "Dashboard
de suivi clients"). Le modele entraine n'a donc aucune dependance a
l'identite du client — un changement de nom n'affecte ni le score ni les
predictions.

Dans ce projet, les identites (prenom/nom) sont **generees synthetiquement**
(aucune personne reelle). En conditions reelles (vrais clients), cette table
d'identite devrait en plus etre :
- stockee hors du depot de code partage (ex. base separee ou fichier chiffre,
  jamais commit sur GitHub) ;
- accessible uniquement au coach (pas d'acces tiers, pas de duplication dans
  des outils d'analytics) ;
- purgee/anonymisee a la fin de la relation contractuelle avec le client
  (voir duree de conservation ci-dessous).

### 1.3 Duree de conservation

Les donnees d'un client sont conservees pour la duree du suivi coaching plus
une periode raisonnable d'analyse de la progression (ex. 12 mois apres la fin
du suivi), puis supprimees ou anonymisees (suppression du lien
`client_id` <-> `prenom/nom`, conservation eventuelle des seules donnees
statistiques anonymisees pour ameliorer le modele).

### 1.4 Droits des personnes concernees

Chaque client dispose des droits d'acces, de rectification, d'effacement,
d'opposition et de portabilite sur ses donnees (art. 15 a 20 RGPD). La
separation identite/dataset facilite concretement l'exercice de ces droits :
supprimer un client revient a supprimer sa ligne dans les deux tables, sans
avoir a "extraire" une identite d'un dataset d'entrainement deja vectorise.

### 1.5 Absence de decision entierement automatisee (art. 22)

Le systeme ne prend **aucune decision automatique** ayant un effet sur le
client : il produit une probabilite et une recommandation textuelle
("profil favorable / a risque / critique"), affichees uniquement au coach.
C'est le coach qui interprete ce resultat et decide de l'accompagnement
propose. Ce principe de supervision humaine est rappele explicitement dans
l'application (page "Prediction en temps reel").

### 1.6 Securite

Pas de champ directement identifiant (email, telephone, reseau social) dans
l'un ou l'autre des deux datasets. Stockage local (CSV/pickle), pas de
transmission a des tiers. En production, l'acces a la table d'identite
devrait etre restreint (voir 1.2).

## 2. AI Act (reglement europeen sur l'intelligence artificielle)

### 2.1 Classification du niveau de risque

L'AI Act classe les systemes d'IA selon 4 niveaux : inacceptable, eleve,
limite, minimal. Le systeme decrit ici :

- n'entre dans **aucune categorie a risque eleve** de l'Annexe III (pas de
  systeme RH/emploi, pas de notation de credit, pas de biometrie, pas de
  dispositif medical au sens reglementaire — il ne pose pas de diagnostic
  de sante, il estime une probabilite d'atteinte d'un objectif sportif) ;
- ne constitue pas une pratique interdite (pas de manipulation, pas de
  notation sociale, pas de surveillance biometrique de masse) ;
- s'apparente a un systeme a **risque limite / minimal**, utilise comme
  outil d'aide a la decision pour un professionnel (le coach), avec
  supervision humaine systematique.

### 2.2 Obligations de transparence (art. 50)

Pour les systemes a risque limite interagissant avec des personnes,
l'AI Act impose une information claire que le contenu/resultat provient d'un
systeme d'IA. Cette obligation est mise en oeuvre :
- bandeau explicite sur la page de prediction ("ce resultat est une
  estimation statistique produite par un systeme d'IA, pas une decision
  automatique") ;
- interpretation textuelle systematique associee a chaque probabilite
  affichee (jamais un chiffre seul sans contexte) ;
- documentation du modele retenu et de ses limites accessible depuis
  l'application (rapport de selection, page "Comparaison des modeles").

### 2.3 Supervision humaine

Le systeme est concu pour rester un outil d'aide a la decision : la
prediction n'est jamais transmise directement au client, elle est destinee
au coach qui reste seul decisionnaire de l'accompagnement. Cette exigence de
supervision humaine (comparable a l'art. 14 AI Act pour les systemes a risque
eleve, appliquee ici par bonne pratique bien que non obligatoire pour un
risque minimal) est documentee et rappelee dans l'interface.

### 2.4 Gestion des biais et documentation technique

Limites assumees et documentees (voir aussi `docs/PROJECT_MANAGEMENT.md`,
section retrospective) :
- le dataset d'entrainement est **synthetique**, calibre pour etre realiste
  (formule de Mifflin-St Jeor, plages de valeurs coherentes), mais ne
  reflete pas la distribution exacte d'une patientele reelle — un
  re-entrainement sur des donnees reelles anonymisees est necessaire avant
  tout usage en production sur de vrais clients ;
- l'equilibre des classes de la variable cible est verifie (62% / 38%,
  voir page "Pipeline ETL") pour eviter un biais de classe majoritaire ;
- les metriques completes (accuracy, F1, precision, rappel, AUC-ROC,
  matrice de confusion, validation croisee) sont conservees et publiees
  (`models/results.json`, `docs/MODEL_SELECTION_REPORT.md`), permettant une
  tracabilite et un audit du modele a tout moment.

### 2.5 Synthese

| Exigence AI Act | Statut dans le projet |
|---|---|
| Classification du risque | Risque minimal/limite, justifie ci-dessus |
| Transparence utilisateur | Bandeau + interpretation textuelle sur chaque prediction |
| Supervision humaine | Le coach reste seul decisionnaire, rappele dans l'UI |
| Documentation technique | Rapport de selection, metriques, learning curves conserves |
| Gestion des biais | Equilibre des classes verifie, limites du dataset synthetique documentees |
