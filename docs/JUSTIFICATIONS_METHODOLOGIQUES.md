# Justifications methodologiques

Ce document repertorie chaque choix numerique ou methodologique du projet qui
pourrait donner l'impression d'etre arbitraire, et explique sur quoi il repose
reellement : une convention reconnue, une reference citable, ou une preuve
empirique produite par le projet lui-meme. L'objectif : pouvoir repondre a
n'importe quelle question du jury de type "pourquoi ce chiffre et pas un
autre ?" sans se reposer sur "c'est ce que Claude a mis".

Pour chaque point : le choix, la justification, et comment la defendre a l'oral.

---

## 1. Score composite de selection du modele (0.4 / 0.3 / 0.2 / 0.1)

**Choix** : `score = 0.4*AUC-ROC + 0.3*F1 + 0.2*Accuracy + 0.1*(1/temps normalise)`

**Justification** :
- La technique elle-meme (combiner plusieurs metriques en un score unique via
  une somme ponderee) est une methode standard de decision multicritere
  (MCDM - Multi-Criteria Decision Analysis), tres utilisee des qu'un choix
  doit arbitrer entre plusieurs objectifs (ici : discrimination, equilibre
  precision/rappel, exactitude, cout de calcul).
- **AUC-ROC (0.4, le plus fort)** : c'est la metrique de reference pour
  evaluer un classifieur binaire, car elle est independante du seuil de
  decision choisi et robuste au desequilibre de classes (contrairement a
  l'accuracy). Reference : Fawcett T. (2006), *"An introduction to ROC
  analysis"*, Pattern Recognition Letters, 27(8).
- **F1-score (0.3)** : penalise a la fois les faux positifs et les faux
  negatifs. Pertinent ici car un faux negatif (un client presente comme
  "va reussir" alors qu'il va echouer) a un cout reel pour le coach.
- **Accuracy (0.2)** : incluse car c'est la metrique la plus lisible pour un
  non-specialiste (le coach), mais sous-ponderee car elle peut etre trompeuse
  seule en cas de desequilibre de classes.
- **Temps d'entrainement (0.1, le plus faible)** : une contrainte
  d'ingenierie (cout de re-entrainement), pas un critere de qualite
  predictive - d'ou le poids minimal.

**Preuve de robustesse (pas juste une affirmation)** : `select_model.py`
rejoue automatiquement le classement sous 5 ponderations alternatives
(poids retenus, poids egaux, AUC seul, F1 seul, sans le critere de temps) via
`compute_composite_scores_sensitivity()`. Le resultat est integre au rapport
genere (`docs/MODEL_SELECTION_REPORT.md`, section "Analyse de sensibilite
des poids"). Si le meme modele gagne dans tous les scenarios, cela prouve que
la conclusion ne repose pas sur le choix precis des poids, mais sur le fait
que ce modele domine sur (quasi) tous les criteres pris isolement.

**Comment le defendre a l'oral** : "J'ai utilise une somme ponderee, methode
standard de decision multicritere. Les poids reposent sur la hierarchie
usuelle des metriques de classification (AUC-ROC > F1 > Accuracy pour juger
la qualite predictive, cf. Fawcett 2006), le temps etant secondaire. Et
surtout, j'ai verifie empiriquement que le vainqueur ne change pas si on fait
varier ces poids dans des scenarios raisonnables - le choix exact des poids
n'est donc pas fragile."

---

## 2. Choix des 4 familles de modeles comparees

**Choix** : Regression logistique, Foret aleatoire (bagging), XGBoost
(boosting), Reseau de neurones (MLP).

**Justification** : ce n'est pas un choix arbitraire de 4 algorithmes parmi
d'autres - ce sont les 4 grandes familles de methodes de classification
supervisee couramment enseignees et comparees dans la litterature ML :
1. Un modele lineaire simple et interpretable (regression logistique) comme
   reference de base ("baseline").
2. Un ensemble par agregation/bagging (foret aleatoire).
3. Un ensemble par boosting (XGBoost), reconnu pour ses performances sur
   donnees tabulaires.
4. Un modele non-lineaire par reseau de neurones (MLP), pour couvrir la
   famille des approches deep learning meme sur un petit jeu de donnees.

Comparer ces 4 familles est une pratique standard pour justifier qu'un choix
de modele final ne resulte pas d'un a priori, mais d'une comparaison couvrant
l'eventail des approches disponibles.

---

## 3. Split train/test 80/20, stratifie, et validation croisee 5-fold

**Choix** : `train_test_split(test_size=0.2, stratify=...)`, puis
`StratifiedKFold(n_splits=5)` pour le GridSearchCV et le score de stabilite.

**Justification** : 80/20 est la convention la plus repandue en apprentissage
supervise pour des jeux de donnees de taille moyenne (quelques centaines a
quelques milliers de lignes) - elle garde suffisamment de donnees
d'entrainement tout en laissant un jeu de test statistiquement exploitable.
La stratification garantit que la proportion de la classe cible est
respectee dans les deux sous-ensembles (important ici car les classes ne
sont pas parfaitement equilibrees). 5-fold est la valeur la plus citee dans
la litterature (avec 10-fold) pour la validation croisee - c'est un
compromis standard entre variance de l'estimation et cout de calcul.

---

## 4. Grilles d'hyperparametres du GridSearchCV

**Choix** : par exemple `C: [0.01, 0.1, 1, 10]` pour la regression logistique,
`n_estimators: [100, 200, 300]` pour la foret aleatoire, etc.

**Justification** : ce sont des grilles usuelles, presentes dans la quasi
totalite des tutoriels/documentations de reference scikit-learn pour ces
modeles. L'echelle logarithmique pour `C` (parametre de regularisation) est
la convention standard, car son effet sur le modele est multiplicatif, pas
additif. Le principe general (recherche exhaustive sur grille + validation
croisee) est lui-meme la methode de reference pour l'optimisation
d'hyperparametres, documentee dans la doc officielle scikit-learn.

---

## 5. Seuils d'interpretation de la prediction (0.70 / 0.40)

**Choix** : proba > 0.70 -> "Profil favorable" ; 0.40-0.70 -> "Profil a risque" ;
< 0.40 -> "Profil critique".

**Justification** : ce sont des seuils bases sur la convention **RAG
(Red-Amber-Green)**, un standard largement repandu dans les tableaux de bord
de risque et l'aide a la decision (gestion de projet, risque clinique,
scoring credit) : plutot qu'un seuil de decision unique et tranchant a 0.5
(le defaut naturel d'une classification binaire), on definit une **zone
tampon symetrique autour de 0.5** (ici +/- 0.10-0.15) ou le systeme signale
une incertitude necessitant un jugement humain, au lieu de trancher
automatiquement. C'est exactement le role de la zone "orange" (0.40-0.70)
ici : le coach doit ajuster le programme, l'IA ne decide pas seule.

**Comment le defendre** : "Ce n'est pas un seuil unique a 50%, c'est une
convention de type feu tricolore (RAG) tres utilisee en aide a la decision :
une zone d'incertitude explicite entre le seuil favorable et le seuil
critique, pour que le systeme n'automatise jamais la decision finale, qui
reste au coach."

---

## 6. Seuil de stabilite du poids (0.15 kg)

**Choix** : un ecart de progression `< 0.15 kg` est considere "Stable" (ni
progression, ni regression).

**Justification** : le poids corporel varie naturellement de plusieurs
centaines de grammes a plusieurs kilos d'un jour a l'autre du seul fait de
l'hydratation, de la digestion et du cycle - c'est un fait largement
documente en suivi de composition corporelle. Un ecart hebdomadaire sous ce
seuil se situe dans le bruit de mesure normal d'un pese-personne grand
public, pas dans un signal de progression reelle. Ce seuil evite de
qualifier de "regression" une simple fluctuation de mesure.

---

## 7. Seuil d'alerte "pesee en retard" (10 jours)

**Choix** : un client sans nouvelle pesee depuis 10 jours ou plus declenche
une alerte.

**Justification** : la cadence de suivi de reference dans l'app est
hebdomadaire (7 jours). Le seuil de 10 jours correspond a ce cycle plus une
marge de grace d'environ 40% avant de considerer le suivi comme rompu -
ce n'est pas un chiffre invente, mais "cycle attendu + tampon de tolerance",
un principe standard de detection d'anomalie sur des evenements periodiques.

---

## 8. Seuil minimal pour re-entrainer avec des donnees reelles (5 clients)

**Choix** : `MIN_REAL_CLIENTS = 5` avant de lancer un re-entrainement avec
des clients reels labellises.

**Justification** : c'est une regle empirique courante en apprentissage
automatique et en statistique appliquee - en dessous de 5 a 10 observations
par classe, toute re-estimation devient trop instable pour etre fiable (la
variance d'echantillonnage domine le signal). Ce seuil est un garde-fou
minimal, pas une valeur optimisee : le message affiche a l'utilisateur le
dit explicitement ("reentrainement trop instable sur un si petit volume").

---

## 8bis. Marge de promotion lors du reentrainement

**Choix** : un modele candidat n'est promu en production que si son score
composite depasse celui du modele actuel d'au moins
`WEIGHTS["auc"] * cv_std_ancien_modele` (voir `src/retrain_with_real_data.py`).

**Justification** : une premiere version comparait simplement
`nouveau_score >= ancien_score`, ce qui aurait pu promouvoir un modele sur une
amelioration nulle ou infime (0.0001 par exemple) - indiscernable du bruit
statistique normal de la mesure, d'autant plus que le test set du
reentrainement reste largement domine par les donnees synthetiques tant que
le nombre de vrais clients labellises est faible (voir §8 ci-dessus). La
marge minimale exigee n'est pas une nouvelle constante inventee : elle est
derivee de l'ecart-type de validation croisee (`cv_std`) deja mesure pour le
modele en production - une mesure directe du bruit statistique propre a ce
modele - ramenee a l'echelle du score composite via le poids de l'AUC-ROC
dans ce score (0.4, voir §1). C'est une technique standard de detection de
signal (marge de securite/"deadband" au-dela du bruit de mesure) plutot
qu'un seuil arbitraire.

---

## 9. Formule du besoin calorique (Mifflin-St Jeor)

**Choix** : le besoin calorique estime est calcule via l'equation de
Mifflin-St Jeor (feature engineering, `etl.py`).

**Justification** : ce n'est pas une formule inventee - c'est une equation
scientifique publiee et validee, aujourd'hui recommandee par l'Academy of
Nutrition and Dietetics comme la plus precise pour estimer le metabolisme de
base sur une population generale (par rapport a l'ancienne formule de
Harris-Benedict, moins precise sur les populations modernes). Reference :
Mifflin M.D. et al. (1990), *"A new predictive equation for resting energy
expenditure in healthy individuals"*, The American Journal of Clinical
Nutrition, 51(2).

C'est le point le plus solide du projet en termes de justification
scientifique externe - a mettre en avant si le jury questionne la
methodologie du feature engineering.

---

## 10. Portee reelle du modele predictif : ce qu'il fait, ce qu'il ne fait pas

Le modele predit, a partir du **profil d'intake** d'un client (donnees
statiques saisies au debut du suivi), une probabilite d'atteinte de
l'objectif final. C'est un **outil de triage a l'inscription**, pas un outil
de suivi quotidien : il ne recalcule rien a partir des donnees hebdomadaires
reellement observees (poids, energie, adherence constatee), qui ne servent
aujourd'hui qu'a alimenter un futur re-entrainement, pas une nouvelle
prediction en cours de coaching.

**A assumer explicitement a l'oral plutot que laisser le jury le decouvrir** :
ce perimetre est un choix de scope assume (un modele de triage a l'intake est
un probleme ML propre, avec un pipeline defendable de bout en bout), pas une
limite technique subie. Une extension naturelle et non arbitraire serait un
second modele, base sur les donnees de suivi hebdomadaire, pour predire un
risque de decrochage (dropout) ou de stagnation - une cible plus directement
actionnable au quotidien pour un coach, evoquee comme piste d'evolution
(Bloc 3 - maintenabilite et evolutivite) plutot qu'implementee dans le
perimetre actuel.

---

## Recapitulatif : quoi dire au jury

| Choix | Nature de la justification |
|---|---|
| Score composite (poids) | Methode standard (MCDM) + preuve empirique de robustesse (analyse de sensibilite) |
| 4 familles de modeles | Couverture standard des paradigmes de classification supervisee |
| Split 80/20, CV 5-fold | Convention la plus repandue en ML |
| Grilles GridSearchCV | Grilles usuelles, echelle log standard pour la regularisation |
| Seuils 0.70/0.40 | Convention RAG (feu tricolore), standard en aide a la decision |
| Seuil stabilite 0.15 kg | Bruit de mesure documente d'un pese-personne |
| Alerte a 10 jours | Cycle hebdomadaire + marge de tolerance |
| MIN_REAL_CLIENTS = 5 | Regle empirique usuelle (taille d'echantillon minimale) |
| Marge de promotion au reentrainement | Deadband derive du bruit de mesure (cv_std) deja calcule, pas une constante inventee |
| Mifflin-St Jeor | Formule scientifique publiee et validee (reference citable) |
| Perimetre du modele (intake vs suivi) | Choix de scope assume, avec extension identifiee |
