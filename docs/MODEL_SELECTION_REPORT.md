# Rapport de selection du modele

## Methodologie

4 modeles ont ete entraines sur le meme split train/test stratifie (80/20) et optimises par GridSearchCV : Regression logistique, Foret aleatoire, XGBoost, Reseau de neurones (MLP). Le scaler/encoders sont ajustes uniquement sur le train set (aucune fuite du test set dans la normalisation).

Un score composite a ete calcule pour chaque modele :

```
score = 0.4 * AUC-ROC + 0.3 * F1 (weighted) + 0.2 * Accuracy + 0.1 * (1 / temps_entrainement normalise)
```

## Classement

| Rang | Modele | Accuracy | F1 (weighted) | AUC-ROC | CV 5-fold (AUC) | Temps (s) | Score composite |
|---|---|---|---|---|---|---|---|
| 1 | Regression logistique | 0.767 | 0.755 | 0.775 | 0.736 ± 0.047 | 2.9 | 0.7898 |
| 2 | XGBoost | 0.742 | 0.728 | 0.740 | 0.726 ± 0.036 | 3.7 | 0.7587 |
| 3 | Foret aleatoire | 0.733 | 0.711 | 0.741 | 0.733 ± 0.049 | 14.4 | 0.7034 |
| 4 | Reseau de neurones (MLP) | 0.708 | 0.706 | 0.718 | 0.684 ± 0.053 | 24.6 | 0.6406 |

## Analyse de sensibilite des poids

Le choix des poids (0.4/0.3/0.2/0.1) reste une decision argumentee mais pas la seule possible (voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md). Le tableau ci-dessous rejoue le classement sous plusieurs ponderations alternatives, pour verifier que la conclusion ne repose pas sur ce choix precis :

| Scenario de ponderation | Modele vainqueur |
|---|---|
| Poids retenus (0.4/0.3/0.2/0.1) | Regression logistique |
| Poids egaux (0.25 chacun) | Regression logistique |
| AUC-ROC seul (1.0) | Regression logistique |
| F1 seul (1.0) | Regression logistique |
| Sans le temps (0.44/0.33/0.22/0) | Regression logistique |

**Regression logistique** l'emporte dans tous les scenarios testes (poids egaux, un seul critere, sans le temps) : le resultat de la selection est donc robuste au choix precis des poids, il ne depend pas d'une ponderation arbitraire.

## Modele retenu : Regression logistique

Le modele **Regression logistique** obtient le meilleur score composite (0.7898), grace a :

- Un AUC-ROC de 0.775 sur le jeu de test (poids 0.4 dans le score),
- Un F1-score pondere de 0.755 (poids 0.3),
- Une accuracy de 0.767 (poids 0.2),
- Un temps d'entrainement de 2.9s (poids 0.1, normalise entre modeles).

La validation croisee 5-fold confirme la stabilite du modele (AUC moyen 0.736 ± 0.047), ce qui ecarte le risque de surapprentissage sur le split train/test unique.

Meilleurs hyperparametres retenus (GridSearchCV) : `{'C': 0.1, 'solver': 'lbfgs'}`

## Matrice de confusion du modele retenu

```
[[24, 21], [7, 68]]
```

## Conclusion

Ce mecanisme de selection automatique, base sur un score composite reproductible, garantit que le modele mis en production est objectivement le plus performant sur l'ensemble des criteres retenus (discrimination, equilibre precision/rappel, exactitude globale et cout de calcul), plutot qu'un choix arbitraire.

Le modele final sauvegarde dans `best_model.pkl` est reentraine avec les memes hyperparametres sur l'integralite des 600 clients (contre 80% pour l'evaluation ci-dessus), afin de maximiser les donnees disponibles pour la prediction en conditions reelles - les metriques rapportees restent celles du split honnete, jamais celles de ce modele final.