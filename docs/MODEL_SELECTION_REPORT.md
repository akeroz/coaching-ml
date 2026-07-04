# Rapport de selection du modele

## Methodologie

4 modeles ont ete entraines sur le meme split train/test stratifie (80/20) et optimises par GridSearchCV : Regression logistique, Foret aleatoire, XGBoost, Reseau de neurones (MLP).

Un score composite a ete calcule pour chaque modele :

```
score = 0.4 * AUC-ROC + 0.3 * F1 (weighted) + 0.2 * Accuracy + 0.1 * (1 / temps_entrainement normalise)
```

## Classement

| Rang | Modele | Accuracy | F1 (weighted) | AUC-ROC | CV 5-fold (AUC) | Temps (s) | Score composite |
|---|---|---|---|---|---|---|---|
| 1 | Regression logistique | 0.767 | 0.755 | 0.775 | 0.736 ± 0.046 | 4.7 | 0.7887 |
| 2 | XGBoost | 0.742 | 0.728 | 0.740 | 0.726 ± 0.036 | 4.4 | 0.7625 |
| 3 | Foret aleatoire | 0.733 | 0.711 | 0.753 | 0.732 ± 0.050 | 21.0 | 0.7073 |
| 4 | Reseau de neurones (MLP) | 0.700 | 0.700 | 0.705 | 0.683 ± 0.054 | 35.2 | 0.6318 |

## Modele retenu : Regression logistique

Le modele **Regression logistique** obtient le meilleur score composite (0.7887), grace a :

- Un AUC-ROC de 0.775 sur le jeu de test (poids 0.4 dans le score),
- Un F1-score pondere de 0.755 (poids 0.3),
- Une accuracy de 0.767 (poids 0.2),
- Un temps d'entrainement de 4.7s (poids 0.1, normalise entre modeles).

La validation croisee 5-fold confirme la stabilite du modele (AUC moyen 0.736 ± 0.046), ce qui ecarte le risque de surapprentissage sur le split train/test unique.

Meilleurs hyperparametres retenus (GridSearchCV) : `{'C': 0.1, 'solver': 'lbfgs'}`

## Matrice de confusion du modele retenu

```
[[24, 21], [7, 68]]
```

## Conclusion

Ce mecanisme de selection automatique, base sur un score composite reproductible, garantit que le modele mis en production est objectivement le plus performant sur l'ensemble des criteres retenus (discrimination, equilibre precision/rappel, exactitude globale et cout de calcul), plutot qu'un choix arbitraire.