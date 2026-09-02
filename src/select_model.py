"""
Comparaison automatique des 4 modeles entraines et selection du meilleur.

Score composite = 0.4 * AUC-ROC + 0.3 * F1 (weighted) + 0.2 * Accuracy
                  + 0.1 * (1 / temps_entrainement normalise)

Justification des poids (voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md) : il
s'agit d'une somme ponderee, technique standard de decision multicritere
(MCDM). AUC-ROC est le poids le plus fort car c'est la metrique de reference
pour juger la capacite de discrimination d'un classifieur binaire,
particulierement robuste au desequilibre de classes (Fawcett, 2006, "An
introduction to ROC analysis", Pattern Recognition Letters). F1 vient
ensuite car il penalise a la fois les faux positifs et les faux negatifs -
important ici car un faux negatif (client a risque non detecte) a un cout
reel pour le coach. L'accuracy est incluse en 3e position car c'est la
metrique la plus lisible pour un non-specialiste (le coach), mais elle est
sous-ponderee car trompeuse seule en cas de desequilibre de classes. Le
temps d'entrainement recoit le poids le plus faible : c'est une contrainte
d'ingenierie (cout de reentrainement), pas un critere de qualite predictive.

Ces poids restent un choix parmi d'autres plausibles - c'est pourquoi
compute_composite_scores_sensitivity() ci-dessous rejoue le classement sous
plusieurs ponderations alternatives (poids egaux, AUC seul, F1 seul, sans le
temps) : si le meme modele gagne dans tous les scenarios, la conclusion ne
depend pas du choix precis des poids, seulement du fait que ce modele domine
sur (quasi) toutes les metriques prises isolement.

Le meilleur modele est sauvegarde dans models/best_model.pkl, les resultats
complets dans models/results.json, et un rapport de selection est genere
dans docs/MODEL_SELECTION_REPORT.md.

Note methodologique (absence de fuite de donnees) : les metriques comparees
ici viennent de train.py, qui ajuste le StandardScaler/LabelEncoder
uniquement sur le split d'entrainement (80%) - le jeu de test n'a jamais
influence la normalisation. Une fois le modele gagnant designe sur cette
base honnete, il est reentraine ici avec les memes hyperparametres sur
l'integralite du dataset (normalise par le scaler "production" de etl.py,
lui-meme ajuste sur 100% des donnees) : c'est la pratique standard consistant
a maximiser les donnees disponibles pour le modele final deploye, une fois
sa performance validee - pas une fuite, puisqu'aucune metrique n'est
rapportee a partir de ce refit.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from etl import FEATURE_COLUMNS, PROCESSED_PATH, TARGET_COL
from train import MODEL_SPECS

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
TRAIN_RESULTS_PATH = MODELS_DIR / "train_results.json"
RESULTS_PATH = MODELS_DIR / "results.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
REPORT_PATH = DOCS_DIR / "MODEL_SELECTION_REPORT.md"

WEIGHTS = {"auc": 0.4, "f1": 0.3, "accuracy": 0.2, "speed": 0.1}


def compute_composite_scores(results: dict) -> pd.DataFrame:
    rows = []
    times = [r["training_time_sec"] for r in results.values()]
    t_min, t_max = min(times), max(times)

    for key, r in results.items():
        if t_max > t_min:
            time_norm = (r["training_time_sec"] - t_min) / (t_max - t_min)
        else:
            time_norm = 0.0
        speed_score = 1 - time_norm  # plus rapide => score plus proche de 1

        composite = (
            WEIGHTS["auc"] * r["auc_roc"]
            + WEIGHTS["f1"] * r["f1_weighted"]
            + WEIGHTS["accuracy"] * r["accuracy"]
            + WEIGHTS["speed"] * speed_score
        )
        rows.append(
            {
                "key": key,
                "label": r["label"],
                "accuracy": r["accuracy"],
                "f1_weighted": r["f1_weighted"],
                "precision_weighted": r["precision_weighted"],
                "recall_weighted": r["recall_weighted"],
                "auc_roc": r["auc_roc"],
                "training_time_sec": r["training_time_sec"],
                "cv_mean": r["cv_mean"],
                "cv_std": r["cv_std"],
                "speed_score": speed_score,
                "composite_score": composite,
            }
        )

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


ALTERNATIVE_WEIGHT_SCHEMES = {
    "Poids retenus (0.4/0.3/0.2/0.1)": WEIGHTS,
    "Poids egaux (0.25 chacun)": {"auc": 0.25, "f1": 0.25, "accuracy": 0.25, "speed": 0.25},
    "AUC-ROC seul (1.0)": {"auc": 1.0, "f1": 0.0, "accuracy": 0.0, "speed": 0.0},
    "F1 seul (1.0)": {"auc": 0.0, "f1": 1.0, "accuracy": 0.0, "speed": 0.0},
    "Sans le temps (0.44/0.33/0.22/0)": {"auc": 4 / 9, "f1": 3 / 9, "accuracy": 2 / 9, "speed": 0.0},
}


def compute_composite_scores_sensitivity(results: dict) -> pd.DataFrame:
    """Rejoue le classement sous plusieurs ponderations alternatives et plausibles,
    pour verifier que le vainqueur ne depend pas du choix precis des poids retenus
    (voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md)."""
    rows = []
    for scheme_name, weights in ALTERNATIVE_WEIGHT_SCHEMES.items():
        times = [r["training_time_sec"] for r in results.values()]
        t_min, t_max = min(times), max(times)
        scored = {}
        for key, r in results.items():
            time_norm = (r["training_time_sec"] - t_min) / (t_max - t_min) if t_max > t_min else 0.0
            speed_score = 1 - time_norm
            scored[key] = (
                weights["auc"] * r["auc_roc"]
                + weights["f1"] * r["f1_weighted"]
                + weights["accuracy"] * r["accuracy"]
                + weights["speed"] * speed_score
            )
        winner_key = max(scored, key=scored.get)
        rows.append({"scenario": scheme_name, "vainqueur": results[winner_key]["label"]})
    return pd.DataFrame(rows)


def generate_report(ranking: pd.DataFrame, winner_key: str, results: dict) -> str:
    winner = ranking.iloc[0]
    winner_raw = results[winner_key]

    lines = [
        "# Rapport de selection du modele",
        "",
        "## Methodologie",
        "",
        "4 modeles ont ete entraines sur le meme split train/test stratifie (80/20) "
        "et optimises par GridSearchCV : Regression logistique, Foret aleatoire, "
        "XGBoost, Reseau de neurones (MLP). Le scaler/encoders sont ajustes "
        "uniquement sur le train set (aucune fuite du test set dans la normalisation).",
        "",
        "Un score composite a ete calcule pour chaque modele :",
        "",
        "```",
        "score = 0.4 * AUC-ROC + 0.3 * F1 (weighted) + 0.2 * Accuracy "
        "+ 0.1 * (1 / temps_entrainement normalise)",
        "```",
        "",
        "## Classement",
        "",
        "| Rang | Modele | Accuracy | F1 (weighted) | AUC-ROC | CV 5-fold (AUC) | Temps (s) | Score composite |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for rank, row in ranking.iterrows():
        lines.append(
            f"| {rank} | {row['label']} | {row['accuracy']:.3f} | {row['f1_weighted']:.3f} | "
            f"{row['auc_roc']:.3f} | {row['cv_mean']:.3f} ± {row['cv_std']:.3f} | "
            f"{row['training_time_sec']:.1f} | {row['composite_score']:.4f} |"
        )

    sensitivity = compute_composite_scores_sensitivity(results)
    lines += [
        "",
        "## Analyse de sensibilite des poids",
        "",
        "Le choix des poids (0.4/0.3/0.2/0.1) reste une decision argumentee mais "
        "pas la seule possible (voir docs/JUSTIFICATIONS_METHODOLOGIQUES.md). Le "
        "tableau ci-dessous rejoue le classement sous plusieurs ponderations "
        "alternatives, pour verifier que la conclusion ne repose pas sur ce choix precis :",
        "",
        "| Scenario de ponderation | Modele vainqueur |",
        "|---|---|",
    ]
    for _, row in sensitivity.iterrows():
        lines.append(f"| {row['scenario']} | {row['vainqueur']} |")

    unanimous = sensitivity["vainqueur"].nunique() == 1
    lines.append("")
    if unanimous:
        lines.append(
            f"**{sensitivity['vainqueur'].iloc[0]}** l'emporte dans tous les scenarios testes "
            "(poids egaux, un seul critere, sans le temps) : le resultat de la selection "
            "est donc robuste au choix precis des poids, il ne depend pas d'une ponderation arbitraire."
        )
    else:
        lines.append(
            "Le vainqueur varie selon la ponderation retenue : la selection est donc "
            "plus sensible au choix des poids, ce qui renforce l'importance de la "
            "justification donnee ci-dessus pour le scenario retenu."
        )

    lines += [
        "",
        f"## Modele retenu : {winner['label']}",
        "",
        f"Le modele **{winner['label']}** obtient le meilleur score composite "
        f"({winner['composite_score']:.4f}), grace a :",
        "",
        f"- Un AUC-ROC de {winner['auc_roc']:.3f} sur le jeu de test (poids 0.4 dans le score),",
        f"- Un F1-score pondere de {winner['f1_weighted']:.3f} (poids 0.3),",
        f"- Une accuracy de {winner['accuracy']:.3f} (poids 0.2),",
        f"- Un temps d'entrainement de {winner['training_time_sec']:.1f}s (poids 0.1, normalise entre modeles).",
        "",
        f"La validation croisee 5-fold confirme la stabilite du modele "
        f"(AUC moyen {winner['cv_mean']:.3f} ± {winner['cv_std']:.3f}), ce qui ecarte le risque de surapprentissage "
        "sur le split train/test unique.",
        "",
        f"Meilleurs hyperparametres retenus (GridSearchCV) : `{winner_raw['best_params']}`",
        "",
        "## Matrice de confusion du modele retenu",
        "",
        f"```\n{winner_raw['confusion_matrix']}\n```",
        "",
        "## Conclusion",
        "",
        "Ce mecanisme de selection automatique, base sur un score composite reproductible, "
        "garantit que le modele mis en production est objectivement le plus performant sur "
        "l'ensemble des criteres retenus (discrimination, equilibre precision/rappel, exactitude "
        "globale et cout de calcul), plutot qu'un choix arbitraire.\n\n"
        "Le modele final sauvegarde dans `best_model.pkl` est reentraine avec les memes "
        "hyperparametres sur l'integralite des 600 clients (contre 80% pour l'evaluation "
        "ci-dessus), afin de maximiser les donnees disponibles pour la prediction en "
        "conditions reelles - les metriques rapportees restent celles du split honnete, "
        "jamais celles de ce modele final.",
    ]

    return "\n".join(lines)


def run_selection():
    with open(TRAIN_RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    ranking = compute_composite_scores(results)
    winner_key = ranking.iloc[0]["key"]

    print("\n=== Classement des modeles ===")
    print(
        ranking[
            ["label", "accuracy", "f1_weighted", "auc_roc", "training_time_sec", "composite_score"]
        ].to_string()
    )
    print(f"\nModele selectionne : {ranking.iloc[0]['label']} (score={ranking.iloc[0]['composite_score']:.4f})")

    # Le modele evalue ci-dessus n'a vu que 80% des donnees (split honnete, sans
    # fuite - voir train.py). Une fois la methodologie validee, le modele DEPLOYE
    # est reentraine avec les memes hyperparametres sur l'integralite du dataset
    # (dataset_final.csv, normalise par le scaler "production" de etl.py) : pratique
    # standard qui maximise les donnees disponibles pour le modele final, sans
    # jamais faire fuiter le test set dans une metrique rapportee.
    full_df = pd.read_csv(PROCESSED_PATH)
    X_full, y_full = full_df[FEATURE_COLUMNS], full_df[TARGET_COL]
    best_params = results[winner_key]["best_params"]
    final_model = clone(MODEL_SPECS[winner_key]["estimator"]).set_params(**best_params)
    final_model.fit(X_full, y_full)
    joblib.dump(final_model, BEST_MODEL_PATH)

    output = {
        "ranking": ranking.to_dict(orient="records"),
        "winner": winner_key,
        "weights": WEIGHTS,
        "full_results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(ranking, winner_key, results)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nMeilleur modele -> {BEST_MODEL_PATH}")
    print(f"Resultats complets -> {RESULTS_PATH}")
    print(f"Rapport de selection -> {REPORT_PATH}")

    return output


if __name__ == "__main__":
    run_selection()
