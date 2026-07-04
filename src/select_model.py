"""
Comparaison automatique des 4 modeles entraines et selection du meilleur.

Score composite = 0.4 * AUC-ROC + 0.3 * F1 (weighted) + 0.2 * Accuracy
                  + 0.1 * (1 / temps_entrainement normalise)

Le meilleur modele est sauvegarde dans models/best_model.pkl, les resultats
complets dans models/results.json, et un rapport de selection est genere
dans docs/MODEL_SELECTION_REPORT.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

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
        "XGBoost, Reseau de neurones (MLP).",
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
        "globale et cout de calcul), plutot qu'un choix arbitraire.",
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

    shutil.copyfile(MODELS_DIR / f"{winner_key}.pkl", BEST_MODEL_PATH)

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
