"""Genere un schema visuel de l'architecture du pipeline IA (docs/architecture_diagram.png)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "docs" / "architecture_diagram.png"

STEPS = [
    "Donnees brutes\n(clients_raw.csv)",
    "ETL\n(etl.py)",
    "Features engineered\n(dataset_final.csv)",
    "Split\ntrain / test",
    "Entrainement\n4 modeles",
    "Evaluation\nautomatique",
    "Selection du\nmeilleur modele",
    "Sauvegarde\n(best_model.pkl)",
    "API de prediction\n(predict.py)",
    "Dashboard\nStreamlit (app.py)",
    "Utilisateur final\n(coach)",
]

COLORS = [
    "#cfe8ff", "#a9d6ff", "#8ec9ff", "#ffe8a3", "#ffd27a",
    "#ffd27a", "#ffb84d", "#b7f0c1", "#9ce0ab", "#f5b8d1", "#e0a3c9",
]


def build_diagram():
    n = len(STEPS)
    fig, ax = plt.subplots(figsize=(9, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1)
    ax.axis("off")

    box_w, box_h = 7, 0.72
    x0 = 1.5

    for i, (label, color) in enumerate(zip(STEPS, COLORS)):
        y = n - i
        box = FancyBboxPatch(
            (x0, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.4,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, y, label, ha="center", va="center", fontsize=10.5, fontweight="bold")

        if i < n - 1:
            arrow = FancyArrowPatch(
                (x0 + box_w / 2, y - box_h / 2 - 0.03),
                (x0 + box_w / 2, y - 1 + box_h / 2 + 0.03),
                arrowstyle="-|>",
                mutation_scale=18,
                color="#333333",
                linewidth=1.4,
            )
            ax.add_patch(arrow)

    ax.set_title(
        "Architecture du pipeline IA - Prediction de progression coaching fitness",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Schema d'architecture sauvegarde -> {OUTPUT_PATH}")


if __name__ == "__main__":
    build_diagram()
