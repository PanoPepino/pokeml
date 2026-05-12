# pokeml/visualisation/confusion_plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pokeml.constants import FINAL_LABELS


def confusion_plot(y_true, y_pred, out_path):
    """
    Save a normalized confusion matrix plus per-class recall bar plot.
    """
    order = FINAL_LABELS

    cm = pd.crosstab(y_true, y_pred).reindex(
        index=order,
        columns=order,
        fill_value=0,
    )

    cm_norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    recall = np.diag(cm_norm.values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im = axes[0].imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_yticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=45, ha="right")
    axes[0].set_yticklabels(order)
    axes[0].set_title("Normalized Confusion Matrix")

    for i in range(len(order)):
        for j in range(len(order)):
            axes[0].text(
                j, i,
                f"{cm_norm.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(order, recall, color="#5DA5DA", edgecolor="black", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-class Recall")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
