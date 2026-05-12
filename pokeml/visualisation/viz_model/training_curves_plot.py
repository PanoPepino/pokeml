from pathlib import Path
import matplotlib.pyplot as plt

from pokeml.visualisation.viz_model.loss_plot import loss_plot


def training_curves_plot(last, evals):
    plot_dir = Path("plots/training")
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(evals)
    fig, axs = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)
    axs = axs.ravel()

    for ax, item in zip(axs, evals):
        loss_plot(
            name_model=item["model_name"],
            evals=item["evals"],
            ax=ax
        )

    fig.suptitle(f"Training / Validation for {last} Run", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_dir / f"{last}_all_models_loss.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
