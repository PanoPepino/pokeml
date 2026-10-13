"""Run-local diagnostics for baseline references, residuals and paired errors."""

from pokeml.artifacts import MODEL_NAMES
from pathlib import Path
from pokeml.evaluation.forward import pooled_summary
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _subplots(nrows=1, ncols=1, figsize=(10, 5)):
    # File rendering uses Agg without changing notebook's interactive backend.
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(nrows, ncols)


def workflow_plots(directory, development, validation, selected):
    directory = Path(directory) / "plots"
    directory.mkdir(exist_ok=True)
    primary = validation.loc[validation.population == "selected_forms"]
    summary = pooled_summary(development).query("population == 'selected_forms'").set_index("model").MAE.sort_values()
    fig, ax = _subplots(figsize=(10, 5))
    summary.plot.barh(ax=ax, color="#5DA5DA")
    ax.set(xlabel="MAE (BST points)", ylabel="", title="Development: pooled Pokémon MAE over generations 6–9")
    fig.tight_layout()
    fig.savefig(directory / "development_comparison.png", dpi=150)
    fig.clear()
    # Every model sees the identical cohort in the same row order.
    fig, axes = _subplots(1, 2, figsize=(13, 5))
    baseline = primary.loc[primary.model == "stage_median"]
    baseline.actual_correction_bst.plot.hist(bins=30, ax=axes[0], color="#5DA5DA")
    axes[0].set(title="Gen 9: actual BST − training baseline", xlabel="BST points", ylabel="Pokémon count")
    for variant in MODEL_NAMES:
        direct = primary.loc[primary.model == f"direct_{variant}", "absolute_error"].to_numpy()
        corrected = primary.loc[primary.model == f"correction_{variant}", "absolute_error"].to_numpy()
        if len(direct):
            axes[1].hist(direct - corrected, bins=25, alpha=0.45, label=variant)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(title="Paired improvement: positive favours correction", xlabel="Direct |error| − correction |error|", ylabel="Pokémon count")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / "correction_diagnostics.png", dpi=150)
    fig.clear()
    chosen = primary.loc[primary.model == selected]
    fig, axes = _subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(chosen.actual_bst, chosen.pred_bst, alpha=0.6)
    limits = (min(chosen.actual_bst.min(), chosen.pred_bst.min()),
              max(chosen.actual_bst.max(), chosen.pred_bst.max()))
    axes[0].plot(limits, limits, "k--")
    axes[0].set(title=f"Gen 9 selection fold: {selected}", xlabel="Actual BST", ylabel="Predicted BST")
    stages = chosen.groupby("stage", observed=True).agg(MAE=("absolute_error", "mean"), n=("name", "size"))
    stages.MAE.plot.bar(ax=axes[1], color="#5DA5DA")
    axes[1].set_xticks(range(len(stages)), [f"{s}\n(n={int(row.n)})" for s, row in stages.iterrows()], rotation=0)
    axes[1].set(title="Stage errors and support", xlabel="", ylabel="MAE (BST points)")
    fig.tight_layout()
    fig.savefig(directory / "selected_validation.png", dpi=150)
    fig.clear()
