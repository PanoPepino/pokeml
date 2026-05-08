import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pokeml.utils.constants_plot import ALPHA, MAIN_COLOR
from .coloring import TYPE_COLORS


# Here, together with the help of AI, I create a func to display the total_stats median deviation for each type construction and stage.


def type_deviations_single_plot(
    stage,
    rarity="regular",
    baseline=None,
    df=None,
    ax=None,
):
    """
    Plot type total_stats deviations by construction for a given stage and rarity.


    Parameters:
    -----------
    stage : str
        Stage to plot (e.g., 's2c2', 's3c3')
    rarity : str
        rarity to filter (default: 'regular')
    baseline: DataFrame
        The dataframe withe the information counting all pokemon with monotype, type as first, type as second to compute respective medians and deviations from baseline. MUST BE FLAT!
    df: DataFrame
        The dataframe withe the information counting all pokemon with monotype, type as first, type as second to compute respective medians and deviations from baseline. MUST BE FLAT!
    ax : matplotlib.axes.Axes
        Axis where the plot will be drawn.
    """
    if baseline is None or df is None:
        raise ValueError("baseline and df must be provided")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    # Get overall median of a given stage
    my_baseline = baseline.loc[stage] if rarity == "regular" else float(baseline)

    # Filter once for the requested stage and rarity
    sub = df[
        (df["stage"] == stage) &
        (df["rarity"] == rarity)
    ].copy()

    if sub.empty:
        ax.set_axis_off()
        ax.set_title(f"Stage: {stage} | Rarity: {rarity} (no data)")
        return fig, ax

    # Pivot data for deviations. It basically rearrange previous tables for easy column comparison/finding.
    pivot = sub.pivot_table(
        index="type",
        columns="construction",
        values="deviation",
        aggfunc="first"
    )

    # Pivot data for counts.
    counts = sub.pivot_table(
        index="type",
        columns="construction",
        values="count",
        aggfunc="first"
    ).fillna(0).astype(int)

    # Check which constructions are available
    available_constructions = [c for c in ["mono", "dual_t1", "dual_t2"] if c in pivot.columns]

    if len(available_constructions) == 0:
        ax.set_axis_off()
        ax.set_title(f"Stage: {stage} | Rarity: {rarity} (no data)")
        return fig, ax

    pivot = pivot[available_constructions]
    counts = counts.reindex(index=pivot.index, columns=available_constructions)

    # Bar positions
    x = np.arange(len(pivot))
    width = 0.8 / len(available_constructions)

    # Construction styles
    construction_styles = {
        "mono": ("", 1.0),
        "dual_t1": ("///", 0.85),
        "dual_t2": ("...", 0.85)
    }

    for i, constr in enumerate(available_constructions):
        pattern, alpha = construction_styles[constr]
        values = pivot[constr].values
        colors = [TYPE_COLORS.get(type_name, "gray") for type_name in pivot.index]

        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=constr,
            color=colors,
            alpha=alpha,
            hatch=pattern,
            edgecolor="indigo",
            linewidth=0.8
        )

        # Add count labels
        for j, (bar, type_name) in enumerate(zip(bars, pivot.index)):
            if pd.notna(values[j]):
                count_val = counts.loc[type_name, constr]
                y_pos = values[j] + (1 if values[j] > 0 else -2)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{count_val}",
                    ha="center",
                    va="bottom" if values[j] > 0 else "top",
                    fontsize=7,
                    fontweight="bold"
                )

    # my_Baseline line
    ax.axhline(0, color=MAIN_COLOR, linestyle="--", linewidth=2, label=f"Median ({my_baseline})")

    # Formatting
    ax.set_ylabel("Deviation from stage median", fontsize=12, fontweight="bold")
    ax.set_xlabel("Type", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Stage: {stage} | Rarity: {rarity} | Median: {my_baseline}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xticks(
        x + width * (len(available_constructions) - 1) / 2,
        labels=[t.capitalize() for t in pivot.index],
        rotation=45,
        ha="right"
    )
    ax.legend(title="Construction", loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=ALPHA, linestyle="--")

    return fig, ax


def type_deviations_plot(
    stage_list,
    rarity="regular",
    baseline=None,
    df=None,
    plot_path: Path = Path("plots/eda/"),
    ncols=2,
):
    if isinstance(stage_list, str):
        stage_list = [stage_list]

    if baseline is None or df is None:
        raise ValueError("baseline and df must be provided")

    n_plots = len(stage_list)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15 * ncols / 2, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    for i, stage in enumerate(stage_list):
        type_deviations_single_plot(
            stage=stage,
            rarity=rarity,
            baseline=baseline,
            df=df,
            ax=axes[i],
        )

    plt.tight_layout()

    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path / f"type_deviations_{rarity}.png", dpi=500, bbox_inches="tight")
    plt.show()
