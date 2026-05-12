import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path


from pokeml.visualisation.constants_plot import ALPHA, MAIN_COLOR
from pokeml.visualisation.coloring import TYPE_COLORS
from pokeml.data.eda_types import compare_type_ordering


def type_order_deviation_plot(
    stage_list,
    baseline,
    df,
    min_count,
    rarity_list=("regular",),
    plot_path: Path = Path("plots/eda/")
):
    if isinstance(stage_list, str):
        stage_list = [stage_list]
    if isinstance(rarity_list, str):
        rarity_list = [rarity_list]

    if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
        raise ValueError("baseline must be [regular_baseline, legendary_baseline]")

    regular_baseline, legendary_baseline = baseline

    plot_specs = []
    for stage in stage_list:
        for rarity in rarity_list:
            if rarity == "regular":
                plot_specs.append((stage, rarity))
    plot_specs.append((None, "legendary"))

    n_plots = len(plot_specs)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for ax in axes.flat:
        ax.set_visible(False)

    if n_plots == 7 and nrows == 3 and ncols == 3:
        positions = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 1),
        ]
    else:
        positions = [(i // ncols, i % ncols) for i in range(n_plots)]

    for i, (stage_value, rarity) in enumerate(plot_specs):
        row, col = positions[i]
        ax = axes[row, col]
        ax.set_visible(True)

        if rarity == "legendary":
            comparison = compare_type_ordering(
                stage=None,
                baseline=legendary_baseline,
                df=df,
                rarity="legendary",
                min_count=min_count,
            )
            title_stage = "All stages"
            current_baseline = float(legendary_baseline)
        else:
            comparison = compare_type_ordering(
                stage=stage_value,
                baseline=regular_baseline,
                df=df,
                rarity="regular",
                min_count=min_count,
            )
            title_stage = stage_value
            current_baseline = float(regular_baseline.loc[stage_value])

        if comparison is None or comparison.empty:
            ax.set_title(f"{title_stage} | {rarity} (no data)")
            ax.axis("off")
            continue

        comparison = comparison.dropna(subset=["dev_z_t1", "dev_z_t2"]).copy()

        if comparison.empty:
            ax.set_title(f"{title_stage} | {rarity} (no usable z-scores)")
            ax.axis("off")
            continue

        for _, row_data in comparison.iterrows():
            type_name = row_data["type_2"]
            x = row_data["dev_z_t1"]
            y = row_data["dev_z_t2"]
            size = max(20, min(row_data["n_t1"], row_data["n_t2"]) * 100)

            ax.scatter(
                x, y,
                s=size,
                color=TYPE_COLORS.get(type_name, "gray"),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7
            )

        lims = [
            min(comparison["dev_z_t1"].min(), comparison["dev_z_t2"].min()) - 0.2,
            max(comparison["dev_z_t1"].max(), comparison["dev_z_t2"].max()) + 0.2,
        ]

        ax.plot(lims, lims, "b--", linewidth=1, alpha=ALPHA, label="No ordering effect (y=x)")
        ax.axhline(0, color=MAIN_COLOR, linestyle=":", linewidth=1)
        ax.axvline(0, color=MAIN_COLOR, linestyle=":", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(alpha=0.3)

        ax.set_xlabel("Deviation for type_1 (dev_z_t1)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Deviation for type_2 (dev_z_t2)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Type Ordering Effect on BST (Z-score)\n"
            f"{title_stage} | {rarity} | baseline={current_baseline:.1f}\n",
            fontsize=13,
            fontweight="bold"
        )
        ax.legend(fontsize=10)

    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)

    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(plot_path) / "type_ordering.png", dpi=500, bbox_inches="tight")
    plt.show()
