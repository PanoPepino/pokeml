import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pokeml.visualisation.constants_plot import ALPHA, MAIN_COLOR, SECOND_COLOR, LINE_COLOR


def get_bst_vs_stage_plot(mean_stage_cat,
                          baseline,
                          nrows=3,
                          ncols=3,
                          plot_path: Path = 'plots/eda/'):
    """
    This function will return 7 plots (one for each pkmn stage + legendary) where the mean and median bst for a given stage and generation will be displayed.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 11.5))
    axes = np.array(axes).reshape(-1)

    positions = [0, 1, 2, 3, 4, 5, 7] if len(mean_stage_cat) == 7 and len(axes) == 9 else list(range(len(mean_stage_cat)))
    if len(positions) > len(axes):
        raise ValueError('Plot grid too small for supplied groups')
    for ax in axes:
        ax.set_visible(False)

    for idx, (category, gen_data) in enumerate(mean_stage_cat.items()):
        ax = axes[positions[idx]]
        ax.set_visible(True)
        if not gen_data:
            ax.set_title(f'{category}: no observations')
            ax.axis('off')
            continue

        gens = np.array(list(gen_data.keys()))
        means = np.array([v["mean"] for v in gen_data.values()])
        medians = np.array([v["median"] for v in gen_data.values()])
        counts = np.array([v["count"] for v in gen_data.values()])

        x = np.arange(len(gens))

        # Lines suit zoomed trend axes; bar height otherwise exaggerates changes.
        ax.plot(x, means, 'o-', label="mean", alpha=ALPHA, color=MAIN_COLOR)
        ax.plot(x, medians, 's-', label="median", color=SECOND_COLOR)

        for xi, mean, count in zip(x, means, counts):
            ax.annotate(f'n={count}', (xi, mean), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)

        stage_median_of_medians = float(baseline[category] if isinstance(baseline, dict) else baseline[idx])
        low, high = min(means.min(), medians.min(), stage_median_of_medians), max(means.max(), medians.max(), stage_median_of_medians)
        margin = max(15, .15 * (high - low))

        ax.set_title(category.capitalize(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Total Stats")
        ax.axhline(
            stage_median_of_medians,
            color=LINE_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=f"Pooled median: {stage_median_of_medians:.1f}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(gens)
        ax.set_xlim(-.7, len(gens) - .3)
        ax.set_ylim(low - margin, high + margin)
        ax.grid(axis='y', alpha=.15)
        ax.legend(fontsize=8)

    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(Path(plot_path) / 'bst_vs_stage.png', dpi=180, bbox_inches='tight')
    return fig, axes
