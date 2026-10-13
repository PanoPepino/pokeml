"""Shared boxplot implementation for preparation overview and stage analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pokeml.eda.stages import STAGE_ORDER, stage_outliers


def plot_stage_buckets(df, rarity='regular', jitter=True):
    """Show median/IQR, Tukey whiskers and named-table-compatible outliers.

    Jitter changes horizontal position only. Seed fixes layout across runs.
    Returns figure/axes; caller decides whether and where to save.
    """
    work = df[df.rarity.eq(rarity)]
    stages = [stage for stage in STAGE_ORDER if work.stage.eq(stage).any()]
    if not stages:
        raise ValueError(f'No stage records for rarity {rarity}')
    values = [work.loc[work.stage.eq(stage), 'total_stats'] for stage in stages]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.boxplot(values, tick_labels=[f'{s}\nn={len(v)}' for s, v in zip(stages, values)],
               patch_artist=True, showfliers=False,
               boxprops={'facecolor': '#e6edf4', 'edgecolor': '#40566a'},
               medianprops={'color': '#d97706', 'linewidth': 2},
               whiskerprops={'color': '#40566a'}, capprops={'color': '#40566a'})
    outliers = stage_outliers(work, rarity)
    rng = np.random.default_rng(1)
    for i, stage in enumerate(stages, 1):
        points = outliers.loc[outliers.stage.eq(stage), 'total_stats'].to_numpy()
        x = np.full(len(points), float(i))
        if jitter:
            x += rng.uniform(-.15, .15, len(points))
        ax.scatter(x, points, s=28, facecolors='white', edgecolors='#40566a', zorder=3)
    ax.set(xlabel='Current evolution stage', ylabel='BST',
           title=f'{rarity.capitalize()} Pokémon: stage baseline and remaining variation')
    ax.grid(axis='y', alpha=.18)
    ax.legend(handles=[
        Line2D([], [], color='#d97706', linewidth=2, label='Median'),
        Patch(facecolor='#e6edf4', edgecolor='#40566a', label='Middle 50% (IQR)'),
        Line2D([], [], color='#40566a', marker='|', label='Whiskers: observations within 1.5 IQR'),
        Line2D([], [], color='#40566a', marker='o', markerfacecolor='white', linestyle='', label='Outside whiskers'),
    ], loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax
