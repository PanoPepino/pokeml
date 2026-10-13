"""Plots for descriptive composition and original generation-trend hypotheses."""

import matplotlib.pyplot as plt


def plot_composition(comparison, proportions):
    """Actual and fixed-composition means above observed stage proportions."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    for column, label in [('actual_mean', 'Observed mean'), ('fixed_composition_mean', 'Mean at constant stage weights')]:
        axes[0].plot(comparison.generation, comparison[column], 'o-', label=label)
    axes[0].set(title='Regular Pokémon: roster composition versus within-stage change', ylabel='BST mean')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.2)
    proportions.plot.bar(stacked=True, ax=axes[1], width=.8, colormap='viridis')
    axes[1].set(ylabel='Share of regular roster', xlabel='Introduction generation')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Stage', bbox_to_anchor=(1.01, 1), fontsize=8)
    return fig, axes


def plot_polynomial_hypotheses(tables):
    """Mean/median fits are descriptive; line style distinguishes fitted values."""
    fig, axes = plt.subplots(1, len(tables), figsize=(12, 4.5), squeeze=False, layout='constrained')
    for ax, (label, table) in zip(axes.flat, tables.items()):
        for statistic, color in [('mean', '#64748b'), ('median', '#2563eb')]:
            ax.plot(table.generation, table[statistic], 'o', color=color, label=f'Observed {statistic}')
            ax.plot(table.generation, table[f'{statistic}_fit'], '--', color=color, label=f'Fitted {statistic}')
        ax.set(title=label, xlabel='Generation', ylabel='BST')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    return fig, axes
