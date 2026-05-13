import joblib
import typer
import pandas as pd

from rich.console import Console
from pathlib import Path
from pokeml.utils.utils_commands import CliUI
from pokeml.evaluation.eval import real_vs_predicted
from pokeml.pipeline.prepare import prepare_data_train
from pokeml.utils.utils_eda import df_to_table
from pokeml.features.feature_registry import get_feature_steps


app = typer.Typer()
console = Console()
ui = CliUI()


@app.command('residual', help='Plot the residuals for your trained models')
def plot_residual(
    input_path: str = typer.Option(
        "datasets/pkdx_min.csv",
        help="Path to the input dataset CSV."
    ),
    model_iter: str = typer.Option(
        "artifacts/models/yyyy_mm_dd_something",
        help="Base path to the trained run artifacts, without file extension."
    ),
    feat_mode: str = typer.Option(
        "none",
        help='Feature engineering mode: "none", "full", or "custom".'
    ),
    feat_steps: str = typer.Option(
        "",
        help='Comma-separated custom feature steps, used when feat_mode="custom".'
    ),
):
    console.print('')
    ui.rule("PokéML Evaluation")
    ui.info(f"Predicted vs Actual values for run: [bold]{model_iter}[/bold]")

    feat_eng_steps = get_feature_steps(mode=feat_mode, active_steps=feat_steps.split(","))

    prepared_data, _ = prepare_data_train(
        input_path,
        feat_eng_steps=feat_eng_steps
    )

    # ----------------------------------------------------------------
    # Load the classifier artifact saved during training (if present).
    # Enrichment is now delegated to real_vs_predicted() so each model
    # call receives a consistent, classifier-aware feature set.
    # ----------------------------------------------------------------
    classifier = None
    classifier_path = Path(f"{model_iter}_classifier.joblib")
    if classifier_path.exists():
        classifier = joblib.load(classifier_path)
        ui.info("BandClassifier loaded — features will be enriched before evaluation.")
    else:
        ui.info(
            f"[yellow]No classifier artifact found at {classifier_path}. "
            "Skipping band enrichment — ensure models were trained without a classifier.[/yellow]"
        )

    metrics_rows = []
    for model in ["cat_native", "cat_ordinal", "light_gbm"]:
        row = real_vs_predicted(
            f"{model_iter}_{model}",
            prepared_data,
            classifier=classifier,   # enrich inside real_vs_predicted
        )
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    out_dir = Path("artifacts/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    last = Path(model_iter).name
    metrics_df.to_csv(out_dir / f"metrics_data_{last}.csv", index=False)

    console.print(df_to_table(metrics_df, show_index=False), justify='center')
    ui.success("Plots complete")
    ui.panel(
        f"Residual plots: [bold]plots/evaluation/{last}_model_name.png[/bold]\n"
        f"Metrics CSV: [bold]artifacts/evaluation/metrics_data_{last}.csv[/bold]",
        title=f"[bold red] Residual information [/bold red]",
    )
    console.print('')
