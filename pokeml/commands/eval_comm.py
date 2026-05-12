import joblib
import typer
import pandas as pd

from rich.console import Console
from pathlib import Path
from pokeml.utils.utils_commands import CliUI
from pokeml.evaluation.eval import real_vs_predicted
from pokeml.pipeline.prepare import prepare_data_train
from pokeml.utils.utils_eda import df_to_table
from pokeml.utils.utils_feat_eng import resolve_feat_steps


app = typer.Typer()
console = Console()
ui = CliUI()


@app.command('residual', help='Plot the residuals for your trained models')
def plot_residual(
    input_path: str = 'datasets/pkdx_min.csv',
    model_iter: str = None,
    feat_mode: str = 'none',
    feat_steps: str = '',
):
    console.print('')
    ui.rule("PokéML Evaluation")
    ui.info(f"Predicted vs Actual values for run: [bold]{model_iter}_models.joblib[/bold]")

    feat_eng_steps = resolve_feat_steps(feat_mode, feat_steps)

    prepared_data, fe_state = prepare_data_train(
        input_path,
        feat_eng_steps=feat_eng_steps
    )

    # ----------------------------------------------------------------
    # Load the classifier artifact saved during training and enrich
    # the prepared data so feature schemas match the trained models.
    # Without this step, models trained with pred_band/proba_* columns
    # will raise a missing-column error during evaluation.
    # ----------------------------------------------------------------

    classifier_path = Path(f"{model_iter}_classifier.joblib")
    if classifier_path.exists():
        classifier = joblib.load(classifier_path)
        enriched_data = {}
        for name, (X_tr, X_te, y_tr, y_te, cats) in prepared_data.items():
            X_tr_enc = classifier.enrich(X_tr)
            X_te_enc = classifier.enrich(X_te)
            if "pred_band" not in cats:
                cats = cats + ["pred_band"]
            enriched_data[name] = (X_tr_enc, X_te_enc, y_tr, y_te, cats)
        prepared_data = enriched_data
    else:
        ui.info(
            f"[yellow]No classifier artifact found at {classifier_path}. "
            "Skipping band enrichment — ensure models were trained without a classifier.[/yellow]"
        )

    metrics_rows = []
    for model in ["cat_native", "cat_ordinal", "light_gbm"]:
        row = real_vs_predicted(f"{model_iter}_{model}", prepared_data)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    out_dir = Path("artifacts/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    last = Path(model_iter).name
    metrics_df.to_csv(out_dir / f"metrics_data_{last}.csv", index=False)

    console.print(df_to_table(metrics_df, show_index=False), justify='center')
    ui.success("Plots complete")
    ui.panel(
        f"Residual plots: [bold]plots/evaluation/{last}_model_name.png[/bold]",
        title=f"[bold red] Residual information [/bold red]",
    )
    console.print('')
