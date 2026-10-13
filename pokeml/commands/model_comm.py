"""CLI adapters for forward training, release prediction and diagnostics."""
import typer
import pandas as pd
from pathlib import Path
from typing import Literal
from rich.console import Console
from pokeml.reporting.tables import df_to_table

app = typer.Typer()
console = Console()


@app.command("workflow", help="Compare baseline/direct/correction, refit selected model and save release.")
def baseline_workflow(
    input_path: str = typer.Option("datasets/pkdx_min.csv", help="Existing corrected local snapshot."),
    prediction_path: str = typer.Option("datasets/new_pokes.csv", help="Announcement CSV with supplied stage."),
    output_root: str = typer.Option("artifacts/workflow", help="New run directories are created here."),
    run_name: str = typer.Option(None, help="Optional unique run name."),
    tune_iterations: int = typer.Option(0, min=0, help="Number of model settings to try; zero skips tuning."),
    selection_scope: Literal["ordinary", "all"] = typer.Option("ordinary", help="Choose using ordinary Pokémon, or include special groups."),
):
    """Package owns training; CLI only passes inputs and displays reviewable results."""
    from pokeml.pipeline.workflow import WorkflowConfig, run_workflow
    kwargs = dict(training_csv=input_path, prediction_csv=prediction_path,
                  output_root=output_root, tune_iterations=tune_iterations, selection_scope=selection_scope)
    if run_name is not None:
        kwargs["run_name"] = run_name
    result = run_workflow(WorkflowConfig(**kwargs))
    console.print(df_to_table(result.evaluation_summary))
    console.print(f"Selection group: {result.selection_scope}")
    console.print(f"Selected: {result.selected_model}")
    console.print(f"Release: {result.release_directory}")
    console.print(df_to_table(result.predictions))


@app.command("predict-release", help="Predict from saved release; no fitting or downloads.")
def predict_release(
    release_directory: Path = typer.Option(..., help="Directory containing manifest and predictor."),
    input_path: Path = typer.Option("datasets/new_pokes.csv", help="CSV with announcement attributes."),
    output_path: Path = typer.Option("artifacts/predictions/release_predictions.csv", help="New prediction CSV."),
):
    """The same predictor serves CSV commands, notebooks and future APIs."""
    from pokeml.models.deployment import PokemonPredictor
    if output_path.exists() or output_path.resolve() == input_path.resolve():
        raise typer.BadParameter("Choose a new output path; existing files are never overwritten")
    predictions = PokemonPredictor.load(release_directory).predict(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    console.print(df_to_table(predictions))


@app.command("training-curves", help="Diagnose frozen run across four folds; no new tuning or release.")
def training_curve_diagnostics(
    run_directory: Path = typer.Option(..., help="Completed parent workflow directory."),
    output_directory: Path = typer.Option(..., help="New diagnostics directory outside parent run."),
    checkpoints: int = typer.Option(200, min=2, help="Maximum sampled boosting-round prefixes."),
):
    from pokeml.evaluation.curves import diagnose_run
    summary = diagnose_run(run_directory, output_directory, checkpoints)
    display = summary[["model", "generation", "final_train_MAE", "final_validation_MAE", "MAE_deterioration"]].rename(
        columns={"generation": "gen", "final_train_MAE": "train MAE",
                 "final_validation_MAE": "valid MAE", "MAE_deterioration": "rise"})
    console.print(df_to_table(display.round(3)))
    console.print(f"Curves and full summary: {output_directory}")
