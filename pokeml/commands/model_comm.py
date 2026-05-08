import joblib
import typer
import pandas as pd

from pathlib import Path
from rich.console import Console
from pokeml.models.predict import predict_all_models
from pokeml.models.tuning import tuning
from pokeml.models.train import train
from pokeml.features.preprocess import prepare_data_train, prepare_data_predict
from pokeml.utils.utils_eda import df_to_table
from pokeml.utils.utils_train import load_json
from pokeml.utils.utils_commands import CliUI
from pokeml.utils.utils_feat_eng import resolve_feat_steps
from pokeml.features.feature_registry import get_feature_steps

app = typer.Typer()
console = Console()

ui = CliUI()


@app.command('tune',
             help='Find the best suited set of parameters with RandomSearchCV. Extract best params.')
def tune_data(
        input_path: str = typer.Option("datasets/pkdx_min.csv", help="Path to the input dataset."),
        input_config: str = typer.Option('configs/tuning_easy.json',
                                         help="The set of configuration to look into the GridSearch"),
        iterations: int = typer.Option(1, help="Number of search iterations in Grid."),
        # Declare name of your json output (w/o extension). System will add later.
        output_name: str = typer.Option('artifacts/tuning/yyyy_mm_dd',
                                        help="Path to output. Do not declare extension! System does automatically."),
        feat_mode: str = typer.Option('none', help='Options ["none", "full", "custom"]'),
        feat_steps: str = typer.Option('', help='If custom -> Options ["initial_tag"]')  # Add more info here.
):
    console.print('')
    ui.rule("PokéML Tuning")
    ui.info(f"Preparing initial data from [bold]{input_path}[/bold]")

    #  Resolving feat steps
    feat_eng_steps = get_feature_steps(mode=feat_mode, active_steps=feat_steps.split(","))
    to_tune, _ = prepare_data_train(input_path,
                                    feat_eng_steps=feat_eng_steps)
    with console.status(
        f"[bold green]Tuning models[/bold green] ({iterations} iterations) ..."
    ):
        tuning(to_tune,
               my_grid=input_config,
               search_iter=iterations,
               output_name=output_name)

    ui.success("Tuning complete")
    ui.info(f"Relevant information for the best run:")
    df = pd.read_csv(Path(f'{output_name}_cv.csv'))
    console.print(df_to_table(df, parse_json_columns=['top_features', 'top_feature_weights']), justify='center')
    ui.panel(
        f"Best parameters: [bold]{output_name}_bp.json[/bold]\n"
        f"CV summary: [bold]{output_name}_cv.csv[/bold]",
        title=f"[bold red] Tuning information [/bold red]",
    )

    console.print('')


@app.command(
    "train",
    help="Based on parameters found with tune, train the model. Create training curves."
)
def train_data(
    input_json: str = typer.Option(
        "artifacts/tuning/yyyy_mm_dd_something_bp.json",
        help="Path to the best-parameters JSON produced by tune."
    ),
    input_path: str = typer.Option(
        "datasets/pkdx_min.csv",
        help="Path to the training dataset CSV."
    ),
    output_joblib: str = typer.Option(
        "artifacts/models/yyyy_mm_dd_something.joblib",
        help="Output path for the trained model artifact. Do not add .joblib ext. That is automatic."
    ),
    stop_loss: bool = typer.Option(
        True,
        help="Enable early stopping during training."
    ),
    early_stop: int = typer.Option(
        30,
        help="Early stopping rounds used when stop_loss is enabled."
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
    ui.rule('PokéML Training')
    ui.info(f"Preparing initial data from [bold]{input_path}[/bold]")
    ui.info(f"Tuning .json file: [bold]{input_json}[/bold]")

    feat_eng_steps = get_feature_steps(mode=feat_mode, active_steps=feat_steps.split(","))
    to_train, fe_state = prepare_data_train(input_path,
                                            feat_eng_steps=feat_eng_steps)
    params = load_json(input_json)

    info_stop = []
    if stop_loss:
        ui.info(f"Stop loss activated. Early stopping rounds at: {early_stop}")
        info_stop.append('w_stop')
        for key in params.keys():
            params[key]["early_stopping_rounds"] = early_stop
    else:
        info_stop.append('no_stop')

    with console.status(
        f"[bold green]Training models[/bold green] ..."
    ):

        train(to_train,
              params=params,
              output_name=output_joblib,
              fe_state=fe_state)

    # Defining path for out
    p = Path(output_joblib)
    out_dir = p.parent
    last = p.name

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = Path("plots/training")
    plot_dir.mkdir(parents=True, exist_ok=True)

    ui.success("Training complete")
    ui.panel(
        f"Trained models: [bold]{output_joblib}.joblib[/bold]\n"
        f"Training curves: [bold]plots/training/{last}_all_models_loss.png[/bold]",
        title=f"[bold red] Training information [/bold red]",
    )

    console.print('')


@app.command(
    "predict",
    help="Load a database of new Pokémon without defined BST and predict their stats."
)
def predict_data(
    input_run: str = typer.Option(
        "artifcats/models/yyyy_mm_dd_info",
        help="Base path to the trained run artifacts, without file extension."
    ),
    new_poke_data: str = typer.Option(
        "datasets/new_pokes.csv",
        help="Path to the CSV with new Pokémon to predict."
    ),
    output_preds: str = typer.Option(
        "artifacts/predictions/yyyy_mm_dd_name",
        help="Base output name for prediction files, without extension."
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

    feat_eng_steps = get_feature_steps(mode=feat_mode, active_steps=feat_steps.split(","))
    fe_state_path = Path(f"{input_run}_fe_state.joblib")
    fe_state = joblib.load(fe_state_path)
    to_predict = prepare_data_predict(new_poke_data,
                                      fe_state=fe_state,
                                      feat_eng_steps=feat_eng_steps)

    predict_all_models(run=input_run,
                       new_poke_data=to_predict,
                       output_preds=output_preds)
