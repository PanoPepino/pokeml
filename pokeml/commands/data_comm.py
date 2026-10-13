import typer

from rich.console import Console
from pathlib import Path

from pokeml.data.build import build_pkdx
from pokeml.data.acquire import acquire_full_pkdx
from pokeml.utils.utils_commands import CliUI

app = typer.Typer()
console = Console()

ui = CliUI()


@app.command('acquire', help='Download the Full Pokédex of existing Pokémon')
def main(
    output_path: Path = typer.Option(
        "datasets/pkdx_raw.csv",
        "--output-path",
        "-o",
        help="Where to save the acquired full Pokédex CSV",
    )
):
    """
    Download the full Pokédex and save it as csv.
    """
    acquire_full_pkdx(output_path, printer=typer.echo)


@app.command('clean', help='Clean a scrapped database for easier manipulation')
def clean(
    input_path: str = "datasets/pkdx_raw.csv",
    output_path_clean: str = "datasets/pkdx_clean.csv",
    output_path_minimal: str = "datasets/pkdx_min.csv",
):
    """
    Clean existing raw CSV offline, validate records and produce minimal CSV.
    """

    console.print('')
    ui.rule('PokéML Cleaning')
    ui.info(f'Modifying wrong entries, cleaning and simplifying raw Pokédex at [bold]{input_path}[/bold]')

    build_pkdx(input_path, output_path_clean, output_path_minimal)
    ui.success("Cleaning complete")
    ui.panel(
        f"Clean Pkdx: [bold]{output_path_clean}[/bold]\n"
        f"Minimal Pkdx: [bold]{output_path_minimal}[/bold]",
        title="[bold red] New Generated Pkdx [/bold red]",
    )

    console.print('')
