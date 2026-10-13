"""Execute active workflow notebook, preserving dataset bytes and visible outputs."""
import argparse
from hashlib import sha256
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=("04_workflow", "05_training_diagnostics"), default="04_workflow")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [p for p in (root / "datasets").rglob("*") if p.is_file()]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in paths}
    path = root / "notebooks" / f"{args.notebook}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Uncached 60-trial tuning can take longer than ten minutes.
    NotebookClient(notebook, timeout=3600, kernel_name="pokemon",
                   resources={"metadata": {"path": str(root)}}).execute()
    if any(sha256(p.read_bytes()).hexdigest() != digest for p, digest in before.items()):
        raise RuntimeError("Workflow changed dataset bytes")
    notebook.metadata["pokeml_dataset_sha256"] = before[root / "datasets/pkdx_min.csv"]
    nbformat.write(notebook, path)
    print(f"Executed {args.notebook}; {len(paths)} dataset files unchanged", flush=True)


if __name__ == "__main__":
    main()
