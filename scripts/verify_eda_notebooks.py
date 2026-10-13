"""Execute six EDA notebooks, validate data immutability and save visible outputs.

Run from project root with project Python. --in-place refreshes reviewable source
notebooks; otherwise executed copies go to artifacts/eda/executed/.
"""

import argparse
import hashlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = ('00_data_preparation', '01_stage_analysis', '02_power_creep', '03_typing_analysis',
             '06_shape_analysis', '07_capture_rate_analysis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--notebook', choices=NOTEBOOKS, action='append',
                        help='Restrict verification to selected notebook(s)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    datasets = [root / 'datasets' / name for name in ('pkdx_raw.csv', 'pkdx_clean.csv', 'pkdx_min.csv', 'new_pokes.csv')]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in datasets}
    for name in args.notebook or NOTEBOOKS:
        path = root / 'notebooks' / f'{name}.ipynb'
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=240, kernel_name='pokemon',
                       resources={'metadata': {'path': str(root)}}).execute()
        notebook.metadata['pokeml_dataset_sha256'] = before[root / 'datasets/pkdx_min.csv']
        target = path if args.in_place else root / 'artifacts/eda/executed' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        print(f'Executed {name}', flush=True)
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != value for p, value in before.items()):
            raise RuntimeError('EDA changed dataset bytes')
    print('Selected notebooks passed; dataset hashes unchanged', flush=True)


if __name__ == '__main__':
    main()
