"""Read and validate the minimal analysis snapshot without rebuilding datasets."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from pokeml.data.validation import validate_pkdx


def load_analysis_snapshot(path='datasets/pkdx_min.csv'):
    """Check manifest hash and schema before notebook exploration."""
    path = Path(path)
    manifest = json.loads(path.with_name('pkdx_manifest.json').read_text())
    expected = next((digest for filename, digest in manifest['files'].items()
                     if Path(filename).resolve() == path.resolve()), None)
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Analysis snapshot differs from build manifest; rebuild explicitly')
    frame = pd.read_csv(path)
    validate_pkdx(frame)
    return frame
