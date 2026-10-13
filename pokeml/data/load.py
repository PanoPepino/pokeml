import pandas as pd

from pathlib import Path


def load_data(path: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load Pokemon data from CSV or copy an in-memory DataFrame."""
    if isinstance(path, pd.DataFrame):
        return path.copy()
    path = Path(path)
    df = pd.read_csv(path)
    return df
