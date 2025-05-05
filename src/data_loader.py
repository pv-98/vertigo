import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    """Load raw spectral CSV into a pandas DataFrame."""
    return pd.read_csv(path)