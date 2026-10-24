"""
============================================================================
 io_helpers.py, Shared Save/Load Helpers (CSV / JSON / TXT)
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Every script saves outputs through these three functions so the
           whole repo writes files identically, with loud prints.
 Usage   : from utils.io_helpers import save_csv, save_json, save_txt
============================================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _jsonable(obj):
    """Convert numpy/pandas scalars so json.dumps never chokes."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_csv(df: pd.DataFrame, path: Path, label: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [SAVED] CSV  {label or path.name:<28s} "
          f"{len(df):>7,} rows x {len(df.columns):>2} cols -> {path}")
    return path


def save_json(data, path: Path, label: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_jsonable))
    print(f"  [SAVED] JSON {label or path.name:<28s} -> {path}")
    return path


def save_txt(text: str, path: Path, label: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    n_lines = text.count("\n") + 1
    print(f"  [SAVED] TXT  {label or path.name:<28s} {n_lines:>7,} lines -> {path}")
    return path


def load_master(master_path: Path) -> pd.DataFrame:
    """Load the master analysis table with a loud, informative failure."""
    master_path = Path(master_path)
    if not master_path.exists():
        raise FileNotFoundError(
            f"Master table not found at {master_path}.\n"
            "  Run the pipeline first:\n"
            "    python data_processing/01_clean_and_merge.py\n"
            "    python data_processing/02_run_static_analysis.py\n"
            "    python data_processing/03_feature_engineering.py")
    df = pd.read_csv(master_path)
    print(f"  [LOAD ] master table: {len(df):,} rows x {len(df.columns)} cols "
          f"from {master_path}")
    return df
