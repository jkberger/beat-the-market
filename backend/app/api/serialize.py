import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd


def clean(obj):
    """Make a value JSON-safe: NaN/inf -> None, numpy -> python, dates -> ISO strings."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.date().isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if is_dataclass(obj):
        return clean(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def series_points(s: pd.Series) -> list[dict]:
    return [{"date": d.date().isoformat(), "value": clean(v)} for d, v in s.items()]
