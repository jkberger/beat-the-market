import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.data.providers import PriceProvider, SymbolNotFound, slice_bars


class CachedPriceProvider:
    """Caches each symbol's full daily history as Parquet; refreshes when stale."""

    def __init__(
        self, inner: PriceProvider, cache_dir: Path, max_age: timedelta = timedelta(days=1)
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.parquet"

    def get_daily_bars(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        path = self._path(symbol)
        fresh = path.exists() and time.time() - path.stat().st_mtime < self.max_age.total_seconds()
        if fresh:
            df = pd.read_parquet(path)
        else:
            try:
                df = self.inner.get_daily_bars(symbol)
                df.to_parquet(path)
            except SymbolNotFound:
                raise
            except Exception:
                if not path.exists():
                    raise
                df = pd.read_parquet(path)
        return slice_bars(df, start, end)
